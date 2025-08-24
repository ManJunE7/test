# app.py
# ---------------------------------------------------------
# 천안 DRT - 맞춤형 AI기반 스마트 교통 가이드 (통합)
# - 기존 DRT 정류장: 천안콜 버스 정류장(v250730)_4326.shp
# - 신규/후보 정류장: new_new_drt_full_utf8.(shp/gpkg/geojson)
# - Mapbox Directions로 실도로 라우팅
# - '덩어리' 폴리곤으로 커버리지 비교(반경 버퍼 병합 + 면적 계산)
# ---------------------------------------------------------

import os, math, re
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import requests
import streamlit as st
import folium
from folium.plugins import MarkerCluster
from folium.features import DivIcon
from shapely.geometry import Point, MultiPolygon, Polygon
from shapely.ops import unary_union
from streamlit_folium import st_folium

# ===================== UI/스타일 =====================
APP_TITLE = "천안 DRT - 맞춤형 AI기반 스마트 교통 가이드"
LOGO_URL  = "https://raw.githubusercontent.com/JeongWon4034/cheongju/main/cheongpung_logo.png"

st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Noto Sans KR', -apple-system, BlinkMacSystemFont, sans-serif; }
.header-container{display:flex;align-items:center;justify-content:center;gap:16px;margin-bottom:14px;padding:8px 0;}
.logo-image{width:70px;height:70px;object-fit:contain}
.main-title{font-size:2rem;font-weight:800;color:#202124;letter-spacing:-0.5px;margin:0}
.title-underline{width:100%;height:3px;background:linear-gradient(90deg,#4285f4,#34a853);margin:0 auto 14px;border-radius:2px;}
.section-header{font-size:1.02rem;font-weight:800;color:#1f2937;margin-bottom:8px;padding-bottom:8px;border-bottom:2px solid #f3f4f6}
.stButton > button{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:#fff;border:none;border-radius:10px;padding:9px 16px;font-size:.9rem;font-weight:700;box-shadow:0 4px 8px rgba(102,126,234,.3)}
.stButton > button:hover{transform:translateY(-1px);box-shadow:0 6px 14px rgba(102,126,234,.4)}
.visit-card{display:flex;align-items:center;gap:10px;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:#fff;border-radius:12px;padding:9px 12px;margin-bottom:8px;box-shadow:0 2px 4px rgba(102,126,234,.3)}
.visit-num{background:#fff;color:#667eea;width:24px;height:24px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:800;font-size:.8rem}
.empty{color:#9ca3af;background:linear-gradient(135deg,#ffecd2 0%,#fcb69f 100%);border-radius:12px;padding:20px 14px;text-align:center}
.badge{display:inline-block;padding:2px 8px;border-radius:999px;font-size:.78rem;font-weight:700}
.badge-red{background:#fee2e2;color:#b91c1c}
.badge-purple{background:#efe5ff;color:#6d28d9}
.badge-blue{background:#dbeafe;color:#1e40af}
.note{font-size:.85rem;color:#6b7280;margin-top:.25rem}
</style>
""", unsafe_allow_html=True)

st.markdown(
    f"""
    <div class="header-container">
        <img src="{LOGO_URL}" alt="앱 로고" class="logo-image" />
        <div class="main-title">{APP_TITLE}</div>
    </div>
    <div class="title-underline"></div>
    """,
    unsafe_allow_html=True
)

# ===================== 경로/상수 =====================
EXISTING_SHP = "천안콜 버스 정류장(v250730)_4326.shp"   # 기존 DRT 정류장
DATA_STEM    = "new_new_drt_full_utf8"                 # 후보/신규 정류장 파일명 앞부분
PALETTE = ["#e74c3c","#8e44ad","#3498db","#e67e22","#16a085","#2ecc71","#1abc9c","#d35400"]

MAPBOX_TOKEN = os.getenv("MAPBOX_TOKEN", "")
if not MAPBOX_TOKEN:
    # 직접 문자열로 넣어도 됨
    MAPBOX_TOKEN = "pk.eyJ1IjoiZ3VyMDUxMDgiLCJhIjoiY21lbWppYjByMDV2ajJqcjQyYXUxdzY3byJ9.yLBRJK_Ib6W3p9f16YlIKQ"

# ===================== 공통 유틸 =====================
def haversine(xy1, xy2):
    R=6371000.0
    lon1,lat1,lon2,lat2 = map(np.radians,[xy1[0],xy1[1],xy2[0],xy2[1]])
    dlon=lon2-lon1; dlat=lat2-lat1
    a=np.sin(dlat/2)**2+np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

def _read_utf8_shp(path: Path) -> gpd.GeoDataFrame:
    # 여러 인코딩 시도
    try:
        from pyogrio import read_dataframe as pio
        try:
            g = pio(path, encoding="utf-8")
        except Exception:
            g = pio(path, encoding="cp949")
        return gpd.GeoDataFrame(g, geometry="geometry", crs=getattr(g, "crs", None))
    except Exception:
        pass
    for enc in ("utf-8", "cp949"):
        try:
            return gpd.read_file(path, encoding=enc)
        except Exception:
            continue
    os.environ["SHAPE_ENCODING"] = "UTF-8"
    return gpd.read_file(path, engine="fiona")

def _open_any(stem: str) -> gpd.GeoDataFrame:
    for ext in (".shp",".gpkg",".geojson"):
        p = Path(f"./{stem}{ext}")
        if p.exists():
            g = _read_utf8_shp(p) if ext==".shp" else gpd.read_file(p)
            try:
                if g.crs and g.crs.to_epsg()!=4326:
                    g = g.to_crs(epsg=4326)
            except Exception:
                pass
            # 포인트화(혹시 폴리곤/라인일 경우 대표점)
            if not g.geom_type.astype(str).str.contains("Point",case=False,na=False).any():
                g = g.copy(); g["geometry"]=g.geometry.representative_point()
            return g
    st.error(f"'{stem}.shp/.gpkg/.geojson' 파일이 필요합니다."); st.stop()

# ===================== 데이터 로드 =====================
@st.cache_data
def load_existing_candidates():
    # 기존 DRT (천안콜 버스 정류장)
    p = Path(EXISTING_SHP)
    if not p.exists():
        st.error(f"기존 정류장 파일이 없습니다: {EXISTING_SHP}")
        st.stop()
    existing = _read_utf8_shp(p)
    if existing.crs is None or existing.crs.to_epsg()!=4326:
        try:
            existing = existing.to_crs(epsg=4326)
        except Exception:
            st.error("기존 정류장 좌표계 변환 실패"); st.stop()
    # 대표 이름 컬럼 정리
    name_col = None
    for cand in ["name","이름","정류장명","정류소명","jibun"]:
        if cand in existing.columns:
            name_col = cand; break
    if name_col is None:
        existing["name"] = existing.index.astype(str)
    else:
        existing["name"] = existing[name_col].astype(str)

    existing = existing[["name","geometry"]].copy()

    # 후보/신규 정류장
    cand = _open_any(DATA_STEM)
    if "name" not in cand.columns:
        if "jibun" in cand.columns:
            cand["name"] = cand["jibun"].astype(str)
        else:
            cand["name"] = cand.index.astype(str)
    cand = cand[["name","geometry"]].copy()

    # 중심점
    all_pts = pd.concat([existing[["geometry"]], cand[["geometry"]]], ignore_index=True)
    ctr_lat = float(gpd.GeoSeries(all_pts["geometry"]).y.mean())
    ctr_lon = float(gpd.GeoSeries(all_pts["geometry"]).x.mean())

    return existing, cand, ctr_lat, ctr_lon

existing_gdf, cand_gdf, ctr_lat, ctr_lon = load_existing_candidates()

# ===================== Directions API =====================
def mapbox_route(lon1,lat1,lon2,lat2, profile="driving", token="", timeout=12):
    if not token: raise RuntimeError("MAPBOX_TOKEN 필요")
    url=f"https://api.mapbox.com/directions/v5/mapbox/{profile}/{lon1},{lat1};{lon2},{lat2}"
    params={"geometries":"geojson","overview":"full","access_token":token}
    r=requests.get(url,params=params,timeout=timeout)
    if r.status_code!=200:
        raise RuntimeError(f"Directions 오류 {r.status_code}: {r.text[:160]}")
    j=r.json(); routes=j.get("routes",[])
    if not routes:
        raise RuntimeError("경로가 반환되지 않았습니다.")
    rt=routes[0]; return rt["geometry"]["coordinates"], float(rt.get("duration",0.0)), float(rt.get("distance",0.0))

# ===================== 노선 생성(단일 차량 연속 경로) =====================
def greedy_pairing(src_xy: List[Tuple[float,float]], dst_xy: List[Tuple[float,float]]) -> List[int]:
    m, n = len(src_xy), len(dst_xy)
    if n == 0: return []
    used = set(); mapping = [-1]*m
    for i in range(m):
        dists = [(haversine(src_xy[i], dst_xy[j]), j) for j in range(n) if j not in used]
        dists.sort(key=lambda x: x[0])
        if dists:
            j = dists[0][1]; mapping[i] = j; used.add(j)
    unused = [j for j in range(n) if j not in used]
    ui = 0
    for i in range(m):
        if mapping[i] == -1 and ui < len(unused):
            mapping[i] = unused[ui]; ui += 1
    return mapping

def build_single_vehicle_steps(stops_df: gpd.GeoDataFrame, starts: List[str], ends: List[str]) -> List[dict]:
    def xy(name):
        r = stops_df.loc[stops_df["name"]==name]
        if r.empty: return None
        pt = r.iloc[0].geometry
        return (float(pt.x), float(pt.y))
    src_xy = [xy(nm) for nm in starts if xy(nm)]
    dst_xy = [xy(nm) for nm in ends if xy(nm)]
    if not src_xy or not dst_xy: return []

    mapping = greedy_pairing(src_xy, dst_xy)
    remaining = list(range(len(src_xy)))
    order = []
    cur_i = 0
    remaining.remove(cur_i)
    order += [
        {"kind":"pickup", "name": starts[cur_i], "xy": src_xy[cur_i]},
        {"kind":"drop",   "name": ends[mapping[cur_i]], "xy": dst_xy[mapping[cur_i]]},
    ]
    current_point = dst_xy[mapping[cur_i]]

    while remaining:
        nxt = min(remaining, key=lambda i: haversine(current_point, src_xy[i]))
        remaining.remove(nxt)
        order.append({"kind":"pickup", "name": starts[nxt], "xy": src_xy[nxt]})
        order.append({"kind":"drop",   "name": ends[mapping[nxt]], "xy": dst_xy[mapping[nxt]]})
        current_point = dst_xy[mapping[nxt]]
    return order

# ===================== 커버리지(덩어리) 유틸 =====================
def _to_proj(gdf, epsg=5179):
    g = gdf.copy()
    if g.crs is None:
        g.set_crs(epsg=4326, inplace=True)
    return g.to_crs(epsg=epsg)

def _to_wgs(gdf):
    return gdf.to_crs(epsg=4326)

def _explode_multipolygon(geom):
    if geom is None:
        return []
    if isinstance(geom, (MultiPolygon,)):
        return [p for p in geom.geoms]
    if isinstance(geom, Polygon):
        return [geom]
    return []

def make_blob_polygons(points_wgs: gpd.GeoDataFrame,
                       radius_m: float = 100,
                       min_area_m2: float = 3_000) -> gpd.GeoDataFrame:
    if points_wgs.empty:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    g_proj = _to_proj(points_wgs, epsg=5179)
    g_proj["buf"] = g_proj.geometry.buffer(radius_m)
    merged = unary_union(g_proj["buf"].values)
    polys = _explode_multipolygon(merged)
    if not polys:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    polys = [p for p in polys if p.area >= min_area_m2]
    out = gpd.GeoDataFrame(geometry=polys, crs="EPSG:5179")
    return _to_wgs(out)

def polygons_difference(A: gpd.GeoDataFrame, B: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if A.empty:
        return A
    if B.empty:
        return A
    a = unary_union(A.geometry.values)
    b = unary_union(B.geometry.values)
    diff = a.difference(b)
    bits = _explode_multipolygon(diff) if isinstance(diff, (MultiPolygon, Polygon)) else []
    out = gpd.GeoDataFrame(geometry=bits, crs=A.crs)
    return out

def area_km2(gdf):
    if gdf.empty:
        return 0.0
    g = _to_proj(gdf, 5179)
    return float(g.area.sum())/1_000_000.0

# ===================== 상단: 노선 추천 UI =====================
col1, col2, col3 = st.columns([1.7,1.2,3.1], gap="large")

with col1:
    st.markdown('<div class="section-header">🚏 노선 추천 설정</div>', unsafe_allow_html=True)
    mode    = st.radio("운행 모드", ["차량(운행)","도보(승객 접근)"], horizontal=True)
    profile = "driving" if mode.startswith("차량") else "walking"

    # 노선 추천은 '신규/후보 정류장' 기준으로 고름
    all_names = cand_gdf["name"].tolist()
    starts = st.multiselect("출발(승차) 정류장", all_names, key="starts")
    ends   = st.multiselect("도착(하차) 정류장", all_names, key="ends")

    route_mode = st.radio("노선 모드", ["개별쌍(모든 조합)","단일 차량(연속 경로)"], index=1)

    st.markdown(
        '<div class="section-header">🧭 범례</div>'
        '<span class="badge badge-red">첫 승차</span> '
        '<span class="badge badge-purple">중간 승차</span> '
        '<span class="badge badge-blue">하차</span>', unsafe_allow_html=True
    )

    cA, cB, cC = st.columns(3)
    run_clicked   = cA.button("노선 추천")
    clear_clicked = cB.button("초기화")
    if cC.button("캐시 초기화"):
        st.cache_data.clear(); st.rerun()
    if clear_clicked:
        for k in ["order","duration","distance"]:
            st.session_state.pop(k, None)
        st.rerun()

with col2:
    st.markdown('<div class="section-header">📍 방문 순서</div>', unsafe_allow_html=True)
    order_list = st.session_state.get("order", [])
    if order_list:
        for i, nm in enumerate(order_list, 1):
            st.markdown(f'<div class="visit-card"><div class="visit-num">{i}</div><div>{nm}</div></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="empty">경로 생성 후 표시됩니다</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.metric("⏱️ 소요시간(합)", f"{st.session_state.get('duration',0.0):.1f}분")
    st.metric("📏 이동거리(합)", f"{st.session_state.get('distance',0.0):.2f}km")

with col3:
    st.markdown('<div class="section-header">🗺️ 추천경로 지도</div>', unsafe_allow_html=True)
    m = folium.Map(location=[ctr_lat, ctr_lon], zoom_start=12, tiles="CartoDB Positron", control_scale=True)

    # 전체 점(후보) 표시
    mc = MarkerCluster().add_to(m)
    for _, r in cand_gdf.iterrows():
        pt = r.geometry
        folium.Marker([pt.y, pt.x], tooltip=str(r["name"]), icon=folium.Icon(color="gray")).add_to(mc)

    if run_clicked:
        if not starts or not ends:
            st.warning("출발/도착 정류장을 각각 1개 이상 선택하세요.")
        elif not MAPBOX_TOKEN:
            st.error("MAPBOX_TOKEN을 입력하세요.")
        else:
            def xy(nm: str):
                row = cand_gdf.loc[cand_gdf["name"]==nm]
                if row.empty: return None
                pt = row.iloc[0].geometry
                return (float(pt.x), float(pt.y))

            total_min, total_km = 0.0, 0.0
            order_names = []

            if route_mode.startswith("개별쌍"):
                for i, s in enumerate(starts):
                    for j, e in enumerate(ends):
                        sxy, exy = xy(s), xy(e)
                        if not sxy or not exy: continue
                        try:
                            coords, dur, dist = mapbox_route(sxy[0], sxy[1], exy[0], exy[1],
                                                             profile=profile, token=MAPBOX_TOKEN)
                            ll = [(c[1], c[0]) for c in coords]
                            folium.PolyLine(ll, color=PALETTE[(i+j) % len(PALETTE)], weight=5, opacity=0.9).add_to(m)
                            total_min += dur/60; total_km += dist/1000
                            order_names.append(f"{s} → {e}")
                        except Exception as e:
                            st.warning(f"{s}→{e} 실패: {e}")

            else:
                steps = build_single_vehicle_steps(cand_gdf, starts, ends)

                def number_marker_html(n: int, color: str) -> str:
                    return (
                        "<div style='"
                        f"background:{color};"
                        "color:#fff;border:2px solid #ffffff;"
                        "border-radius:50%;width:30px;height:30px;"
                        "line-height:30px;text-align:center;font-weight:800;"
                        "box-shadow:0 2px 6px rgba(0,0,0,.35);font-size:13px;'>"
                        f"{n}</div>"
                    )

                prev = None
                for idx, step in enumerate(steps, start=1):
                    lon, lat = step["xy"]; name = step["name"]
                    if step["kind"] == "pickup":
                        color = "#e74c3c" if idx==1 else "#8e44ad"
                    else:
                        color = "#3498db"

                    folium.Marker(
                        [lat, lon],
                        tooltip=f"{idx}. {('승차' if step['kind']=='pickup' else '하차')}: {name}",
                        icon=DivIcon(html=number_marker_html(idx, color))
                    ).add_to(m)

                    if prev is not None:
                        try:
                            coords, dur, dist = mapbox_route(prev[0], prev[1], lon, lat,
                                                             profile=profile, token=MAPBOX_TOKEN)
                            ll = [(c[1], c[0]) for c in coords]
                            folium.PolyLine(ll, color=PALETTE[(idx-1) % len(PALETTE)],
                                            weight=5, opacity=0.9).add_to(m)
                            total_min += dur/60; total_km += dist/1000
                        except Exception as e:
                            st.warning(f"세그먼트 {idx-1}→{idx} 실패: {e}")

                    prev = (lon, lat)
                    order_names.append(f"{name}")

            st.session_state["order"]    = order_names
            st.session_state["duration"] = total_min
            st.session_state["distance"] = total_km

    st_folium(m, height=560, returned_objects=[], use_container_width=True, key="main_map")

# ===================== 하단: 커버리지(덩어리) 지도 =====================
st.markdown("### 🗺 커버리지 비교 (반경 기반 · 덩어리 병합)")
col_r, col_a = st.columns([1,1])
with col_r:
    cov_radius = st.slider("버퍼 반경 (m)", 50, 300, 100, 10, help="정류장 1개가 커버한다고 보는 반경")
with col_a:
    min_blob_area = st.slider("덩어리 최소 면적 (㎡)", 0, 50_000, 3_000, 500,
                              help="이 면적 미만의 작은 조각은 숨깁니다")

# 기존/전체/추가 커버 덩어리 폴리곤
blob_exist = make_blob_polygons(existing_gdf[["geometry"]], radius_m=cov_radius, min_area_m2=min_blob_area)
blob_all   = make_blob_polygons(pd.concat([existing_gdf[["geometry"]], cand_gdf[["geometry"]]], ignore_index=True),
                                radius_m=cov_radius, min_area_m2=min_blob_area)
blob_added = polygons_difference(blob_all, blob_exist)

area_exist = area_km2(blob_exist)
area_all   = area_km2(blob_all)
area_add   = area_km2(blob_added)
pct_incr   = (area_add/area_exist*100.0) if area_exist>0 else 0.0

m2 = folium.Map(location=[ctr_lat, ctr_lon], zoom_start=11, tiles="CartoDB Positron")

def _style(color, fill_opacity=0.25):
    return lambda x: {"color": color, "weight": 2, "fillColor": color, "fillOpacity": fill_opacity}

if not blob_exist.empty:
    folium.GeoJson(blob_exist.__geo_interface__, name="기존 커버", style_function=_style("#1e40af", 0.15)).add_to(m2)
if not blob_added.empty:
    folium.GeoJson(blob_added.__geo_interface__, name="추가 커버", style_function=_style("#16a085", 0.35)).add_to(m2)
if not blob_all.empty:
    folium.GeoJson(blob_all.__geo_interface__, name="전체 커버", style_function=_style("#6b7280", 0.08)).add_to(m2)

folium.LayerControl(collapsed=True).add_to(m2)
st_folium(m2, height=560, returned_objects=[], use_container_width=True, key="coverage_map_blobs")

c1, c2, c3, c4 = st.columns(4)
c1.metric("기존 커버 면적", f"{area_exist:,.3f} km²")
c2.metric("전체(기존+신규) 면적", f"{area_all:,.3f} km²")
c3.metric("면적 증가", f"+{area_add:,.3f} km²")
c4.metric("증가율", f"{pct_incr:,.1f}%")
st.caption("※ 정류장 반경 버퍼를 병합하여 연결된 '덩어리' 폴리곤으로 표시합니다. 작은 조각은 최소 면적 기준으로 제거됩니다.")
