# app.py
# ---------------------------------------------------------
# 천안 DRT - 맞춤형 AI기반 스마트 교통 가이드 (Fiona 미사용 / pyogrio 전용)
# ---------------------------------------------------------

import os, math
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.ops import unary_union
from shapely.geometry import Polygon, MultiPolygon

import requests
import streamlit as st
import folium
from folium.plugins import MarkerCluster
from folium.features import DivIcon
from streamlit_folium import st_folium

# ===================== 기본 UI =====================
APP_TITLE = "천안 DRT - 맞춤형 AI기반 스마트 교통 가이드"
LOGO_URL  = "https://raw.githubusercontent.com/JeongWon4034/cheongju/main/cheongpung_logo.png"

st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Noto Sans KR', -apple-system, BlinkMacSystemFont, sans-serif; }
.header{display:flex;align-items:center;gap:12px;margin:8px 0 12px}
.title{font-size:1.7rem;font-weight:800}
.section{font-weight:800;border-bottom:2px solid #f3f4f6;padding-bottom:6px;margin:10px 0}
.legend-chip{display:inline-flex;align-items:center;gap:6px;margin-right:10px}
.legend-dot{width:10px;height:10px;border-radius:50%}
.visit-card{display:flex;align-items:center;gap:10px;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:#fff;border-radius:12px;padding:8px 10px;margin-bottom:6px}
.visit-num{background:#fff;color:#667eea;width:22px;height:22px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:800;font-size:.75rem}
.empty{color:#9ca3af;background:linear-gradient(135deg,#ffecd2 0%,#fcb69f 100%);border-radius:12px;padding:18px 12px;text-align:center}
</style>
""", unsafe_allow_html=True)
st.markdown(f'<div class="header"><img src="{LOGO_URL}" width="56"/><div class="title">{APP_TITLE}</div></div>', unsafe_allow_html=True)

# ===================== 상수 =====================
EXISTING_STEM  = "천안콜 버스 정류장(v250730)_4326"  # 기존 DRT 파일 스템
CANDIDATE_STEM = "new_new_drt_full_utf8"           # 신규/후보 파일 스템
PALETTE = ["#e74c3c","#8e44ad","#3498db","#e67e22","#16a085","#2ecc71","#1abc9c","#d35400"]

MAPBOX_TOKEN = os.getenv("MAPBOX_TOKEN", "")
if not MAPBOX_TOKEN:
    MAPBOX_TOKEN = "pk.eyJ1IjoiZ3VyMDUxMDgiLCJhIjoiY21lbWppYjByMDV2ajJqcjQyYXUxdzY3byJ9.yLBRJK_Ib6W3p9f16YlIKQ"

# ===================== 파일 로드 (pyogrio 전용) =====================
def _read_shp_pyogrio(path: Path) -> gpd.GeoDataFrame:
    """Shapefile을 pyogrio로만 읽는다. (utf-8 → cp949 → 기본)"""
    try:
        from pyogrio import read_dataframe as pio
    except Exception:
        st.error("pyogrio가 필요합니다. requirements에 'pyogrio'를 추가하거나 "
                 "Shapefile을 .gpkg 또는 .geojson으로 변환해 주세요.")
        raise

    for enc in ("utf-8", "cp949", None):
        try:
            g = pio(path, encoding=enc)
            return gpd.GeoDataFrame(g, geometry="geometry", crs=getattr(g, "crs", None))
        except Exception:
            continue
    st.error(f"Shapefile 읽기 실패: {path.name} (UTF-8/CP949 모두 실패)")
    raise RuntimeError("Shapefile read failed")

def _open_any(stem: str) -> gpd.GeoDataFrame:
    """stem.(gpkg|geojson|shp) 중 하나를 읽어 WGS84(Point)로 반환."""
    gpkg  = Path(f"./{stem}.gpkg")
    geojs = Path(f"./{stem}.geojson")
    shp   = Path(f"./{stem}.shp")

    if gpkg.exists():
        g = gpd.read_file(gpkg)       # engine 지정 안 함
    elif geojs.exists():
        g = gpd.read_file(geojs)      # engine 지정 안 함
    elif shp.exists():
        g = _read_shp_pyogrio(shp)    # pyogrio만 사용
    else:
        st.error(f"'{stem}.gpkg/.geojson/.shp' 중 하나가 필요합니다.")
        st.stop()

    try:
        if g.crs and g.crs.to_epsg() != 4326:
            g = g.to_crs(epsg=4326)
    except Exception:
        pass

    # 포인트가 아니면 대표점으로 변환
    if not g.geom_type.astype(str).str.contains("Point", case=False, na=False).any():
        g = g.copy(); g["geometry"] = g.geometry.representative_point()
    return g

@st.cache_data
def load_existing_candidates():
    """당신 앱이 호출하는 함수명 그대로 유지 (기존+신규 로드)."""
    existing = _open_any(EXISTING_STEM)
    # 이름 컬럼 유추
    nm_col = None
    for c in ["name","정류장명","정류소명","정류장명_한글","jibun","NAME"]:
        if c in existing.columns:
            nm_col = c; break
    existing["name"] = existing[nm_col].astype(str) if nm_col else existing.index.astype(str)
    existing["lon"] = existing.geometry.x; existing["lat"] = existing.geometry.y
    existing = existing[["name","lon","lat","geometry"]]

    cand = _open_any(CANDIDATE_STEM)
    if "name" in cand.columns: cand["name"] = cand["name"].astype(str)
    elif "jibun" in cand.columns: cand["name"] = cand["jibun"].astype(str)
    else: cand["name"] = cand.index.astype(str)
    cand["lon"] = cand.geometry.x; cand["lat"] = cand.geometry.y
    cand = cand[["name","lon","lat","geometry"]]

    ctr_lat = float(pd.concat([existing["lat"], cand["lat"]]).mean())
    ctr_lon = float(pd.concat([existing["lon"], cand["lon"]]).mean())
    return existing, cand, ctr_lat, ctr_lon

# ===================== 라우팅/순회 유틸 =====================
def haversine(xy1, xy2):
    R=6371000.0
    lon1,lat1,lon2,lat2 = map(np.radians,[xy1[0],xy1[1],xy2[0],xy2[1]])
    dlon=lon2-lon1; dlat=lat2-lat1
    a=np.sin(dlat/2)**2+np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

def mapbox_route(lon1,lat1,lon2,lat2, profile="driving", token="", timeout=12):
    if not token: raise RuntimeError("MAPBOX_TOKEN 필요")
    url=f"https://api.mapbox.com/directions/v5/mapbox/{profile}/{lon1},{lat1};{lon2},{lat2}"
    params={"geometries":"geojson","overview":"full","access_token":token}
    r=requests.get(url,params=params,timeout=timeout)
    if r.status_code!=200:
        raise RuntimeError(f"Directions 오류 {r.status_code}: {r.text[:160]}")
    j=r.json(); routes=j.get("routes",[])
    if not routes: raise RuntimeError("경로가 반환되지 않았습니다.")
    rt=routes[0]; return rt["geometry"]["coordinates"], float(rt.get("duration",0.0)), float(rt.get("distance",0.0))

def greedy_pairing(src_xy: List[Tuple[float,float]], dst_xy: List[Tuple[float,float]]) -> List[int]:
    m, n = len(src_xy), len(dst_xy)
    if n == 0: return []
    used = set(); mapping = [-1]*m
    for i in range(m):
        dists = [(haversine(src_xy[i], dst_xy[j]), j) for j in range(n) if j not in used]
        if dists:
            j = min(dists, key=lambda x:x[0])[1]; mapping[i]=j; used.add(j)
    unused = [j for j in range(n) if j not in used]; ui=0
    for i in range(m):
        if mapping[i]==-1 and ui<len(unused):
            mapping[i]=unused[ui]; ui+=1
    return mapping

def build_single_vehicle_steps(starts, ends, points_df: pd.DataFrame):
    def xy(name):
        r = points_df.loc[points_df["name"]==name]
        if r.empty: return None
        rr = r.iloc[0]; return (float(rr["lon"]), float(rr["lat"]))
    src_xy = [xy(nm) for nm in starts if xy(nm)]
    dst_xy = [xy(nm) for nm in ends if xy(nm)]
    if not src_xy or not dst_xy: return []
    mapping = greedy_pairing(src_xy, dst_xy)
    remaining = list(range(len(src_xy))); order=[]
    cur=0; remaining.remove(cur)
    order += [{"kind":"pickup","name":starts[cur],"xy":src_xy[cur]},
              {"kind":"drop","name":ends[mapping[cur]],"xy":dst_xy[mapping[cur]]}]
    cur_pt = dst_xy[mapping[cur]]
    while remaining:
        nxt = min(remaining, key=lambda i: haversine(cur_pt, src_xy[i]))
        remaining.remove(nxt)
        order.append({"kind":"pickup","name":starts[nxt],"xy":src_xy[nxt]})
        order.append({"kind":"drop","name":ends[mapping[nxt]],"xy":dst_xy[mapping[nxt]]})
        cur_pt = dst_xy[mapping[nxt]]
    return order

# ===================== 커버리지(버퍼→합집합→차집합) =====================
def buffers_union_wgs(points: gpd.GeoDataFrame, radius_m: float):
    """points(WGS84) → 3857로 투영 → buffer → unary_union → (m², WGS84 폴리곤)"""
    if points.empty: return 0.0, None
    g_m = points.to_crs(epsg=3857)
    unioned = unary_union(g_m.buffer(radius_m).values)
    if unioned.is_empty: return 0.0, None
    area_m2 = float(unioned.area)
    poly_wgs = gpd.GeoSeries([unioned], crs=3857).to_crs(epsg=4326).iloc[0]
    return area_m2, poly_wgs

def coverage_metrics(existing_pts: gpd.GeoDataFrame, added_pts: gpd.GeoDataFrame, radius_m: float):
    base_area, base_poly = buffers_union_wgs(existing_pts, radius_m)
    all_pts = pd.concat([existing_pts, added_pts], ignore_index=True)
    prop_area, prop_poly = buffers_union_wgs(all_pts, radius_m)
    inc_area = max(prop_area - base_area, 0.0)
    pct = (inc_area / base_area * 100.0) if base_area>0 else (100.0 if prop_area>0 else 0.0)
    diff_poly = None
    if base_poly is not None and prop_poly is not None:
        try:
            diff_poly = prop_poly.difference(base_poly)
        except Exception:
            diff_poly = None
    return base_area, prop_area, inc_area, pct, base_poly, prop_poly, diff_poly

# ===================== 데이터 로딩 =====================
existing_gdf, cand_gdf, ctr_lat, ctr_lon = load_existing_candidates()
all_points = pd.concat([existing_gdf, cand_gdf], ignore_index=True)

# ===================== 상단: 노선 추천 =====================
st.markdown('<div class="section">🚏 노선 추천</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns([1.8,1.2,3.2], gap="large")

with c1:
    mode    = st.radio("운행 모드", ["차량(운행)","도보(승객 접근)"], horizontal=True)
    profile = "driving" if mode.startswith("차량") else "walking"
    names = all_points["name"].tolist()
    starts = st.multiselect("출발(승차) 정류장", names)
    ends   = st.multiselect("도착(하차) 정류장", names)
    route_mode = st.radio("노선 모드", ["개별쌍(모든 조합)","단일 차량(연속 경로)"], index=1)
    run_btn = st.button("노선 추천 실행")

with c2:
    st.markdown("**방문 순서**")
    ord_list = st.session_state.get("order", [])
    if ord_list:
        for i, nm in enumerate(ord_list, 1):
            st.markdown(f'<div class="visit-card"><div class="visit-num">{i}</div><div>{nm}</div></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="empty">경로 생성 후 표시됩니다</div>', unsafe_allow_html=True)
    st.metric("소요시간(합)", f"{st.session_state.get('duration',0.0):.1f}분")
    st.metric("이동거리(합)", f"{st.session_state.get('distance',0.0):.2f}km")

with c3:
    m = folium.Map(location=[ctr_lat, ctr_lon], zoom_start=12, tiles="CartoDB Positron", control_scale=True)
    mc = MarkerCluster().add_to(m)
    for _, r in all_points.iterrows():
        folium.Marker([r["lat"], r["lon"]], tooltip=str(r["name"]), icon=folium.Icon(color="gray")).add_to(mc)

    if run_btn:
        if not starts or not ends:
            st.warning("출발/도착 정류장을 각각 1개 이상 선택하세요.")
        elif not MAPBOX_TOKEN:
            st.error("MAPBOX_TOKEN을 설정하세요.")
        else:
            def xy(nm: str):
                row = all_points.loc[all_points["name"]==nm]
                if row.empty: return None
                rr = row.iloc[0]; return (float(rr["lon"]), float(rr["lat"]))
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
                steps = build_single_vehicle_steps(starts, ends, all_points)
                def badge(n, color):
                    return ("<div style='background:"+color+";color:#fff;border:2px solid #fff;"
                            "border-radius:50%;width:30px;height:30px;line-height:30px;text-align:center;"
                            "font-weight:800;box-shadow:0 2px 6px rgba(0,0,0,.35);font-size:13px;'>"+str(n)+"</div>")
                prev = None
                for idx, step in enumerate(steps, start=1):
                    lon, lat = step["xy"]; name = step["name"]
                    color = "#e74c3c" if (step["kind"]=="pickup" and idx==1) else ("#8e44ad" if step["kind"]=="pickup" else "#3498db")
                    folium.Marker([lat, lon], tooltip=f"{idx}. {step['kind']} : {name}",
                                  icon=DivIcon(html=badge(idx, color))).add_to(m)
                    if prev is not None:
                        try:
                            coords, dur, dist = mapbox_route(prev[0], prev[1], lon, lat, profile=profile, token=MAPBOX_TOKEN)
                            ll = [(c[1], c[0]) for c in coords]
                            folium.PolyLine(ll, color=PALETTE[(idx-1) % len(PALETTE)], weight=5, opacity=0.9).add_to(m)
                            total_min += dur/60; total_km += dist/1000
                        except Exception as e:
                            st.warning(f"세그먼트 {idx-1}→{idx} 실패: {e}")
                    prev = (lon, lat); order_names.append(name)
            st.session_state["order"]    = order_names
            st.session_state["duration"] = total_min
            st.session_state["distance"] = total_km

    st_folium(m, height=520, returned_objects=[], use_container_width=True, key="routing_map")

# ===================== 하단: 커버리지 비교 =====================
st.markdown('<div class="section">🗺️ 커버리지 비교 (반경 기반 · 덩어리 병합)</div>', unsafe_allow_html=True)
colr, cola = st.columns(2)
with colr:
    radius_m = st.slider("버퍼 반경 (m)", 50, 300, 100, 10)
with cola:
    st.caption("※ 면적은 EPSG:3857에서 계산(㎡), 표시는 WGS84 지도")

base_area, prop_area, inc_area, pct, base_poly, prop_poly, diff_poly = coverage_metrics(
    existing_gdf[["geometry"]], cand_gdf[["geometry"]], radius_m
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("기존 커버 면적", f"{base_area/1_000_000:.3f} km²")
c2.metric("전체(기존+신규) 면적", f"{prop_area/1_000_000:.3f} km²")
c3.metric("면적 증가", f"+{inc_area/1_000_000:.3f} km²")
c4.metric("증가율", f"{pct:.1f}%")

m2 = folium.Map(location=[ctr_lat, ctr_lon], zoom_start=11, tiles="CartoDB Positron", control_scale=True)

def sty(color, op):
    return lambda x: {"color": color, "weight": 2, "fillColor": color, "fillOpacity": op}

if base_poly is not None and not base_poly.is_empty:
    folium.GeoJson(gpd.GeoSeries([base_poly], crs="EPSG:4326").__geo_interface__,
                   name="기존 커버", style_function=sty("#6b7280", 0.15)).add_to(m2)

if diff_poly is not None and not diff_poly.is_empty:
    folium.GeoJson(gpd.GeoSeries([diff_poly], crs="EPSG:4326").__geo_interface__,
                   name="추가 커버(증설 효과)", style_function=sty("#16a085", 0.35)).add_to(m2)

ex_fg = folium.FeatureGroup(name=f"기존 정류장({len(existing_gdf)})").add_to(m2)
for _, r in existing_gdf.iterrows():
    folium.CircleMarker([r["lat"], r["lon"]], radius=3.5, color="#ef4444", fill=True, fill_opacity=0.9,
                        tooltip=f"[기존] {r['name']}").add_to(ex_fg)

new_fg = folium.FeatureGroup(name=f"신규 정류장({len(cand_gdf)})").add_to(m2)
for _, r in cand_gdf.iterrows():
    folium.CircleMarker([r["lat"], r["lon"]], radius=3.0, color="#3b82f6", fill=True, fill_opacity=0.9,
                        tooltip=f"[신규] {r['name']}").add_to(new_fg)

folium.LayerControl(collapsed=False).add_to(m2)
st_folium(m2, height=560, returned_objects=[], use_container_width=True, key="coverage_map")
