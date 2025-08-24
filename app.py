# app.py
# ---------------------------------------------------------
# 천안 DRT - 맞춤형 AI기반 스마트 교통 가이드
# - 데이터(같은 폴더에 둘 것):
#   1) 기존 DRT:   천안콜 버스 정류장(v250730)_4326.(gpkg/geojson/shp)
#   2) 신규 후보:  new_new_drt_full_utf8.(gpkg/geojson/shp)
# - Fiona 없이(pyogrio)만 사용해 Shapefile을 읽도록 구성
# - 라우팅: Mapbox Directions
# - 커버리지: 반경(m) 버퍼 → union → 면적/증가량 + 차집합(추가 영역) 시각화
# ---------------------------------------------------------

import os
import math
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.ops import unary_union
from shapely.geometry import Point, Polygon, MultiPolygon

import requests
import streamlit as st
import folium
from folium.plugins import MarkerCluster
from folium.features import DivIcon
from streamlit_folium import st_folium

# ===================== 기본 설정/스타일 =====================
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

# ===================== 토큰/상수 =====================
MAPBOX_TOKEN = os.getenv("MAPBOX_TOKEN", "")
if not MAPBOX_TOKEN:
    # 필요시 직접 입력
    MAPBOX_TOKEN = "pk.eyJ1IjoiZ3VyMDUxMDgiLCJhIjoiY21lbWppYjByMDV2ajJqcjQyYXUxdzY3byJ9.yLBRJK_Ib6W3p9f16YlIKQ"

PALETTE = ["#e74c3c","#8e44ad","#3498db","#e67e22","#16a085","#2ecc71","#1abc9c","#d35400"]

# 파일 스템(확장자 없이)
EXISTING_STEM = "천안콜 버스 정류장(v250730)_4326"   # 기존 DRT
CANDIDATE_STEM = "new_new_drt_full_utf8"           # 신규 후보

# ===================== 유틸: 파일 열기 (pyogrio만 사용) =====================
def _read_utf8_shp(path: Path) -> gpd.GeoDataFrame:
    """
    Fiona 없이 pyogrio만 사용해서 Shapefile을 읽는다. (utf-8 → cp949 순으로 시도)
    """
    try:
        from pyogrio import read_dataframe as pio
    except Exception:
        st.error("pyogrio가 필요합니다. requirements에 'pyogrio'를 추가하거나 "
                 "Shapefile을 .gpkg/.geojson으로 변환해 주세요.")
        raise

    for enc in ("utf-8", "cp949", None):
        try:
            g = pio(path, encoding=enc)
            g = gpd.GeoDataFrame(g, geometry="geometry", crs=getattr(g, "crs", None))
            return g
        except Exception:
            continue

    st.error(f"Shapefile 읽기에 실패했습니다: {path.name}\n"
             f"- 인코딩(UTF-8/CP949) 모두 실패\n"
             f"- QGIS로 .gpkg 또는 .geojson으로 변환해 보세요.")
    raise RuntimeError("Shapefile read failed")

def _open_any(stem: str) -> gpd.GeoDataFrame:
    """
    stem.(gpkg|geojson|shp) 중 존재하는 것을 읽어 WGS84(EPSG:4326)로 반환.
    """
    gpkg   = Path(f"./{stem}.gpkg")
    geojs  = Path(f"./{stem}.geojson")
    shp    = Path(f"./{stem}.shp")

    if gpkg.exists():
        g = gpd.read_file(gpkg)
    elif geojs.exists():
        g = gpd.read_file(geojs)
    elif shp.exists():
        g = _read_utf8_shp(shp)
    else:
        st.error(f"'{stem}.gpkg/.geojson/.shp' 중 하나가 필요합니다.")
        st.stop()

    try:
        if g.crs and g.crs.to_epsg() != 4326:
            g = g.to_crs(epsg=4326)
    except Exception:
        pass

    if not g.geom_type.astype(str).str.contains("Point", case=False, na=False).any():
        g = g.copy()
        g["geometry"] = g.geometry.representative_point()
    return g

@st.cache_data
def load_candidates() -> gpd.GeoDataFrame:
    g = _open_any(CANDIDATE_STEM)
    # name 기본값은 jibun -> name
    if "name" in g.columns:
        g["name"] = g["name"].astype(str)
    elif "jibun" in g.columns:
        g["name"] = g["jibun"].astype(str)
    else:
        g["name"] = g.index.astype(str)

    g["lon"] = g.geometry.x
    g["lat"] = g.geometry.y
    st.caption(f"신규 후보 정류장: {len(g)}개")
    return g[["name","lon","lat","geometry"]]

@st.cache_data
def load_existing_candidates():
    existing = _open_any(EXISTING_STEM)  # 기존 DRT
    # 기존 파일에서 표시할 이름 열 유추
    nm_col = None
    for c in ["name","정류장명","정류장명_한글","정류장명_영문","정류장"]:
        if c in existing.columns:
            nm_col = c; break
    if nm_col is None:
        existing["name"] = existing.index.astype(str)
    else:
        existing["name"] = existing[nm_col].astype(str)

    existing["lon"] = existing.geometry.x
    existing["lat"] = existing.geometry.y

    cand = load_candidates()
    ctr_lat = float(pd.concat([existing["lat"], cand["lat"]]).mean())
    ctr_lon = float(pd.concat([existing["lon"], cand["lon"]]).mean())
    st.caption(f"기존 DRT 정류장: {len(existing)}개")
    return existing[["name","lon","lat","geometry"]], cand, ctr_lat, ctr_lon

# ===================== 라우팅 =====================
def mapbox_route(lon1,lat1,lon2,lat2, profile="driving", token="", timeout=12):
    if not token:
        raise RuntimeError("MAPBOX_TOKEN 필요")
    url = f"https://api.mapbox.com/directions/v5/mapbox/{profile}/{lon1},{lat1};{lon2},{lat2}"
    params = {"geometries":"geojson","overview":"full","access_token":token}
    r = requests.get(url, params=params, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"Directions 오류 {r.status_code}: {r.text[:160]}")
    j = r.json(); routes = j.get("routes",[])
    if not routes:
        raise RuntimeError("경로가 반환되지 않았습니다.")
    rt = routes[0]
    return rt["geometry"]["coordinates"], float(rt.get("duration",0.0)), float(rt.get("distance",0.0))

# ===================== 순회 경로 헬퍼 =====================
def haversine(xy1, xy2):
    R=6371000.0
    lon1,lat1,lon2,lat2 = map(np.radians,[xy1[0],xy1[1],xy2[0],xy2[1]])
    dlon=lon2-lon1; dlat=lat2-lat1
    a=np.sin(dlat/2)**2+np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

def greedy_pairing(src_xy: List[Tuple[float,float]], dst_xy: List[Tuple[float,float]]) -> List[int]:
    m, n = len(src_xy), len(dst_xy)
    if n == 0: return []
    used = set()
    mapping = [-1]*m
    for i in range(m):
        dists = [(haversine(src_xy[i], dst_xy[j]), j) for j in range(n) if j not in used]
        dists.sort(key=lambda x: x[0])
        if dists:
            j = dists[0][1]
            mapping[i] = j
            used.add(j)
    unused = [j for j in range(n) if j not in used]
    ui = 0
    for i in range(m):
        if mapping[i] == -1 and ui < len(unused):
            mapping[i] = unused[ui]; ui += 1
    return mapping

def build_single_vehicle_steps(starts: List[str], ends: List[str], all_points: gpd.GeoDataFrame) -> List[dict]:
    def xy(label):
        r = all_points.loc[all_points["name"]==label]
        if r.empty: return None
        rr = r.iloc[0]; return (float(rr["lon"]), float(rr["lat"]))

    src_xy = [xy(nm) for nm in starts if xy(nm)]
    dst_xy = [xy(nm) for nm in ends if xy(nm)]
    if not src_xy or not dst_xy:
        return []

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

# ===================== 커버리지(버퍼/유니온/차집합) =====================
def _buffers_union(points: gpd.GeoDataFrame, radius_m: float):
    """
    points(WGS84) -> 3857로 투영 → buffer → union → (m^2, WGS84 polygon)
    """
    if points.empty:
        return 0.0, None

    g_m = points.to_crs(epsg=3857)
    polys = g_m.buffer(radius_m, cap_style=1)  # round cap
    unioned = unary_union(polys.values)

    if unioned.is_empty:
        return 0.0, None

    area_m2 = float(unioned.area)
    g_w = gpd.GeoSeries([unioned], crs=3857).to_crs(epsg=4326).iloc[0]
    return area_m2, g_w

def coverage_metrics(existing_pts: gpd.GeoDataFrame,
                     added_pts: gpd.GeoDataFrame,
                     radius_m: float):
    """
    기존 커버(기존 포인트만) vs 제안 커버(기존+추가) 및 증가·증가율과
    차집합(제안-기존) 폴리곤 반환
    """
    base_area, base_poly = _buffers_union(existing_pts, radius_m)
    all_pts = pd.concat([existing_pts, added_pts], ignore_index=True)
    prop_area, prop_poly = _buffers_union(all_pts, radius_m)

    inc_area = max(prop_area - base_area, 0.0)
    pct = (inc_area / base_area * 100.0) if base_area > 0 else (100.0 if prop_area > 0 else 0.0)

    # 추가 영역(제안 - 기존)
    diff_poly = None
    if base_poly is not None and prop_poly is not None:
        try:
            diff_poly = prop_poly.difference(base_poly)
        except Exception:
            diff_poly = None

    return base_area, prop_area, inc_area, pct, base_poly, prop_poly, diff_poly

# ===================== 데이터 로드 =====================
existing_gdf, cand_gdf, ctr_lat, ctr_lon = load_existing_candidates()
all_points = pd.concat([existing_gdf, cand_gdf], ignore_index=True)

# ===================== UI =====================
col1, col2, col3 = st.columns([1.8,1.2,3.2], gap="large")

with col1:
    st.markdown('<div class="section-header">🚏 DRT 노선 추천 설정</div>', unsafe_allow_html=True)
    mode    = st.radio("운행 모드", ["차량(운행)","도보(승객 접근)"], horizontal=True)
    profile = "driving" if mode.startswith("차량") else "walking"

    names = all_points["name"].tolist()
    starts = st.multiselect("출발(승차) 정류장", names, key="starts")
    ends   = st.multiselect("도착(하차) 정류장", names, key="ends")

    route_mode = st.radio("노선 모드", ["개별쌍(모든 조합)","단일 차량(연속 경로)"], index=1)

    st.markdown(
        '<div class="section-header">🧭 범례</div>'
        '<span class="badge badge-red">첫 승차</span> '
        '<span class="badge badge-purple">중간 승차</span> '
        '<span class="badge badge-blue">하차</span>'
        , unsafe_allow_html=True
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
    st.markdown('<div class="section-header">🗺️ 추천경로 지도시각화</div>', unsafe_allow_html=True)
    m = folium.Map(location=[ctr_lat, ctr_lon], zoom_start=12, tiles="CartoDB Positron", control_scale=True)
    mc = MarkerCluster().add_to(m)
    for _, r in all_points.iterrows():
        folium.Marker([r["lat"], r["lon"]], tooltip=str(r["name"]), icon=folium.Icon(color="gray")).add_to(mc)

    if run_clicked:
        if not starts or not ends:
            st.warning("출발/도착 정류장을 각각 1개 이상 선택하세요.")
        elif not MAPBOX_TOKEN:
            st.error("MAPBOX_TOKEN을 입력하세요.")
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
                            folium.PolyLine(ll, color=PALETTE[(i+j) % len(PALETTE)],
                                            weight=5, opacity=0.9).add_to(m)
                            total_min += dur/60; total_km += dist/1000
                            order_names.append(f"{s} → {e}")
                        except Exception as e:
                            st.warning(f"{s}→{e} 실패: {e}")
            else:
                steps = build_single_vehicle_steps(starts, ends, all_points)

                def number_marker_html(n: int, color: str) -> str:
                    return (
                        "<div style='"
                        f"background:{color};"
                        "color:#fff;border:2px solid #ffffff;"
                        "border-radius:50%;width:30px;height:30px;"
                        "line-height:30px;text-align:center;font-weight:800;"
                        "box-shadow:0 2px 6px rgba(0,0,0,.35);"
                        "font-size:13px;'>"
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

# ===================== 커버리지 비교(전체 기준) =====================
st.markdown("----")
st.markdown('<div class="section-header">🗺️ 커버리지 비교 (반경 100m 기준 · 선택무관 전체 기준)</div>', unsafe_allow_html=True)

radius_m = st.slider("커버리지 반경(미터)", min_value=50, max_value=300, value=100, step=10)

base_area, prop_area, inc_area, pct, base_poly, prop_poly, diff_poly = coverage_metrics(
    existing_gdf, cand_gdf, radius_m
)

colA, colB, colC, colD = st.columns(4)
colA.metric("기존 커버 면적", f"{base_area/1e6:.3f} km²")
colB.metric("제안(기존+추가) 커버 면적", f"{prop_area/1e6:.3f} km²")
colC.metric("면적 증가", f"+{inc_area/1e6:.3f} km²")
colD.metric("증가율", f"+{pct:.1f}%")

m2 = folium.Map(location=[ctr_lat, ctr_lon], zoom_start=11, tiles="CartoDB Positron", control_scale=True)

# 기존/추가/제안 표시
# 1) 기존 커버(회색)
if base_poly is not None and not base_poly.is_empty:
    folium.GeoJson(
        data=gpd.GeoSeries([base_poly], crs="EPSG:4326").__geo_interface__,
        name="기존 커버",
        style_function=lambda x: {"color":"#666","weight":1.5,"fill":True,"fillColor":"#666","fillOpacity":0.18}
    ).add_to(m2)

# 2) 추가 커버(diff = 제안 - 기존, 파란색)
if diff_poly is not None and not diff_poly.is_empty:
    folium.GeoJson(
        data=gpd.GeoSeries([diff_poly], crs="EPSG:4326").__geo_interface__,
        name="추가 커버(증설 효과)",
        style_function=lambda x: {"color":"#1f77b4","weight":1.5,"fill":True,"fillColor":"#1f77b4","fillOpacity":0.28}
    ).add_to(m2)

# 3) 기준점 찍기 (기존=빨강, 신규=파랑)
ex_grp = folium.FeatureGroup(name=f"기존 정류장({len(existing_gdf)})").add_to(m2)
for _, r in existing_gdf.iterrows():
    folium.CircleMarker([r["lat"], r["lon"]], radius=3.5, color="#d62728", fill=True, fill_opacity=0.9,
                        tooltip=f"[기존] {r['name']}").add_to(ex_grp)

new_grp = folium.FeatureGroup(name=f"신규 후보 정류장({len(cand_gdf)})").add_to(m2)
for _, r in cand_gdf.iterrows():
    folium.CircleMarker([r["lat"], r["lon"]], radius=3.0, color="#1f77b4", fill=True, fill_opacity=0.9,
                        tooltip=f"[신규] {r['name']}").add_to(new_grp)

folium.LayerControl(collapsed=False).add_to(m2)
st_folium(m2, height=560, use_container_width=True, key="coverage_map_all")
