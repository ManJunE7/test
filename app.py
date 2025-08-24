# app.py
# ---------------------------------------------------------
# 천안 DRT - 맞춤형 AI기반 스마트 교통 가이드 (비교/효과 + 색상 구분 포함)
# - 데이터: new_new_drt_full_utf8.(shp/gpkg/geojson) (UTF-8)
# - 기본 이름(name)=지번(jibun)
# - Mapbox Directions로 실도로 라우팅
# - 단일차량(연속 경로) 방문 순서에 숫자 아이콘 표시
# - 중간 승차 지점 = 보라색, 첫 승차 = 빨강, 하차 = 파랑
# - '기존 DRT 정류장'은 별도 색상/레이어로 표시
# - 기존 vs 제안: 총 소요시간/총거리/커버리지 면적 비교
# ---------------------------------------------------------

import os, math
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
from streamlit_folium import st_folium
from shapely.geometry import Point
from shapely.ops import unary_union

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
.legend-chip{display:inline-flex;align-items:center;gap:6px;margin-right:10px}
.legend-dot{width:10px;height:10px;border-radius:50%}
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
    try:
        MAPBOX_TOKEN = st.secrets["MAPBOX_TOKEN"]
    except Exception:
        pass
if not MAPBOX_TOKEN:
    # 데모용(가능하면 자신의 토큰으로 교체)
    MAPBOX_TOKEN = "pk.eyJ1IjoiZ3VyMDUxMDgiLCJhIjoiY21lbWppYjByMDV2ajJqcjQyYXUxdzY3byJ9.yLBRJK_Ib6W3p9f16YlIKQ"

PALETTE = ["#e74c3c","#8e44ad","#3498db","#e67e22","#16a085","#2ecc71","#1abc9c","#d35400"]
DATA_STEM = "new_new_drt_full_utf8"   # 데이터 파일명 앞부분

# ===================== 유틸 =====================
def haversine(xy1, xy2):
    R=6371000.0
    lon1,lat1,lon2,lat2 = map(np.radians,[xy1[0],xy1[1],xy2[0],xy2[1]])
    dlon=lon2-lon1; dlat=lat2-lat1
    a=np.sin(dlat/2)**2+np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

def _read_utf8_shp(path: Path) -> gpd.GeoDataFrame:
    try:
        from pyogrio import read_dataframe as pio
        g = pio(path, encoding="utf-8")
        return gpd.GeoDataFrame(g, geometry="geometry", crs=getattr(g, "crs", None))
    except Exception:
        pass
    try:
        return gpd.read_file(path, encoding="utf-8")
    except Exception:
        pass
    os.environ["SHAPE_ENCODING"] = "UTF-8"
    return gpd.read_file(path, engine="fiona")

def _open_any() -> gpd.GeoDataFrame:
    for ext in (".shp",".gpkg",".geojson"):
        p = Path(f"./{DATA_STEM}{ext}")
        if p.exists():
            g = _read_utf8_shp(p) if ext==".shp" else gpd.read_file(p)
            try:
                if g.crs and g.crs.to_epsg()!=4326:
                    g = g.to_crs(epsg=4326)
            except Exception:
                pass
            if not g.geom_type.astype(str).str.contains("Point",case=False,na=False).any():
                g = g.copy(); g["geometry"]=g.geometry.representative_point()
            return g
    st.error(f"'{DATA_STEM}.shp/.gpkg/.geojson' 파일을 같은 폴더에 두세요."); st.stop()

@st.cache_data
def load_stops() -> gpd.GeoDataFrame:
    g = _open_any()
    if "jibun" not in g.columns:
        st.error("소스에 'jibun' 필드가 없습니다."); st.stop()
    g = g.copy()
    g["jibun"] = g["jibun"].astype(str).str.strip()
    g["name"]  = g.get("name", g["jibun"]).astype(str)
    g["lon"]   = g.geometry.x; g["lat"]=g.geometry.y
    st.caption(f"데이터셋: {DATA_STEM} (포인트 {len(g)}개 · UTF-8)")
    return g[["jibun","name","lon","lat","geometry"]]

stops = load_stops()

# ===================== Mapbox Directions/Isochrone =====================
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

def mapbox_isochrone_polygon(lon: float, lat: float, minutes: int = 10, profile: str = "walking", token: str = "") -> Optional[gpd.GeoDataFrame]:
    if not token:
        return None
    url = f"https://api.mapbox.com/isochrone/v1/mapbox/{profile}/{lon},{lat}"
    params = {"contours_minutes": str(minutes), "polygons": "true", "access_token": token}
    r = requests.get(url, params=params, timeout=12)
    if r.status_code != 200:
        return None
    gj = r.json()
    if not gj.get("features"):
        return None
    return gpd.GeoDataFrame.from_features(gj, crs="EPSG:4326")

# ===================== 커버리지(버퍼) =====================
def union_buffer_area_km2(points_lonlat: List[Tuple[float,float]], radius_m: float) -> Tuple[gpd.GeoDataFrame, float]:
    if not points_lonlat:
        return gpd.GeoDataFrame(geometry=[]), 0.0
    g = gpd.GeoDataFrame(geometry=[Point(lon, lat) for lon,lat in points_lonlat], crs="EPSG:4326").to_crs(epsg=3857)
    bufs = g.buffer(radius_m)
    unioned = unary_union(bufs)
    area_km2 = (gpd.GeoSeries([unioned], crs="EPSG:3857").area.iloc[0]) / 1_000_000
    out = gpd.GeoDataFrame(geometry=[unioned], crs="EPSG:3857").to_crs(epsg=4326)
    return out, float(area_km2)

# ===================== 단일차량(연속 경로) =====================
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

def build_single_vehicle_steps(starts: List[str], ends: List[str]) -> List[dict]:
    def xy(label):
        r = stops.loc[stops["name"]==label]
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

# ===================== 공통 유틸(비교 계산) =====================
def total_route_minutes_km(order_points: List[Tuple[float,float]], profile: str, token: str) -> Tuple[float, float]:
    if len(order_points) < 2:
        return 0.0, 0.0
    tot_min, tot_km = 0.0, 0.0
    for (lon1,lat1), (lon2,lat2) in zip(order_points[:-1], order_points[1:]):
        coords, dur, dist = mapbox_route(lon1, lat1, lon2, lat2, profile=profile, token=token)
        tot_min += dur/60; tot_km += dist/1000
    return tot_min, tot_km

def coords_from_names(names: List[str]) -> List[Tuple[float,float]]:
    pts = []
    for nm in names:
        row = stops.loc[stops["name"]==nm]
        if not row.empty:
            rr = row.iloc[0]; pts.append((float(rr["lon"]), float(rr["lat"])))
    return pts

def greedy_order(points: List[Tuple[float,float]]):
    if len(points) <= 1: return points
    remain = list(range(1, len(points)))
    order = [0]
    cur = 0
    while remain:
        nxt = min(remain, key=lambda j: haversine(points[cur], points[j]))
        order.append(nxt); remain.remove(nxt); cur = nxt
    return [points[i] for i in order]

# ===================== UI =====================
col1, col2, col3 = st.columns([1.8,1.2,3.2], gap="large")

with col1:
    st.markdown('<div class="section-header">🚏 DRT 노선 추천 설정</div>', unsafe_allow_html=True)
    mode    = st.radio("운행 모드", ["차량(운행)","도보(승객 접근)"], horizontal=True)
    profile = "driving" if mode.startswith("차량") else "walking"

    all_names = stops["name"].tolist()
    starts = st.multiselect("출발(승차) 정류장", all_names, key="starts")
    ends   = st.multiselect("도착(하차) 정류장", all_names, key="ends")

    # 비교/효과 섹션
    st.markdown('<div class="section-header">📈 비교/효과</div>', unsafe_allow_html=True)
    cmp_on = st.checkbox("기존 DRT 대비 효과 계산", value=True)
    buff_r = st.slider("커버리지 반경(버퍼·m)", 200, 1200, 500, 50)
    use_iso = st.checkbox("보행 아이소크론(10분)으로 커버리지 계산(선택)", value=False)

    baseline_names = st.multiselect("기존 DRT 정류장", all_names, help="기존 운영 기준으로 삼을 정류장들")

    # 제안안 집합(현재 선택의 유니크)
    proposed_names = sorted(set(starts + ends))

    route_mode = st.radio("노선 모드", ["개별쌍(모든 조합)","단일 차량(연속 경로)"], index=1)

    st.markdown(
        '<div class="section-header">🧭 범례</div>'
        '<span class="legend-chip"><span class="legend-dot" style="background:#e74c3c"></span>첫 승차</span>'
        '<span class="legend-chip"><span class="legend-dot" style="background:#8e44ad"></span>중간 승차</span>'
        '<span class="legend-chip"><span class="legend-dot" style="background:#3498db"></span>하차</span>'
        '<span class="legend-chip"><span class="legend-dot" style="background:#22c55e;border:1px solid #065f46"></span>기존 정류장</span>'
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
    ctr_lat = float(stops["lat"].mean()); ctr_lon = float(stops["lon"].mean())
    if math.isnan(ctr_lat) or math.isnan(ctr_lon): ctr_lat, ctr_lon = 36.80, 127.15

    m = folium.Map(location=[ctr_lat, ctr_lon], zoom_start=12, tiles="CartoDB Positron", control_scale=True)

    # 모든 정류장 기본 마커(연한 회색)
    mc_all = MarkerCluster(name="모든 정류장").add_to(m)
    for _, r in stops.iterrows():
        folium.Marker(
            [r["lat"], r["lon"]],
            tooltip=str(r["name"]),
            icon=folium.Icon(color="lightgray", icon="circle", prefix="fa")
        ).add_to(mc_all)

    # 기존 DRT 정류장(요청: 다른 색으로 강조)
    if baseline_names:
        fg_base = folium.FeatureGroup(name="기존 DRT 정류장", show=True).add_to(m)
        for nm in baseline_names:
            row = stops.loc[stops["name"]==nm]
            if row.empty: continue
            rr = row.iloc[0]
            # 진한 초록 원형 마커
            folium.CircleMarker(
                [float(rr["lat"]), float(rr["lon"])],
                radius=7, color="#065f46", weight=2, fill=True, fill_color="#22c55e", fill_opacity=0.9,
                tooltip=f"[기존] {nm}"
            ).add_to(fg_base)

    # 실행
    total_min, total_km = 0.0, 0.0
    order_names = []

    if run_clicked:
        if not starts or not ends:
            st.warning("출발/도착 정류장을 각각 1개 이상 선택하세요.")
        elif not MAPBOX_TOKEN:
            st.error("MAPBOX_TOKEN을 입력하세요.")
        else:
            def xy(nm: str):
                row = stops.loc[stops["name"]==nm]
                if row.empty: return None
                rr = row.iloc[0]; return (float(rr["lon"]), float(rr["lat"]))

            if route_mode.startswith("개별쌍"):
                # 모든 (출발,도착) 조합 라우팅
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
                # 단일 차량(연속 경로) — 숫자 아이콘 + 색상 규칙
                steps = build_single_vehicle_steps(starts, ends)

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

    # ========= 비교/효과 계산 & 지도 오버레이 =========
    if cmp_on and baseline_names and proposed_names:
        # 시간/거리 비교(공정 비교: 최근점 그리디 순서)
        base_pts = coords_from_names(baseline_names)
        prop_pts = coords_from_names(proposed_names)
        base_seq = greedy_order(base_pts)
        prop_seq = greedy_order(prop_pts)
        base_min, base_km = total_route_minutes_km(base_seq, profile, MAPBOX_TOKEN)
        prop_min, prop_km = total_route_minutes_km(prop_seq, profile, MAPBOX_TOKEN)

        # 커버리지(버퍼 or 아이소크론)
        if use_iso and MAPBOX_TOKEN:
            def iso_union(names):
                polys = []
                for nm in names:
                    row = stops.loc[stops["name"]==nm]
                    if row.empty: continue
                    rr = row.iloc[0]
                    g = mapbox_isochrone_polygon(float(rr["lon"]), float(rr["lat"]), minutes=10, profile="walking", token=MAPBOX_TOKEN)
                    if g is not None and not g.empty:
                        polys.append(g.unary_union)
                if polys:
                    unioned = unary_union(polys)
                    a_km2 = gpd.GeoSeries([unioned], crs="EPSG:4326").to_crs(epsg=3857).area.iloc[0]/1_000_000
                    return gpd.GeoDataFrame(geometry=[unioned], crs="EPSG:4326"), float(a_km2)
                return gpd.GeoDataFrame(geometry=[]), 0.0
            base_cov_gdf, base_km2 = iso_union(baseline_names)
            prop_cov_gdf, prop_km2 = iso_union(proposed_names)
        else:
            base_cov_gdf, base_km2 = union_buffer_area_km2(base_pts, buff_r)
            prop_cov_gdf, prop_km2 = union_buffer_area_km2(prop_pts, buff_r)

        # 메트릭
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.metric("⏱️ 총 소요시간(기존→제안)", f"{base_min:.1f} → {prop_min:.1f} 분", f"{base_min - prop_min:+.1f} 분")
        c2.metric("📏 총 이동거리(기존→제안)", f"{base_km:.2f} → {prop_km:.2f} km", f"{base_km - prop_km:+.2f} km")
        c3.metric("🗺️ 커버리지 면적(기존→제안)", f"{base_km2:.2f} → {prop_km2:.2f} km²", f"{prop_km2 - base_km2:+.2f} km²")

        # 지도 오버레이
        try:
            if not base_cov_gdf.empty:
                folium.GeoJson(base_cov_gdf.__geo_interface__, name="기존 커버리지",
                               style_function=lambda x: {"color":"#ef4444","fillColor":"#ef4444","fillOpacity":0.15, "weight":2}).add_to(m)
            if not prop_cov_gdf.empty:
                folium.GeoJson(prop_cov_gdf.__geo_interface__, name="제안 커버리지",
                               style_function=lambda x: {"color":"#10b981","fillColor":"#10b981","fillOpacity":0.15, "weight":2}).add_to(m)
            folium.LayerControl(collapsed=True).add_to(m)
        except Exception as e:
            st.info(f"커버리지 시각화 생략: {e}")

    st_folium(m, height=560, returned_objects=[], use_container_width=True, key="main_map")
