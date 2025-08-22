# app.py
# ---------------------------------------------------------
# 천안 DRT - 맞춤형 AI기반 스마트 교통 가이드
# - 데이터: new_new_drt_full_utf8.(shp/gpkg/geojson) (UTF-8)
# - 기본 이름(name)=지번(jibun)
# - Mapbox Directions로 실도로 라우팅
# - 단일차량(연속 경로) 방문 순서에 숫자 아이콘 표시
# - 중간 승차 지점 = 보라색, 첫 승차 = 빨강, 하차 = 파랑
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
MAPBOX_TOKEN = ""  # << 여기에 Mapbox access token 직접 입력하세요 (또는 환경변수/Secrets)
if not MAPBOX_TOKEN:
    MAPBOX_TOKEN = os.getenv("MAPBOX_TOKEN", "pk.eyJ1IjoiZ3VyMDUxMDgiLCJhIjoiY21lbWppYjByMDV2ajJqcjQyYXUxdzY3byJ9.yLBRJK_Ib6W3p9f16YlIKQ")
if not MAPBOX_TOKEN:
    try:
        MAPBOX_TOKEN = st.secrets["MAPBOX_TOKEN"]
    except Exception:
        pass

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

# ===================== Mapbox Directions =====================
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

# ===================== 단일차량(연속 경로) - 방문 순서 구성 =====================
def greedy_pairing(src_xy: List[Tuple[float,float]], dst_xy: List[Tuple[float,float]]) -> List[int]:
    """각 승차에 대해 가장 가까운 하차 인덱스(중복 없이) 할당"""
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
    # 남은 승차가 있다면 아직 안 쓴 하차로 채움
    unused = [j for j in range(n) if j not in used]
    ui = 0
    for i in range(m):
        if mapping[i] == -1 and ui < len(unused):
            mapping[i] = unused[ui]; ui += 1
    return mapping

def build_single_vehicle_steps(starts: List[str], ends: List[str]) -> List[dict]:
    """[{'kind':'pickup'|'drop', 'name':..., 'xy':(lon,lat)} ...]"""
    def xy(label):
        r = stops.loc[stops["name"]==label]
        if r.empty: return None
        rr = r.iloc[0]; return (float(rr["lon"]), float(rr["lat"]))

    src_xy = [xy(nm) for nm in starts if xy(nm)]
    dst_xy = [xy(nm) for nm in ends if xy(nm)]
    if not src_xy or not dst_xy:
        return []

    # 승차→하차 매칭
    mapping = greedy_pairing(src_xy, dst_xy)

    # 1개 쌍으로 시작, 이후 "마지막 하차" 기준 가장 가까운 다음 승차를 선택
    remaining = list(range(len(src_xy)))
    order = []

    # 시작 승차: 사용자가 첫 번째로 선택한 승차
    cur_i = 0
    remaining.remove(cur_i)
    order += [
        {"kind":"pickup", "name": starts[cur_i], "xy": src_xy[cur_i]},
        {"kind":"drop",   "name": ends[mapping[cur_i]], "xy": dst_xy[mapping[cur_i]]},
    ]
    current_point = dst_xy[mapping[cur_i]]

    # 나머지 승차 순차 연결
    while remaining:
        # 현재 지점에서 가장 가까운 다음 승차
        nxt = min(remaining, key=lambda i: haversine(current_point, src_xy[i]))
        remaining.remove(nxt)
        order.append({"kind":"pickup", "name": starts[nxt], "xy": src_xy[nxt]})
        order.append({"kind":"drop",   "name": ends[mapping[nxt]], "xy": dst_xy[mapping[nxt]]})
        current_point = dst_xy[mapping[nxt]]

    return order

# ===================== UI =====================
col1, col2, col3 = st.columns([1.8,1.2,3.2], gap="large")

with col1:
    st.markdown('<div class="section-header">🚏 DRT 노선 추천 설정</div>', unsafe_allow_html=True)
    mode    = st.radio("운행 모드", ["차량(운행)","도보(승객 접근)"], horizontal=True)
    profile = "driving" if mode.startswith("차량") else "walking"

    all_names = stops["name"].tolist()
    starts = st.multiselect("출발(승차) 정류장", all_names, key="starts")
    ends   = st.multiselect("도착(하차) 정류장", all_names, key="ends")

    route_mode = st.radio("노선 모드", ["개별쌍(모든 조합)","단일 차량(연속 경로)"], index=1)
    order_policy = st.selectbox("순서 방식", ["가까운 우선(최근점)"], index=0)

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
    ctr_lat = float(stops["lat"].mean()); ctr_lon = float(stops["lon"].mean())
    if math.isnan(ctr_lat) or math.isnan(ctr_lon): ctr_lat, ctr_lon = 36.80, 127.15

    m = folium.Map(location=[ctr_lat, ctr_lon], zoom_start=12, tiles="CartoDB Positron", control_scale=True)
    mc = MarkerCluster().add_to(m)
    for _, r in stops.iterrows():
        folium.Marker([r["lat"], r["lon"]], tooltip=str(r["name"]), icon=folium.Icon(color="gray")).add_to(mc)

    # 실행
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

            total_min, total_km = 0.0, 0.0
            order_names = []

            if route_mode.startswith("개별쌍"):
                # 모든 (출발,도착) 조합을 각각 라우팅
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
                # 단일 차량(연속 경로) — 방문 순서 만들고, '숫자 아이콘' + 색상 규칙으로 마커 찍기
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
                    # 색상 규칙: 첫 승차(1번 pickup)=빨강, 이후 pickup=보라, drop=파랑
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
                        # 이전 지점에서 현재 지점까지 실도로 라우팅
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
