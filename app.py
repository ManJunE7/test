# app.py  — 단일 차량 연속 경로 지원 (Directions API만 사용)

import os, math, re
from pathlib import Path
from typing import List, Tuple, Optional

import geopandas as gpd
import pandas as pd
import numpy as np
import requests
import streamlit as st
import folium
from folium.plugins import MarkerCluster
from folium.features import DivIcon
from streamlit_folium import st_folium

APP_TITLE = "천안 DRT - 맞춤형 AI기반 스마트 교통 가이드"
LOGO_URL  = "https://raw.githubusercontent.com/JeongWon4034/cheongju/main/cheongpung_logo.png"

st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Noto Sans KR', -apple-system, BlinkMacSystemFont, sans-serif; }
.header-container{display:flex;align-items:center;justify-content:center;gap:20px;margin-bottom:1.2rem;padding:0.6rem 0;}
.logo-image{width:72px;height:72px;object-fit:contain}
.main-title{font-size:2.1rem;font-weight:800;color:#202124;letter-spacing:-0.5px;margin:0}
.title-underline{width:100%;height:3px;background:linear-gradient(90deg,#4285f4,#34a853);margin:0 auto 1rem;border-radius:2px;}
.section-header{font-size:1.05rem;font-weight:800;color:#1f2937;margin-bottom:10px;padding-bottom:8px;border-bottom:2px solid #f3f4f6}
.stButton > button{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:#fff;border:none;border-radius:10px;padding:10px 18px;font-size:.9rem;font-weight:700;box-shadow:0 4px 8px rgba(102,126,234,.3)}
.stButton > button:hover{transform:translateY(-1px);box-shadow:0 6px 14px rgba(102,126,234,.4)}
.visit-card{display:flex;align-items:center;gap:10px;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:#fff;border-radius:12px;padding:10px 12px;margin-bottom:8px;box-shadow:0 2px 4px rgba(102,126,234,.3)}
.visit-num{background:#fff;color:#667eea;width:24px;height:24px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:800;font-size:.8rem}
.empty{color:#9ca3af;background:linear-gradient(135deg,#ffecd2 0%,#fcb69f 100%);border-radius:12px;padding:22px 14px;text-align:center}
.table-note{font-size:.85rem;color:#6b7280;margin-top:.25rem}
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

# ------------------------- 토큰 -------------------------
MAPBOX_TOKEN = os.getenv("MAPBOX_TOKEN", "pk.eyJ1IjoiZ3VyMDUxMDgiLCJhIjoiY21lbWppYjByMDV2ajJqcjQyYXUxdzY3byJ9.yLBRJK_Ib6W3p9f16YlIKQ")
try:
    if not MAPBOX_TOKEN:
        MAPBOX_TOKEN = st.secrets["MAPBOX_TOKEN"]
except Exception:
    pass
# 직접 넣고 싶으면 아래 주석을 풀고 사용
# MAPBOX_TOKEN = "여기에_본인_토큰"

PALETTE = ["#4285f4","#34a853","#ea4335","#fbbc04","#7e57c2","#26a69a","#ef6c00","#c2185b"]
DATA_STEM = "new_new_drt_full_utf8"   # 같은 폴더에 .shp/.gpkg/.geojson 중 하나

# ------------------------- 유틸 -------------------------
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
    try:
        os.environ["SHAPE_ENCODING"] = "UTF-8"
        return gpd.read_file(path, engine="fiona")
    except Exception as e:
        raise e

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
    g = _open_any().copy()
    if "jibun" not in g.columns:
        st.error("소스에 'jibun' 필드가 없습니다."); st.stop()
    g["name"] = (g["name"] if "name" in g.columns else g["jibun"]).astype(str).str.strip()
    g["jibun"] = g["jibun"].astype(str).str.strip()
    g["lon"]   = g.geometry.x
    g["lat"]   = g.geometry.y
    st.caption(f"데이터셋: {DATA_STEM} (포인트 {len(g)}개 · UTF-8)")
    return g[["jibun","name","lon","lat","geometry"]]

stops = load_stops()

# ------------------------- Mapbox Directions -------------------------
def mapbox_route(lon1,lat1,lon2,lat2, profile="driving", token="", timeout=12):
    if not token: raise RuntimeError("MAPBOX_TOKEN 필요")
    url=f"https://api.mapbox.com/directions/v5/mapbox/{profile}/{lon1},{lat1};{lon2},{lat2}"
    params={"geometries":"geojson","overview":"full","access_token":token}
    r=requests.get(url,params=params,timeout=timeout)
    if r.status_code!=200: raise RuntimeError(f"Directions 오류 {r.status_code}: {r.text[:160]}")
    j=r.json(); routes=j.get("routes",[])
    if not routes: raise RuntimeError("경로가 반환되지 않았습니다.")
    rt=routes[0]; return rt["geometry"]["coordinates"], float(rt.get("duration",0.0)), float(rt.get("distance",0.0))

# ------------------------- 이름 제안(원하면 유지) -------------------------
def _mbx_geocode(lon, lat, types="poi,intersection,address", limit=10, language="ko"):
    if not MAPBOX_TOKEN: return []
    url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{lon},{lat}.json"
    params = {"access_token": MAPBOX_TOKEN, "types": types, "limit": limit, "language": language, "proximity": f"{lon},{lat}"}
    try:
        r = requests.get(url, params=params, timeout=8)
        if r.status_code != 200: return []
        return r.json().get("features", []) or []
    except Exception:
        return []

def _clean_text_ko(s: str) -> str:
    if not s: return ""
    s = str(s)
    for bad in ["대한민국","대한민국 ", "South Korea", "Republic of Korea"]:
        s = s.replace(bad,"")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def pick_stop_name(lon: float, lat: float) -> Optional[str]:
    feats = _mbx_geocode(lon, lat)
    if not feats: return None
    # 간단히 첫 후보 사용 (원하면 가중치 로직 유지 가능)
    return _clean_text_ko(feats[0].get("text_ko") or feats[0].get("text"))

# ------------------------- 연속 경로용 헬퍼 -------------------------
def nearest_neighbor_order(coords: List[Tuple[float,float]], start_idx: int = 0) -> List[int]:
    """coords: (lon,lat) 목록. start_idx에서 시작, 최근접 이웃 순회 순서 반환"""
    n = len(coords)
    if n <= 1: return list(range(n))
    unvisited = set(range(n)); unvisited.remove(start_idx)
    order = [start_idx]; cur = start_idx
    while unvisited:
        nxt = min(unvisited, key=lambda j: haversine(coords[cur], coords[j]))
        order.append(nxt); unvisited.remove(nxt); cur = nxt
    return order

def coord_of_name(nm: str):
    row = stops.loc[stops["name"]==nm]
    if row.empty: return None
    rr = row.iloc[0]
    return float(rr["lon"]), float(rr["lat"])

# ------------------------- UI -------------------------
col1, col2, col3 = st.columns([1.9,1.2,3.2], gap="large")

with col1:
    st.markdown('<div class="section-header">🚏 DRT 노선 추천 설정</div>', unsafe_allow_html=True)
    drive_mode = st.radio("운행 모드", ["차량(운행)","도보(승객 접근)"], horizontal=True)
    profile = "driving" if drive_mode.startswith("차량") else "walking"

    all_names = stops["name"].tolist()
    starts = st.multiselect("출발(승차) 정류장", all_names, key="starts")
    ends   = st.multiselect("도착(하차) 정류장", all_names, key="ends")

    route_mode = st.radio("노선 모드", ["개별쌍(모든 조합)","단일 차량(연속 경로)"], horizontal=False)

    seq_order_mode = None
    if route_mode == "단일 차량(연속 경로)":
        seq_order_mode = st.selectbox("순서 방식", ["선택 순서 그대로", "가까운 곳 우선(최근접)"], index=1)

    # 선택/초기화
    cA, cB, cC = st.columns(3)
    run_clicked   = cA.button("노선 추천")
    clear_clicked = cB.button("초기화")
    if cC.button("캐시 초기화"): st.cache_data.clear(); st.rerun()
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

    if run_clicked:
        if not MAPBOX_TOKEN:
            st.error("MAPBOX_TOKEN을 환경변수 또는 secrets에 설정하세요.")
        elif not starts or (route_mode=="개별쌍(모든 조합)" and not ends) or (route_mode=="단일 차량(연속 경로)" and not (ends or len(starts)>1)):
            st.warning("필요한 정류장을 선택하세요.")
        else:
            total_min, total_km = 0.0, 0.0

            if route_mode == "개별쌍(모든 조합)":
                # 모든 (start, end) 조합을 각자 그리기 (이전 동작)
                pairs=[]
                for i,snm in enumerate(starts):
                    for j,enm in enumerate(ends):
                        pairs.append((snm, enm))
                st.session_state["order"] = [f"{a} → {b}" for a,b in pairs]

                for idx,(snm,enm) in enumerate(pairs):
                    sxy, exy = coord_of_name(snm), coord_of_name(enm)
                    if not sxy or not exy: continue
                    try:
                        coords, dur, dist = mapbox_route(sxy[0], sxy[1], exy[0], exy[1], profile=profile, token=MAPBOX_TOKEN)
                        ll = [(c[1], c[0]) for c in coords]
                        folium.PolyLine(ll, color=PALETTE[idx % len(PALETTE)], weight=5, opacity=0.9).add_to(m)
                        total_min += dur/60; total_km += dist/1000
                    except Exception as e:
                        st.warning(f"{snm}→{enm} 실패: {e}")

            else:
                # 단일 차량 연속 경로
                # 출발지는 '출발(승차)'의 첫 번째. 나머지(나머지 승차 + 모든 하차)를 연속 방문.
                start_name = starts[0]
                start_xy   = coord_of_name(start_name)
                if not start_xy:
                    st.warning("출발지 좌표를 찾을 수 없습니다.")
                else:
                    pool_names = list(dict.fromkeys(starts[1:] + ends))  # 중복 제거, 순서 유지
                    pool_xy    = [coord_of_name(nm) for nm in pool_names if coord_of_name(nm)]
                    pool_names = [nm for nm,xy in zip(pool_names,pool_xy) if xy]  # 좌표 없는 항목 제거

                    if not pool_xy:
                        st.warning("방문할 다음 정류장이 없습니다.")
                    else:
                        # 순서 결정
                        if seq_order_mode == "선택 순서 그대로":
                            order_idx = list(range(len(pool_xy)))
                        else:
                            coords = [start_xy] + pool_xy
                            nn_order = nearest_neighbor_order(coords, start_idx=0)[1:]  # 0=출발지 제외
                            order_idx = nn_order

                        visit_names = [start_name] + [pool_names[i] for i in order_idx]
                        st.session_state["order"] = visit_names

                        # 연속 구간을 차례대로 Directions 호출
                        seg_idx = 0
                        cur_xy = start_xy
                        for next_nm in visit_names[1:]:
                            nxt_xy = coord_of_name(next_nm)
                            if not nxt_xy: continue
                            try:
                                coords, dur, dist = mapbox_route(cur_xy[0], cur_xy[1], nxt_xy[0], nxt_xy[1], profile=profile, token=MAPBOX_TOKEN)
                                ll = [(c[1], c[0]) for c in coords]
                                folium.PolyLine(ll, color=PALETTE[seg_idx % len(PALETTE)], weight=5, opacity=0.9).add_to(m)
                                total_min += dur/60; total_km += dist/1000
                                cur_xy = nxt_xy; seg_idx += 1
                            except Exception as e:
                                st.warning(f"연속 구간 실패({next_nm}): {e}")

            st.session_state["duration"] = total_min
            st.session_state["distance"] = total_km

    st_folium(m, height=560, returned_objects=[], use_container_width=True, key="main_map")
