# app_cheonan_drt.py
# ---------------------------------------------------------
# 천안 DRT - 승/하차 다중 선택 → 실도로(Mapbox) 라우팅
# ---------------------------------------------------------
import streamlit as st
import geopandas as gpd
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from folium.features import DivIcon
from streamlit_folium import st_folium
from shapely.geometry import Point
from shapely.ops import unary_union
from pathlib import Path
import requests
import math

# =========================
# 0) 페이지/헤더
# =========================
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
.map-box{width:100%;height:560px;border-radius:12px;overflow:hidden;border:2px solid #e5e7eb}
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

# =========================
# 1) 토큰/기본 상수
# =========================
MAPBOX_TOKEN = st.secrets.get("MAPBOX_TOKEN", "pk.eyJ1IjoiZ3VyMDUxMDgiLCJhIjoiY21lZ2k1Y291MTdoZjJrb2k3bHc3cTJrbSJ9.DElgSQ0rPoRk1eEacPI8uQ")  # ← 여기에 secrets.toml로 넣어줘야 함
PALETTE = ["#4285f4","#34a853","#ea4335","#fbbc04","#7e57c2","#26a69a","#ef6c00","#c2185b"]

# =========================
# 2) 데이터 로더
#    - 정류장(POINT)과 경계(선택) 자동 탐색
# =========================
def _read_vector_any(basename: str):
    base = Path(".")
    for pat in (f"{basename}.shp", f"{basename}.geojson", f"{basename}.gpkg", f"{basename}.json"):
        for p in base.glob(f"**/{pat}"):
            try:
                return gpd.read_file(p)
            except Exception:
                pass
    return None

@st.cache_data
def load_stops():
    # 네 환경에 맞게 파일명/우선순위 추가 가능
    for nm in ["cb_tour","stops","poi","bus_stops","drt_points"]:
        g = _read_vector_any(nm)
        if g is not None:
            break
    if g is None:
        st.error("정류장(POINT) 레이어를 찾지 못했습니다. 예: cb_tour.shp")
        st.stop()
    g = g.to_crs(epsg=4326) if g.crs else g.set_crs(epsg=4326)
    g["lon"] = g.geometry.x
    g["lat"] = g.geometry.y
    # 이름 컬럼 추정
    name_col = None
    for c in ["name","NAME","Name","정류장명","station","st_name","title"]:
        if c in g.columns:
            name_col = c; break
    if name_col is None:
        name_col = g.columns[0]
    g = g.rename(columns={name_col:"name"})
    return g[["name","lon","lat","geometry"]]

@st.cache_data
def load_boundary():
    b = _read_vector_any("cb_shp")
    if b is None:
        return None
    return b.to_crs(epsg=4326) if b.crs else b.set_crs(epsg=4326)

stops = load_stops()
boundary = load_boundary()

# 중심점
try:
    ctr_lat = float(stops["lat"].mean()); ctr_lon = float(stops["lon"].mean())
    if math.isnan(ctr_lat) or math.isnan(ctr_lon): raise ValueError
except:
    ctr_lat, ctr_lon = 36.80, 127.15

# =========================
# 3) 레이아웃
# =========================
col1, col2, col3 = st.columns([1.6,1.2,3.2], gap="large")

# ---------- 좌측: 설정 ----------
with col1:
    st.markdown('<div class="section-header">🚏 DRT 노선 추천 설정</div>', unsafe_allow_html=True)
    mode = st.radio("운행 모드", ["차량(운행)","도보(승객 접근)"], horizontal=True)
    profile = "driving" if mode.startswith("차량") else "walking"

    st.markdown("**출발(승차) 정류장**")
    starts = st.multiselect("", stops["name"].tolist(), key="starts", label_visibility="collapsed")

    st.markdown("**도착(하차) 정류장**")
    ends   = st.multiselect("", stops["name"].tolist(), key="ends", label_visibility="collapsed")

    pairing = st.selectbox("매칭 방식", ["인덱스 쌍(1:1)", "모든 조합"], index=0)
    max_routes = st.slider("최대 생성 경로 수(과금 보호)", 1, 100, 20)

    cA, cB = st.columns(2)
    run_clicked   = cA.button("노선 추천")
    clear_clicked = cB.button("초기화")

    if clear_clicked:
        for k in ["segments","order","duration","distance","_pairs","_latlon"]:
            st.session_state.pop(k, None)
        st.experimental_rerun()

# ---------- 중간: 방문 순서/메트릭 ----------
with col2:
    st.markdown('<div class="section-header">📍 여행 방문 순서</div>', unsafe_allow_html=True)
    order_list = st.session_state.get("order", [])
    if order_list:
        for i, nm in enumerate(order_list, 1):
            st.markdown(f'<div class="visit-card"><div class="visit-num">{i}</div><div>{nm}</div></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="empty">경로 생성 후 표시됩니다</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.metric("⏱️ 소요시간", f"{st.session_state.get('duration',0.0):.1f}분")
    st.metric("📏 이동거리", f"{st.session_state.get('distance',0.0):.2f}km")

# ---------- 우측: 지도 ----------
with col3:
    st.markdown('<div class="section-header">🗺️ 추천경로 지도시각화</div>', unsafe_allow_html=True)
    m = folium.Map(location=[ctr_lat, ctr_lon], zoom_start=12, tiles="CartoDB Positron", control_scale=True)

    if boundary is not None:
        folium.GeoJson(boundary, style_function=lambda f: {"color":"#9aa0a6","weight":2,"dashArray":"4,4","fillOpacity":0.05}).add_to(m)

    mc = MarkerCluster().add_to(m)
    for _, r in stops.iterrows():
        folium.Marker([r["lat"], r["lon"]], tooltip=str(r["name"]), icon=folium.Icon(color="gray")).add_to(mc)

    # Mapbox 호출 함수
    def mapbox_route(lon1, lat1, lon2, lat2, profile="driving", token="", timeout=12):
        if not token:
            raise RuntimeError("MAPBOX_TOKEN이 없습니다. secrets.toml에 설정하세요.")
        url = f"https://api.mapbox.com/directions/v5/mapbox/{profile}/{lon1},{lat1};{lon2},{lat2}"
        params = {"geometries":"geojson","overview":"full","access_token":token}
        r = requests.get(url, params=params, timeout=timeout)
        if r.status_code != 200:
            raise RuntimeError(f"Mapbox 오류 {r.status_code}: {r.text[:160]}")
        j = r.json()
        if not j.get("routes"):
            raise RuntimeError("경로가 반환되지 않았습니다.")
        rt = j["routes"][0]
        return rt["geometry"]["coordinates"], float(rt.get("duration",0.0)), float(rt.get("distance",0.0))

    # 이름→좌표
    def xy(name: str):
        s = stops.loc[stops["name"]==name]
        if s.empty: return None
        r = s.iloc[0]
        return float(r["lon"]), float(r["lat"])

    # 실행
    if run_clicked:
        if not starts or not ends:
            st.warning("출발/도착 정류장을 각각 1개 이상 선택하세요.")
        elif not MAPBOX_TOKEN:
            st.error("MAPBOX_TOKEN이 없습니다. secrets.toml에 추가하세요.")
        else:
            # 매칭 목록 구성
            pairs = []
            if pairing.startswith("인덱스"):
                n = min(len(starts), len(ends))
                for i in range(n):
                    pairs.append((starts[i], ends[i]))
            else:  # 모든 조합
                for s in starts:
                    for e in ends:
                        pairs.append((s, e))
            if len(pairs) > max_routes:
                st.info(f"요청 경로 {len(pairs)}건 중 {max_routes}건만 생성합니다.")
                pairs = pairs[:max_routes]

            segs, total_sec, total_m, latlon_all = [], 0.0, 0.0, []
            order_names = []
            for i, (S, E) in enumerate(pairs):
                sxy, exy = xy(S), xy(E)
                if sxy is None or exy is None:
                    st.warning(f"좌표 없음: {S} → {E}")
                    continue
                try:
                    coords, dur, dist = mapbox_route(sxy[0], sxy[1], exy[0], exy[1], profile=profile, token=MAPBOX_TOKEN)
                    segs.append(coords); total_sec += dur; total_m += dist
                    ll = [(c[1], c[0]) for c in coords]
                    folium.PolyLine(ll, color=PALETTE[i % len(PALETTE)], weight=5, opacity=0.85).add_to(m)
                    # 라벨(중간점)
                    mid = ll[len(ll)//2]
                    folium.map.Marker(mid, icon=DivIcon(html=f"<div style='background:{PALETTE[i%len(PALETTE)]};color:#fff;border-radius:50%;width:26px;height:26px;line-height:26px;text-align:center;font-weight:700;'>{i+1}</div>")).add_to(m)
                    # 시작/끝 마커
                    folium.Marker([sxy[1], sxy[0]], icon=folium.Icon(color="red"), tooltip=f"승차: {S}").add_to(m)
                    folium.Marker([exy[1], exy[0]], icon=folium.Icon(color="blue"), tooltip=f"하차: {E}").add_to(m)
                    latlon_all += ll
                    order_names += [f"{S} → {E}"]
                except Exception as e:
                    st.warning(f"{S} → {E} 실패: {e}")

            # 상태 저장 + 메트릭
            if segs:
                st.session_state["segments"] = segs
                st.session_state["order"]    = order_names
                st.session_state["duration"] = total_sec/60
                st.session_state["distance"] = total_m/1000
                # fit bounds
                if latlon_all:
                    m.fit_bounds([[min(y for y,x in latlon_all), min(x for y,x in latlon_all)],
                                  [max(y for y,x in latlon_all), max(x for y,x in latlon_all)]])
                st.success("✅ 노선이 생성되었습니다!")

    # 지도 렌더
    st.markdown('<div class="map-box">', unsafe_allow_html=True)
    st_folium(m, width="100%", height=560, returned_objects=[], use_container_width=True, key="main_map")
    st.markdown('</div>', unsafe_allow_html=True)
