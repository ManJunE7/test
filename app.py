# app_cheonan_drt_best_only_rg.py
# ---------------------------------------------------------
# 천안 DRT - 최적 1개 라우팅 + 역지오코딩으로 이름 매핑
# ---------------------------------------------------------
import math
from pathlib import Path
from glob import glob
import geopandas as gpd
import pandas as pd
import requests
import streamlit as st
from folium.features import DivIcon
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import folium

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

# ---- 토큰/상수
MAPBOX_TOKEN = "여기에_네_토큰"   # 실제 토큰으로 교체
PALETTE = ["#4285f4"]
MATRIX_MAX_COORDS = 25

# ---- 정류장 로더 (동일)
def _find_first(glob_pattern: str):
    try: return next(Path(".").rglob(glob_pattern))
    except StopIteration: return None

def _pick_name_col(df: pd.DataFrame):
    for c in ["name","NAME","Name","정류장명","정류장","stop_name","station","st_name","bus_stop_nm","bus_stops"]:
        if c in df.columns: return c
    return None

@st.cache_data
def load_stops():
    ff = _find_first("ff_drt_dh.shp")
    if ff:
        g = gpd.read_file(ff)
        g = g.to_crs(epsg=4326) if g.crs else g.set_crs(epsg=4326)
        use = None
        cand = [c for c in g.columns if c.lower() == "bus_stops"]
        if cand:
            c = cand[0]
            use = g[g[c].astype(str).str.strip().str.lower().isin(["1","true","y","yes","bus_stops"])]
        if use is None or use.empty:
            cat_cols = [c for c in g.columns if c.lower() in ("layer","type","category","class","feature","theme","kind","group","분류","구분","시설구분")]
            for c in cat_cols:
                m = g[c].astype(str).str.lower().str.contains(r"bus[\s_\-]*stop", na=False)
                if m.any(): use = g[m]; break
        if use is None or use.empty:
            use = g[g.geom_type.astype(str).str.contains("Point", case=False, na=False)]
        if use.empty: st.error("ff_drt_dh.shp에서 bus_stops 포인트를 찾지 못했습니다."); st.stop()
        name_col = _pick_name_col(use) or "name"
        if name_col == "name" and "name" not in use.columns:
            use["name"] = [f"정류장_{i+1}" for i in range(len(use))]
        use["lon"], use["lat"] = use.geometry.x, use.geometry.y
        return use.rename(columns={name_col:"name"})[["name","lon","lat","geometry"]]
    # 폴백
    candidates = []
    for bn in ["cb_tour","stops","poi","bus_stops","drt_points"]:
        candidates += glob(f"**/{bn}.shp", recursive=True)
        candidates += glob(f"**/{bn}.geojson", recursive=True)
        candidates += glob(f"**/{bn}.gpkg", recursive=True)
        candidates += glob(f"**/{bn}.json", recursive=True)
    for p in sorted(set(candidates)):
        try:
            g = gpd.read_file(p)
            g = g.to_crs(epsg=4326) if g.crs else g.set_crs(epsg=4326)
            pts = g[g.geom_type.astype(str).str.contains("Point", case=False, na=False)].copy()
            if pts.empty: continue
            name_col = _pick_name_col(pts) or "name"
            if name_col == "name" and "name" not in pts.columns:
                pts["name"] = [f"정류장_{i+1}" for i in range(len(pts))]
            pts["lon"], pts["lat"] = pts.geometry.x, pts.geometry.y
            return pts.rename(columns={name_col:"name"})[["name","lon","lat","geometry"]]
        except Exception:
            continue
    st.error("정류장(POINT) 레이어를 찾지 못했습니다."); st.stop()

@st.cache_data
def load_boundary():
    for nm in ["cb_shp","boundary","admin_boundary","cheonan_boundary"]:
        for ext in ["shp","geojson","gpkg","json"]:
            p = _find_first(f"**/{nm}.{ext}")
            if p:
                g = gpd.read_file(p)
                return g.to_crs(epsg=4326) if g.crs else g.set_crs(epsg=4326)
    return None

stops = load_stops()
boundary = load_boundary()

# ---- 중심
try:
    ctr_lat, ctr_lon = float(stops["lat"].mean()), float(stops["lon"].mean())
    if math.isnan(ctr_lat) or math.isnan(ctr_lon): raise ValueError
except:
    ctr_lat, ctr_lon = 36.80, 127.15

# ---- Mapbox API
def mapbox_route(lon1, lat1, lon2, lat2, profile="driving", token="", timeout=12):
    if not token: raise RuntimeError("MAPBOX_TOKEN 필요")
    url = f"https://api.mapbox.com/directions/v5/mapbox/{profile}/{lon1},{lat1};{lon2},{lat2}"
    params = {"geometries":"geojson","overview":"full","access_token":token}
    r = requests.get(url, params=params, timeout=timeout)
    if r.status_code != 200: raise RuntimeError(f"Mapbox 오류 {r.status_code}: {r.text[:160]}")
    j = r.json(); routes = j.get("routes", [])
    if not routes: raise RuntimeError("경로가 반환되지 않았습니다.")
    rt = routes[0]
    return rt["geometry"]["coordinates"], float(rt.get("duration",0.0)), float(rt.get("distance",0.0))

def mapbox_matrix(sources_xy, destinations_xy, profile="driving", token="", timeout=12):
    if not token: raise RuntimeError("MAPBOX_TOKEN 필요")
    coords = sources_xy + destinations_xy
    if len(coords) > MATRIX_MAX_COORDS:
        raise RuntimeError(f"Matrix 좌표 총합 {len(coords)}개 — {MATRIX_MAX_COORDS} 이하로 줄여주세요.")
    coord_str = ";".join([f"{x},{y}" for x,y in coords])
    src_idx = ";".join(map(str, range(len(sources_xy))))
    dst_idx = ";".join(map(str, range(len(sources_xy), len(coords))))
    url = f"https://api.mapbox.com/directions-matrix/v1/mapbox/{profile}/{coord_str}"
    params = {"access_token": token, "annotations": "duration,distance", "sources": src_idx, "destinations": dst_idx}
    r = requests.get(url, params=params, timeout=timeout)
    if r.status_code != 200: raise RuntimeError(f"Matrix 오류 {r.status_code}: {r.text[:160]}")
    j = r.json()
    return j.get("durations"), j.get("distances")

def mapbox_reverse_name(lon, lat, token="", lang="ko"):
    """근처 POI/주소/도로명 중 가장 그럴듯한 하나를 반환"""
    if not token: return None
    url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{lon},{lat}.json"
    params = {
        "access_token": token,
        "language": lang,
        "types": "poi,address,street,place,neighborhood"
    }
    r = requests.get(url, params=params, timeout=10)
    if r.status_code != 200: return None
    j = r.json()
    feats = j.get("features", [])
    if not feats: return None
    # POI > address > street 우선
    def score(ft):
        t = ft.get("place_type", [""])[0]
        order = {"poi":0, "address":1, "street":2, "place":3, "neighborhood":4}
        return order.get(t, 9)
    feats.sort(key=score)
    ft = feats[0]
    # 짧은 레이블 우선 (text), 없으면 place_name
    return ft.get("text") or ft.get("place_name")

# ---- UI
col1, col2, col3 = st.columns([1.6,1.2,3.2], gap="large")

with col1:
    st.markdown('<div class="section-header">🚏 DRT 노선 추천 설정</div>', unsafe_allow_html=True)
    mode = st.radio("운행 모드", ["차량(운행)","도보(승객 접근)"], horizontal=True)
    profile = "driving" if mode.startswith("차량") else "walking"
    st.markdown("**출발(승차) 정류장**")
    starts = st.multiselect("", stops["name"].tolist(), key="starts", label_visibility="collapsed")
    st.markdown("**도착(하차) 정류장**")
    ends   = st.multiselect("", stops["name"].tolist(), key="ends", label_visibility="collapsed")
    st.caption(f"선택한 조합 수(평가): {len(starts)} × {len(ends)} → {len(starts)*len(ends)}개")
    cA, cB = st.columns(2)
    run_clicked   = cA.button("노선 추천")
    clear_clicked = cB.button("초기화")
    if clear_clicked:
        for k in ["segments","order","duration","distance","best_pair"]:
            st.session_state.pop(k, None)
        st.rerun()

with col2:
    st.markdown('<div class="section-header">📍 방문 순서</div>', unsafe_allow_html=True)
    if "order" in st.session_state and st.session_state["order"]:
        for i, nm in enumerate(st.session_state["order"], 1):
            st.markdown(f'<div class="visit-card"><div class="visit-num">{i}</div><div>{nm}</div></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="empty">경로 생성 후 표시됩니다</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.metric("⏱️ 소요시간", f"{st.session_state.get('duration',0.0):.1f}분")
    st.metric("📏 이동거리", f"{st.session_state.get('distance',0.0):.2f}km")

with col3:
    st.markdown('<div class="section-header">🗺️ 추천경로 지도시각화</div>', unsafe_allow_html=True)
    m = folium.Map(location=[ctr_lat, ctr_lon], zoom_start=12, tiles="CartoDB Positron", control_scale=True)
    if boundary is not None:
        folium.GeoJson(boundary, style_function=lambda f: {"color":"#9aa0a6","weight":2,"dashArray":"4,4","fillOpacity":0.05}).add_to(m)
    mc = MarkerCluster().add_to(m)
    for _, r in stops.iterrows():
        folium.Marker([r["lat"], r["lon"]], tooltip=str(r["name"]), icon=folium.Icon(color="gray")).add_to(mc)

    def xy(name: str):
        s = stops.loc[stops["name"]==name]
        if s.empty: return None
        r = s.iloc[0]; return float(r["lon"]), float(r["lat"])

    if run_clicked:
        if not starts or not ends:
            st.warning("출발/도착 정류장을 각각 1개 이상 선택하세요.")
        elif (len(starts)+len(ends)) > MATRIX_MAX_COORDS:
            st.warning(f"선택 좌표 총합이 {len(starts)+len(ends)}개입니다. {MATRIX_MAX_COORDS}개 이하로 줄여주세요.")
        elif not MAPBOX_TOKEN:
            st.error("MAPBOX_TOKEN을 코드 상단에 입력하세요.")
        else:
            src_xy = [xy(nm) for nm in starts if xy(nm)]
            dst_xy = [xy(nm) for nm in ends if xy(nm)]
            if not src_xy or not dst_xy:
                st.warning("유효한 좌표가 없습니다.")
            else:
                durations, distances = mapbox_matrix(src_xy, dst_xy, profile=profile, token=MAPBOX_TOKEN)
                best_s, best_d, best_val, best_dist = None, None, float("inf"), float("inf")
                for i in range(len(src_xy)):
                    for j in range(len(dst_xy)):
                        dur = None if not durations or not durations[i] else durations[i][j]
                        if dur is None: continue
                        if dur < best_val:
                            best_val = dur; best_s, best_d = i, j
                            if distances and distances[i]:
                                best_dist = distances[i][j] if distances[i][j] is not None else float("inf")
                        elif dur == best_val and distances and distances[i]:
                            d2 = distances[i][j] if distances[i][j] is not None else float("inf")
                            if d2 < best_dist:
                                best_dist = d2; best_s, best_d = i, j
                if best_s is None:
                    st.warning("유효한 최적 조합을 찾지 못했습니다.")
                else:
                    # 매핑 이름 생성 (역지오코딩)
                    sxy, exy = src_xy[best_s], dst_xy[best_d]
                    s_label = mapbox_reverse_name(sxy[0], sxy[1], token=MAPBOX_TOKEN) or starts[best_s]
                    e_label = mapbox_reverse_name(exy[0], exy[1], token=MAPBOX_TOKEN) or ends[best_d]

                    # 실제 경로 1개
                    coords, dur, dist = mapbox_route(sxy[0], sxy[1], exy[0], exy[1], profile=profile, token=MAPBOX_TOKEN)
                    ll = [(c[1], c[0]) for c in coords]
                    folium.PolyLine(ll, color=PALETTE[0], weight=5, opacity=0.85).add_to(m)
                    mid = ll[len(ll)//2]
                    folium.map.Marker(mid, icon=DivIcon(html=f"<div style='background:{PALETTE[0]};color:#fff;border-radius:50%;width:26px;height:26px;line-height:26px;text-align:center;font-weight:700;'>★</div>")).add_to(m)
                    folium.Marker([sxy[1], sxy[0]], icon=folium.Icon(color="red"), tooltip=f"승차: {s_label}").add_to(m)
                    folium.Marker([exy[1], exy[0]], icon=folium.Icon(color="blue"), tooltip=f"하차: {e_label}").add_to(m)
                    m.fit_bounds([[min(y for y,x in ll), min(x for y,x in ll)],
                                  [max(y for y,x in ll), max(x for y,x in ll)]])
                    st.session_state["order"]    = [f"{s_label} → {e_label} (최적)"]
                    st.session_state["duration"] = dur/60
                    st.session_state["distance"] = dist/1000

    st_folium(m, height=560, returned_objects=[], use_container_width=True, key="main_map")
