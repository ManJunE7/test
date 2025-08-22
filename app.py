# app_cheonan_drt_new_drt_first.py
# ---------------------------------------------------------
# DRT 노선 추천 (new_new_drt.* 최우선)
# - new_new_drt.shp / new_new_drt.csv 우선 사용
# - 역지오코딩 한글 라벨(POI/주소/도로명 + 카테고리)
# - 과금보호: 상위 N개만 Directions 호출
# ---------------------------------------------------------
import os
import math
from pathlib import Path
from glob import glob

import geopandas as gpd
import pandas as pd
import numpy as np
import requests
import streamlit as st
import folium
from folium.plugins import MarkerCluster
from folium.features import DivIcon
from shapely.geometry import Point
from streamlit_folium import st_folium

# ===================== 기본 설정 / 스타일 =====================
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

# ===================== 토큰 / 상수 =====================
MAPBOX_TOKEN = "pk.eyJ1IjoiZ3VyMDUxMDgiLCJhIjoiY21lZ2k1Y291MTdoZjJrb2k3bHc3cTJrbSJ9.DElgSQ0rPoRk1eEacPI8uQ"  # << 여기 직접 넣거나, 환경변수/Secrets에 MAPBOX_TOKEN 설정
if not MAPBOX_TOKEN:
    MAPBOX_TOKEN = os.getenv("MAPBOX_TOKEN", "")
if not MAPBOX_TOKEN:
    try:
        MAPBOX_TOKEN = st.secrets["MAPBOX_TOKEN"]
    except Exception:
        pass

PALETTE = ["#4285f4","#34a853","#ea4335","#fbbc04","#7e57c2","#26a69a","#ef6c00","#c2185b"]
MATRIX_MAX_COORDS = 25  # Matrix 좌표 총합 권장 상한

# ===================== 좌표계 보정 유틸 =====================
KOREA_CRS_CANDIDATES = ["EPSG:5179","EPSG:5181","EPSG:5186","EPSG:5187","EPSG:2097","EPSG:32651","EPSG:32652","EPSG:32653"]

def _looks_like_korea(gdf_wgs84: gpd.GeoDataFrame) -> bool:
    if gdf_wgs84.empty: return False
    xs, ys = gdf_wgs84.geometry.x, gdf_wgs84.geometry.y
    ok = ((xs > 120) & (xs < 135) & (ys > 30) & (ys < 45)).mean()
    return ok > 0.9

def to_wgs84_auto(points_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    graw = points_gdf.copy()
    if graw.crs:
        try:
            w = graw.to_crs(epsg=4326)
            if _looks_like_korea(w): return w
        except: pass
    for cand in KOREA_CRS_CANDIDATES:
        try:
            tmp = graw.copy().set_crs(cand, allow_override=True).to_crs(epsg=4326)
            if _looks_like_korea(tmp): return tmp
        except: continue
    w = graw.copy()
    w = w.set_crs(epsg=4326) if not w.crs else w.to_crs(epsg=4326)
    return w

def _is_text_series(s: pd.Series) -> bool:
    return s.dtype == "object" or s.dtype.name.startswith("string")

# ===================== 데이터 로더( new_new_drt.* 최우선 ) =====================
def _find_first(glob_pattern: str):
    try: return next(Path(".").rglob(glob_pattern))
    except StopIteration: return None

def _pick_name_col(df: pd.DataFrame):
    for c in ["name","정류장명","정류장","stop_name","station","st_name","poi_name","label","title","NAME","Name"]:
        if c in df.columns and _is_text_series(df[c]): return c
    return None

def _guess_lon_lat_cols(df: pd.DataFrame):
    lon_keys = ["lon","lng","long","x","경도","LON","LNG","LONG","X"]
    lat_keys = ["lat","y","위도","LAT","Y"]
    lon_col = next((c for c in df.columns if str(c) in lon_keys), None)
    lat_col = next((c for c in df.columns if str(c) in lat_keys), None)
    # 조금 더 공격적으로 탐색(케이스 무시)
    if lon_col is None:
        for c in df.columns:
            s = str(c).lower()
            if s in ["lon","lng","long","x","경도"]:
                lon_col = c; break
    if lat_col is None:
        for c in df.columns:
            s = str(c).lower()
            if s in ["lat","y","위도"]:
                lat_col = c; break
    return lon_col, lat_col

def _gdf_from_csv(p: Path) -> gpd.GeoDataFrame:
    # 인코딩 추정
    for enc in ["utf-8-sig","cp949","euc-kr","utf-8"]:
        try:
            df = pd.read_csv(p, encoding=enc)
            break
        except Exception:
            df = None
    if df is None or df.empty:
        raise RuntimeError(f"CSV 로드 실패: {p}")

    lon_col, lat_col = _guess_lon_lat_cols(df)
    if lon_col is None or lat_col is None:
        raise RuntimeError("CSV에서 경도/위도 컬럼을 찾지 못했습니다. (예: lon/lat, x/y, 경도/위도)")

    # 숫자화
    df = df.copy()
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df = df.dropna(subset=[lon_col, lat_col])

    g = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df[lon_col], df[lat_col])], crs="EPSG:4326")
    # 범위가 이상하면 한국 TM로 가정해 변환
    xs, ys = g.geometry.x, g.geometry.y
    if not (((xs > 120) & (xs < 135) & (ys > 30) & (ys < 45))).all():
        # TM 좌표로 보일 경우(값이 수십~수백 km 규모), 5179 추정
        g = g.set_crs(epsg=5179, allow_override=True).to_crs(epsg=4326)
    return g

@st.cache_data
def load_stops():
    # 1) new_new_drt.shp 최우선
    shp = _find_first("new_new_drt.shp")
    if shp:
        g0 = gpd.read_file(shp)
        pts = g0[g0.geom_type.astype(str).str.contains("Point", case=False, na=False)]
        if pts.empty:
            # 라인/폴리곤만 있으면 중심점으로 대체
            g0 = g0.copy()
            g0["geometry"] = g0.geometry.representative_point()
            pts = g0
        g = to_wgs84_auto(pts)
        name_col = _pick_name_col(g) or None
        if name_col is None:
            g["name"] = [f"정류장_{i+1}" for i in range(len(g))]
            name_col = "name"
        g = g.rename(columns={name_col:"name"})
        g["lon"], g["lat"] = g.geometry.x, g.geometry.y
        # sanity
        if not ((g["lon"].between(-180,180)) & (g["lat"].between(-90,90))).all():
            st.error("new_new_drt.shp 좌표가 경위도 범위를 벗어납니다. 원본 CRS 확인 필요."); st.stop()
        st.caption(f"데이터셋: new_new_drt.shp · {len(g)}개 포인트 사용")
        return g[["name","lon","lat","geometry"]]

    # 2) new_new_drt.csv 다음 우선
    csvp = _find_first("new_new_drt.csv")
    if csvp:
        g = _gdf_from_csv(csvp)
        name_col = _pick_name_col(g) or None
        if name_col is None:
            g["name"] = [f"정류장_{i+1}" for i in range(len(g))]
            name_col = "name"
        g = g.rename(columns={name_col:"name"})
        g["lon"], g["lat"] = g.geometry.x, g.geometry.y
        st.caption(f"데이터셋: new_new_drt.csv · {len(g)}개 포인트 사용")
        return g[["name","lon","lat","geometry"]]

    # 3) 백업: ff_drt_dh.shp (bus_stops 필터 → 없으면 포인트 전체)
    ff = _find_first("ff_drt_dh.shp")
    if ff:
        g0 = gpd.read_file(ff)
        use = None
        # bus_stops 컬럼 탐색
        cand_cols = [c for c in g0.columns if c.lower()=="bus_stops"]
        if cand_cols:
            c = cand_cols[0]
            s = g0[c]
            if pd.api.types.is_bool_dtype(s):
                mask = s.fillna(False)
            elif pd.api.types.is_numeric_dtype(s):
                mask = s.fillna(0).astype(float) > 0
            else:
                mask = s.astype(str).str.strip().str.lower().isin(
                    ["1","true","y","yes","t","on","bus_stops","ok"]
                )
            use = g0[mask]
        # 분류 컬럼에서 bus stop 문자열
        if use is None or use.empty:
            cat_cols = [c for c in g0.columns if c.lower() in ("layer","type","category","class","feature","theme","kind","group","분류","구분","시설구분")]
            for c in cat_cols:
                m = g0[c].astype(str).str.lower().str.contains(r"bus[\s_\-]*stop", na=False)
                if m.any():
                    use = g0[m]; break
        # 최후: 포인트
        if use is None or use.empty:
            use = g0[g0.geom_type.astype(str).str.contains("Point", case=False, na=False)]
        if use.empty:
            st.error("ff_drt_dh.shp에서 버스 정류장 포인트를 찾지 못했습니다."); st.stop()
        g = to_wgs84_auto(use)

        name_col = _pick_name_col(g) or None
        if name_col is None:
            g["name"] = [f"정류장_{i+1}" for i in range(len(g))]
            name_col = "name"
        g = g.rename(columns={name_col:"name"})
        g["lon"], g["lat"] = g.geometry.x, g.geometry.y
        if not ((g["lon"].between(-180,180)) & (g["lat"].between(-90,90))).all():
            st.error("좌표 변환 실패(경위도 범위 벗어남)."); st.stop()
        st.caption(f"데이터셋: ff_drt_dh.shp · {len(g)}개 포인트 사용")
        return g[["name","lon","lat","geometry"]]

    # 4) 마지막 백업: 기타 후보
    candidates = []
    for bn in ["cb_tour","stops","poi","bus_stops","drt_points"]:
        candidates += glob(f"**/{bn}.shp", recursive=True)
        candidates += glob(f"**/{bn}.geojson", recursive=True)
        candidates += glob(f"**/{bn}.gpkg", recursive=True)
        candidates += glob(f"**/{bn}.json", recursive=True)
    for p in sorted(set(candidates)):
        try:
            g0 = gpd.read_file(p)
            pts = g0[g0.geom_type.astype(str).str.contains("Point", case=False, na=False)]
            if pts.empty: continue
            g = to_wgs84_auto(pts)
            name_col = _pick_name_col(g) or None
            if name_col is None:
                g["name"] = [f"정류장_{i+1}" for i in range(len(g))]
                name_col = "name"
            g = g.rename(columns={name_col:"name"})
            g["lon"], g["lat"] = g.geometry.x, g.geometry.y
            st.caption(f"데이터셋: {Path(p).name} · {len(g)}개 포인트 사용")
            return g[["name","lon","lat","geometry"]]
        except:
            continue

    st.error("정류장(포인트) 데이터를 찾지 못했습니다. new_new_drt.shp/csv를 확인해주세요.")
    st.stop()

@st.cache_data
def load_boundary():
    for nm in ["cb_shp","boundary","admin_boundary","cheonan_boundary"]:
        for ext in ["shp","geojson","gpkg","json"]:
            p = _find_first(f"**/{nm}.{ext}")
            if p:
                g0 = gpd.read_file(p)
                return g0.to_crs(epsg=4326) if g0.crs else to_wgs84_auto(g0)
    return None

stops = load_stops()
boundary = load_boundary()

# 중심
ctr_lat = float(stops["lat"].mean()); ctr_lon = float(stops["lon"].mean())
if math.isnan(ctr_lat) or math.isnan(ctr_lon): ctr_lat, ctr_lon = 36.80, 127.15

# ===================== Mapbox API 래퍼 =====================
def mapbox_route(lon1, lat1, lon2, lat2, profile="driving", token="", timeout=12):
    if not token: raise RuntimeError("MAPBOX_TOKEN 필요")
    url = f"https://api.mapbox.com/directions/v5/mapbox/{profile}/{lon1},{lat1};{lon2},{lat2}"
    params = {"geometries":"geojson","overview":"full","access_token":token}
    r = requests.get(url, params=params, timeout=timeout)
    if r.status_code != 200: raise RuntimeError(f"Directions 오류 {r.status_code}: {r.text[:160]}")
    j = r.json(); routes = j.get("routes", [])
    if not routes: raise RuntimeError("경로가 반환되지 않았습니다.")
    rt = routes[0]
    return rt["geometry"]["coordinates"], float(rt.get("duration",0.0)), float(rt.get("distance",0.0))

def mapbox_matrix(sources_xy, destinations_xy, profile="driving", token="", timeout=12):
    if not token: raise RuntimeError("MAPBOX_TOKEN 필요")
    coords = sources_xy + destinations_xy
    if len(coords) > MATRIX_MAX_COORDS:
        raise RuntimeError(f"Matrix 좌표 총합 {len(coords)}개 — {MATRIX_MAX_COORDS}개 이하로 줄여주세요.")
    coord_str = ";".join([f"{x},{y}" for x,y in coords])
    src_idx = ";".join(map(str, range(len(sources_xy))))
    dst_idx = ";".join(map(str, range(len(sources_xy), len(coords))))
    url = f"https://api.mapbox.com/directions-matrix/v1/mapbox/{profile}/{coord_str}"
    params = {"access_token": token, "annotations": "duration,distance", "sources": src_idx, "destinations": dst_idx}
    r = requests.get(url, params=params, timeout=timeout)
    if r.status_code != 200: raise RuntimeError(f"Matrix 오류 {r.status_code}: {r.text[:160]}")
    j = r.json()
    return j.get("durations"), j.get("distances")

@st.cache_data(show_spinner=False)
def mapbox_reverse_name_pretty(lon, lat, token="", lang="ko"):
    """역지오코딩 → 보기 좋은 한글 라벨 + 카테고리 태그"""
    if not token:
        return None, "NO_TOKEN"
    url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{lon},{lat}.json"
    params = {
        "access_token": token, "language": lang,
        "types": "poi,address,street,place,neighborhood",
        "limit": 8, "country": "kr", "proximity": f"{lon},{lat}"
    }
    try:
        r = requests.get(url, params=params, timeout=10)
    except Exception as e:
        return None, f"REQUEST_FAIL: {e}"
    if r.status_code != 200:
        return None, f"{r.status_code}: {r.text[:160]}"
    feats = (r.json() or {}).get("features", [])
    if not feats:
        return None, "NO_FEATURE"
    order = {"poi":0, "address":1, "street":2, "place":3, "neighborhood":4}
    def rank(ft):
        t = (ft.get("place_type") or [""])[0]
        base = order.get(t, 9)
        if (ft.get("properties") or {}).get("category"): base -= 0.2
        if ft.get("properties", {}).get("maki"): base -= 0.1
        return base
    feats.sort(key=rank)
    ft = feats[0]
    label = ft.get("text") or ft.get("place_name") or ""
    addr  = ft.get("address")
    if addr and addr not in label:
        label = f"{label} {addr}"
    cat = (ft.get("properties") or {}).get("category", "")
    cat_l = cat.lower()
    TAG_MAP = {
        "supermarket":"마트","department store":"백화점","convenience":"편의점",
        "school":"학교","university":"대학",
        "hospital":"병원","clinic":"의원","pharmacy":"약국",
        "bus":"버스","subway":"지하철","train":"기차역","railway":"철도",
        "city hall":"시청","government":"관공서",
        "park":"공원","library":"도서관","museum":"박물관",
        "market":"시장","restaurant":"식당","cafe":"카페"
    }
    tag = next((v for k,v in TAG_MAP.items() if k in cat_l), "")
    if tag:
        label = f"{label} ({tag})"
    label = label.strip()
    return (label if label else None), None

# ===================== UI =====================
col1, col2, col3 = st.columns([1.7,1.1,3.2], gap="large")

with col1:
    st.markdown('<div class="section-header">🚏 DRT 노선 추천 설정</div>', unsafe_allow_html=True)
    mode = st.radio("운행 모드", ["차량(운행)","도보(승객 접근)"], horizontal=True)
    profile = "driving" if mode.startswith("차량") else "walking"

    starts = st.multiselect("출발(승차) 정류장", stops["name"].tolist(), key="starts")
    ends   = st.multiselect("도착(하차) 정류장", stops["name"].tolist(), key="ends")

    pairing = st.selectbox("매칭 방식", ["인덱스 쌍(1:1)","모든 조합"], index=1)

    top_k = st.slider(
        "과금보호: 최대 경로 수(N)", 1, 50, 5,
        help="모든 조합을 Matrix 1회로 평가 후, 소요시간이 짧은 상위 N개만 Directions로 실제 경로 요청."
    )

    cA, cB, cC = st.columns(3)
    run_clicked   = cA.button("노선 추천")
    clear_clicked = cB.button("초기화")
    if cC.button("캐시 초기화"):
        st.cache_data.clear()
        st.rerun()

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
    if boundary is not None and not boundary.empty:
        folium.GeoJson(boundary, style_function=lambda f: {"color":"#9aa0a6","weight":2,"dashArray":"4,4","fillOpacity":0.05}).add_to(m)
    mc = MarkerCluster().add_to(m)
    for _, r in stops.iterrows():
        folium.Marker([r["lat"], r["lon"]], tooltip=str(r["name"]), icon=folium.Icon(color="gray")).add_to(mc)

    def xy(nm: str):
        row = stops.loc[stops["name"]==nm]
        if row.empty: return None
        r = row.iloc[0]; return float(r["lon"]), float(r["lat"])

    if run_clicked:
        if not starts or not ends:
            st.warning("출발/도착 정류장을 각각 1개 이상 선택하세요.")
        elif not MAPBOX_TOKEN:
            st.error("MAPBOX_TOKEN을 코드 상단에 입력하거나 환경변수/Secrets에 설정하세요.")
        else:
            src_xy = [xy(nm) for nm in starts if xy(nm)]
            dst_xy = [xy(nm) for nm in ends if xy(nm)]
            if not src_xy or not dst_xy:
                st.warning("유효한 좌표가 없습니다.")
            else:
                total_min, total_km = 0.0, 0.0
                pairs_to_draw = []

                if pairing.startswith("인덱스"):
                    n = min(len(src_xy), len(dst_xy), top_k)
                    pairs_to_draw = [(i, i) for i in range(n)]
                else:
                    pair_count = len(src_xy) * len(dst_xy)
                    if pair_count == 1:
                        pairs_to_draw = [(0,0)]
                    else:
                        try:
                            durations, distances = mapbox_matrix(src_xy, dst_xy, profile=profile, token=MAPBOX_TOKEN)
                            scored=[]
                            for i in range(len(src_xy)):
                                for j in range(len(dst_xy)):
                                    dur = None if not durations or not durations[i] else durations[i][j]
                                    if dur is None: continue
                                    dist = (distances[i][j] if distances and distances[i] else float("inf"))
                                    scored.append((dur, dist, i, j))
                            scored.sort(key=lambda x: (x[0], x[1]))
                            pairs_to_draw = [(i,j) for _,_,i,j in scored[:top_k]]
                        except Exception as e:
                            st.warning(f"Matrix 오류로 근사 정렬 사용: {e}")
                            # 허버사인 근사
                            def hav(xy1, xy2):
                                R=6371000.0
                                lon1,lat1,lon2,lat2 = map(np.radians,[xy1[0],xy1[1],xy2[0],xy2[1]])
                                dlon=lon2-lon1; dlat=lat2-lat1
                                a=np.sin(dlat/2)**2+np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
                                return 2*R*np.arcsin(np.sqrt(a))
                            scored=[]
                            for i,s in enumerate(src_xy):
                                for j,d in enumerate(dst_xy):
                                    scored.append((hav(s,d), i, j))
                            scored.sort(key=lambda x: x[0])
                            pairs_to_draw = [(i,j) for _,i,j in scored[:top_k]]

                # Directions: 상위 N개만 호출 + 역지오코딩 라벨
                for idx, (si, dj) in enumerate(pairs_to_draw):
                    sxy, exy = src_xy[si], dst_xy[dj]
                    s_label, s_err = mapbox_reverse_name_pretty(sxy[0], sxy[1], token=MAPBOX_TOKEN)
                    e_label, e_err = mapbox_reverse_name_pretty(exy[0], exy[1], token=MAPBOX_TOKEN)
                    S = s_label or starts[si]
                    E = e_label or ends[dj]
                    if s_err or e_err:
                        st.caption(f"역지오코딩 실패(승차:{s_err}, 하차:{e_err}) → 원래 이름 사용")

                    try:
                        coords, dur, dist = mapbox_route(sxy[0], sxy[1], exy[0], exy[1], profile=profile, token=MAPBOX_TOKEN)
                        ll = [(c[1], c[0]) for c in coords]
                        folium.PolyLine(ll, color=PALETTE[idx % len(PALETTE)], weight=5, opacity=0.9).add_to(m)
                        mid = ll[len(ll)//2]
                        folium.map.Marker(mid, icon=DivIcon(html=f"<div style='background:{PALETTE[idx%len(PALETTE)]};color:#fff;border-radius:50%;width:26px;height:26px;line-height:26px;text-align:center;font-weight:700;'>{idx+1}</div>")).add_to(m)
                        folium.Marker([sxy[1], sxy[0]], icon=folium.Icon(color="red"),  tooltip=f"승차: {S}").add_to(m)
                        folium.Marker([exy[1], exy[0]], icon=folium.Icon(color="blue"), tooltip=f"하차: {E}").add_to(m)
                        total_min += dur/60; total_km += dist/1000
                    except Exception as e:
                        st.warning(f"{S}→{E} Directions 실패: {e}")

                # 방문 순서/메트릭
                st.session_state["order"]    = []
                for (si,dj) in pairs_to_draw:
                    sxy, exy = src_xy[si], dst_xy[dj]
                    s_label, _ = mapbox_reverse_name_pretty(sxy[0], sxy[1], token=MAPBOX_TOKEN)
                    e_label, _ = mapbox_reverse_name_pretty(exy[0], exy[1], token=MAPBOX_TOKEN)
                    st.session_state["order"].append(f"{s_label or starts[si]} → {e_label or ends[dj]}")
                st.session_state["duration"] = total_min
                st.session_state["distance"] = total_km

                # 보기 범위
                try:
                    all_pts=[]
                    for (si,dj) in pairs_to_draw:
                        sxy, exy = src_xy[si], dst_xy[dj]
                        all_pts += [(sxy[1], sxy[0]), (exy[1], exy[0])]
                    if all_pts:
                        m.fit_bounds([[min(y for y,x in all_pts), min(x for y,x in all_pts)],
                                      [max(y for y,x in all_pts), max(x for y,x in all_pts)]])
                except: pass

    st_folium(m, height=560, returned_objects=[], use_container_width=True, key="main_map")
