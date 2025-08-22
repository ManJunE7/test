# app.py
# ---------------------------------------------------------
# 천안 DRT - 맞춤형 AI기반 스마트 교통 가이드
# - 데이터 소스: new_new_drt_min_utf8.(shp/gpkg/geojson)
# - 정류장 이름(name) = 지번(jibun)
# - UTF-8 강제 로더(휴리스틱 포함)로 한글 깨짐 방지
# - Mapbox Matrix + Directions (과금보호)
# ---------------------------------------------------------

import os, math, re
from pathlib import Path
from typing import List, Tuple

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

# ===================== 기본 설정/스타일 =====================
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

# ===================== 토큰/상수 =====================
MAPBOX_TOKEN = "pk.eyJ1IjoiZ3VyMDUxMDgiLCJhIjoiY21lbWppYjByMDV2ajJqcjQyYXUxdzY3byJ9.yLBRJK_Ib6W3p9f16YlIKQ"  # << 직접 입력하거나 환경변수/Secrets로 설정
if not MAPBOX_TOKEN:
    MAPBOX_TOKEN = os.getenv("MAPBOX_TOKEN", "")
if not MAPBOX_TOKEN:
    try:
        MAPBOX_TOKEN = st.secrets["MAPBOX_TOKEN"]
    except Exception:
        pass

PALETTE = ["#4285f4","#34a853","#ea4335","#fbbc04","#7e57c2","#26a69a","#ef6c00","#c2185b"]
MATRIX_MAX_COORDS = 25
KOREA_CRS_METRIC   = "EPSG:5179"

# ===================== 데이터 로더 (UTF-8 강제 + 휴리스틱) =====================
DATA_STEM = "new_new_drt_min_utf8"

def _looks_garbled_korean(texts: list[str]) -> bool:
    """한글이 있어야 할 샘플이 모지바케로 보이는지 간단히 판별"""
    if not texts: return False
    hangul = sum(len(re.findall(r"[\uac00-\ud7a3]", s)) for s in texts)
    # UTF-8이 라틴1/기타로 잘못 디코딩됐을 때 자주 보이는 글자들
    weird  = sum(len(re.findall(r"[ìÍÎÏÃãÂåäÄÅæÆ¸¼½¾¤¦©«»¿�]", s)) for s in texts)
    # 한글 거의 없고 이상한 글자 많으면 깨짐으로 간주
    return (hangul < 3 and weird >= 2)

def _read_utf8_shp(path: Path) -> gpd.GeoDataFrame:
    """shp를 UTF-8로 강제 읽기 (pyogrio → geopandas → fiona 순서)"""
    # 1) pyogrio 우선
    try:
        from pyogrio import read_dataframe as pio
        g = pio(path, encoding="utf-8")
        return gpd.GeoDataFrame(g, geometry="geometry", crs=getattr(g, "crs", None))
    except Exception:
        pass
    # 2) geopandas 기본
    try:
        return gpd.read_file(path, encoding="utf-8")
    except Exception:
        pass
    # 3) fiona 엔진
    try:
        os.environ["SHAPE_ENCODING"] = "UTF-8"
        return gpd.read_file(path, engine="fiona")
    except Exception as e:
        raise e

def _open_any() -> gpd.GeoDataFrame:
    """new_new_drt_min_utf8.* 중 존재하는 걸 하나 연다. shp는 UTF-8 강제."""
    for ext in (".shp", ".gpkg", ".geojson"):
        p = Path(f"./{DATA_STEM}{ext}")
        if p.exists():
            if ext == ".shp":
                g = _read_utf8_shp(p)
            else:
                g = gpd.read_file(p)  # GPKG/GeoJSON은 UTF-8 고정
            # 좌표계 보정
            try:
                if g.crs and g.crs.to_epsg() != 4326:
                    g = g.to_crs(epsg=4326)
            except Exception:
                pass
            # 포인트 보장
            if not g.geom_type.astype(str).str.contains("Point", case=False, na=False).any():
                g = g.copy()
                g["geometry"] = g.geometry.representative_point()

            # --- 인코딩 휴리스틱 검사: jibun 샘플 확인 ---
            sample = []
            cols = [c for c in ["jibun","name"] if c in g.columns]
            for c in cols:
                sample += g[c].dropna().astype(str).head(80).tolist()
            if _looks_garbled_korean(sample):
                # 혹시 환경이 cp949로 강제 디코딩했을 가능성 → 다시 한번 pyogrio utf-8 시도
                try:
                    from pyogrio import read_dataframe as pio
                    g2 = pio(p, encoding="utf-8")
                    g = gpd.GeoDataFrame(g2, geometry="geometry", crs=getattr(g2, "crs", None))
                except Exception:
                    pass
            return g

    st.error(f"'{DATA_STEM}.shp/.gpkg/.geojson' 중 하나를 같은 폴더에 두세요.")
    st.stop()

@st.cache_data
def load_stops() -> gpd.GeoDataFrame:
    g = _open_any()
    if "jibun" not in g.columns:
        st.error("소스에 'jibun' 필드가 없습니다. (정류장 지번 필드가 필요합니다)")
        st.stop()
    g = g.copy()
    g["jibun"] = g["jibun"].astype(str).str.strip()
    g["name"]  = g["jibun"]  # 이름=지번
    g["lon"]   = g.geometry.x
    g["lat"]   = g.geometry.y
    st.caption(f"데이터셋: {DATA_STEM} (포인트 {len(g)}개 · UTF-8 · 이름=지번)")
    return g[["jibun","name","lon","lat","geometry"]]

@st.cache_data
def load_label_source() -> gpd.GeoDataFrame:
    g = _open_any()
    if "jibun" not in g.columns:
        st.error("소스에 'jibun' 필드가 없습니다.")
        st.stop()
    g = g.copy()
    g["name"] = g["jibun"].astype(str).str.strip()
    return g

stops     = load_stops()
label_gdf = load_label_source()

# 경계(선택)
@st.cache_data
def load_boundary():
    for nm in ["boundary","admin_boundary","cb_shp","cheonan_boundary"]:
        for ext in ["shp","geojson","gpkg","json"]:
            p = next(Path(".").rglob(f"{nm}.{ext}"), None)
            if p:
                try:
                    g0 = gpd.read_file(p)
                    if g0.crs and g0.crs.to_epsg() != 4326:
                        g0 = g0.to_crs(epsg=4326)
                    return g0
                except Exception:
                    pass
    return None

boundary = load_boundary()

# 중심점
ctr_lat = float(stops["lat"].mean()); ctr_lon = float(stops["lon"].mean())
if math.isnan(ctr_lat) or math.isnan(ctr_lon): ctr_lat, ctr_lon = 36.80, 127.15

# ===================== 로컬 역지오코딩 =====================
@st.cache_data
def local_reverse_label(lon: float, lat: float) -> str | None:
    if label_gdf is None or label_gdf.empty:
        return None
    try:
        g_m = label_gdf.to_crs(KOREA_CRS_METRIC)
        p_m = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(KOREA_CRS_METRIC).iloc[0]
        idx_candidates = list(g_m.sindex.nearest(p_m.bounds, 1))
        idx0 = idx_candidates[0] if idx_candidates else None
        if idx0 is None:
            dists = g_m.geometry.distance(p_m)
            idx0 = int(dists.idxmin())
        nm = str(label_gdf.loc[idx0, "name"]).strip()
        return nm or None
    except Exception:
        dists = label_gdf.geometry.distance(Point(lon, lat))
        idx0 = int(dists.idxmin())
        nm = str(label_gdf.loc[idx0, "name"]).strip()
        return nm or None

# ===================== Mapbox 라우팅/매트릭스 =====================
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

def mapbox_matrix(sources_xy: List[Tuple[float,float]],
                  destinations_xy: List[Tuple[float,float]],
                  profile="driving", token="", timeout=12):
    if not token: raise RuntimeError("MAPBOX_TOKEN 필요")
    coords = sources_xy + destinations_xy
    if len(coords) > MATRIX_MAX_COORDS:
        raise RuntimeError(f"Matrix 좌표 총합 {len(coords)}개 — {MATRIX_MAX_COORDS}개 이하만 지원")
    coord_str = ";".join([f"{x},{y}" for x,y in coords])
    src_idx = ";".join(map(str, range(len(sources_xy))))
    dst_idx = ";".join(map(str, range(len(sources_xy), len(coords))))
    url = f"https://api.mapbox.com/directions-matrix/v1/mapbox/{profile}/{coord_str}"
    params = {"access_token": token, "annotations": "duration,distance",
              "sources": src_idx, "destinations": dst_idx}
    r = requests.get(url, params=params, timeout=timeout)
    if r.status_code != 200: raise RuntimeError(f"Matrix 오류 {r.status_code}: {r.text[:160]}")
    j = r.json()
    return j.get("durations"), j.get("distances")

def haversine(xy1, xy2):
    R=6371000.0
    lon1,lat1,lon2,lat2 = map(np.radians,[xy1[0],xy1[1],xy2[0],xy2[1]])
    dlon=lon2-lon1; dlat=lat2-lat1
    a=np.sin(dlat/2)**2+np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

# ===================== UI =====================
col1, col2, col3 = st.columns([1.7,1.1,3.2], gap="large")

with col1:
    st.markdown('<div class="section-header">🚏 DRT 노선 추천 설정</div>', unsafe_allow_html=True)
    mode    = st.radio("운행 모드", ["차량(운행)","도보(승객 접근)"], horizontal=True)
    profile = "driving" if mode.startswith("차량") else "walking"

    all_names = stops["name"].tolist()
    starts = st.multiselect("출발(승차) 정류장", all_names, key="starts")
    ends   = st.multiselect("도착(하차) 정류장", all_names, key="ends")

    pairing = st.selectbox("매칭 방식", ["인덱스 쌍(1:1)","모든 조합"], index=1)
    top_k   = st.slider("과금보호: 최대 경로 수(N)", 1, 50, 5,
                        help="모든 조합을 Matrix로 평가 후 상위 N개만 Directions 호출. (Matrix 좌표 총합 25 제한)")

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
                pairs_to_draw: List[Tuple[int,int]] = []

                if pairing.startswith("인덱스"):
                    n = min(len(src_xy), len(dst_xy), top_k)
                    pairs_to_draw = [(i, i) for i in range(n)]
                else:
                    pair_count = len(src_xy) * len(dst_xy)
                    if pair_count == 1:
                        pairs_to_draw = [(0,0)]
                    else:
                        if len(src_xy) + len(dst_xy) <= MATRIX_MAX_COORDS:
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
                        if not pairs_to_draw:
                            scored=[]
                            for i,s in enumerate(src_xy):
                                for j,d in enumerate(dst_xy):
                                    scored.append((haversine(s,d), i, j))
                            scored.sort(key=lambda x: x[0])
                            pairs_to_draw = [(i,j) for _,i,j in scored[:top_k]]

                # Directions 호출 + 라벨 표시
                for idx, (si, dj) in enumerate(pairs_to_draw):
                    sxy, exy = src_xy[si], dst_xy[dj]
                    s_label = local_reverse_label(sxy[0], sxy[1]) or starts[si]
                    e_label = local_reverse_label(exy[0], exy[1]) or ends[dj]

                    try:
                        coords, dur, dist = mapbox_route(sxy[0], sxy[1], exy[0], exy[1], profile=profile, token=MAPBOX_TOKEN)
                        ll = [(c[1], c[0]) for c in coords]
                        folium.PolyLine(ll, color=PALETTE[idx % len(PALETTE)], weight=5, opacity=0.9).add_to(m)
                        mid = ll[len(ll)//2]
                        folium.map.Marker(
                            mid,
                            icon=DivIcon(html=f"<div style='background:{PALETTE[idx%len(PALETTE)]};color:#fff;border-radius:50%;width:26px;height:26px;line-height:26px;text-align:center;font-weight:700;'>{idx+1}</div>")
                        ).add_to(m)
                        folium.Marker([sxy[1], sxy[0]], icon=folium.Icon(color="red"),  tooltip=f"승차: {s_label}").add_to(m)
                        folium.Marker([exy[1], exy[0]], icon=folium.Icon(color="blue"), tooltip=f"하차: {e_label}").add_to(m)
                        total_min += dur/60; total_km += dist/1000
                    except Exception as e:
                        st.warning(f"{s_label}→{e_label} Directions 실패: {e}")

                st.session_state["order"]    = []
                for (si,dj) in pairs_to_draw:
                    sxy, exy = src_xy[si], dst_xy[dj]
                    s_label = local_reverse_label(sxy[0], sxy[1]) or starts[si]
                    e_label = local_reverse_label(exy[0], exy[1]) or ends[dj]
                    st.session_state["order"].append(f"{s_label} → {e_label}")
                st.session_state["duration"] = total_min
                st.session_state["distance"] = total_km

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
