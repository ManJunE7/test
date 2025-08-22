# app.py
# ---------------------------------------------------------
# 천안 DRT - 맞춤형 AI기반 스마트 교통 가이드 + 정류장명 자동 제안
# - 데이터: new_new_drt_min_utf8.(shp/gpkg/geojson)  (UTF-8)
# - 기본 이름(name)=지번(jibun)
# - Mapbox Geocoding(POI/교차로) + OSM 교차로 추론으로 정류장명 제안
# - Mapbox Directions/Matrix로 실도로 라우팅 (과금보호)
# ---------------------------------------------------------

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

# ===================== 토큰/상수 =====================
MAPBOX_TOKEN = "pk.eyJ1IjoiZ3VyMDUxMDgiLCJhIjoiY21lbWppYjByMDV2ajJqcjQyYXUxdzY3byJ9.yLBRJK_Ib6W3p9f16YlIKQ"  # << 직접 입력하거나 환경변수/Secrets(MAPBOX_TOKEN)로 설정
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

DATA_STEM = "new_new_drt_full_utf8"  # 파일명 앞부분 고정

# ===================== 유틸 =====================
def haversine(xy1, xy2):
    R=6371000.0
    lon1,lat1,lon2,lat2 = map(np.radians,[xy1[0],xy1[1],xy2[0],xy2[1]])
    dlon=lon2-lon1; dlat=lat2-lat1
    a=np.sin(dlat/2)**2+np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

def _read_utf8_shp(path: Path) -> gpd.GeoDataFrame:
    # shp UTF-8 강제
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
    g = _open_any()
    if "jibun" not in g.columns:
        st.error("소스에 'jibun' 필드가 없습니다."); st.stop()
    g = g.copy()
    g["jibun"] = g["jibun"].astype(str).str.strip()
    g["name"]  = g["jibun"]  # 기본 이름=지번
    g["lon"]   = g.geometry.x; g["lat"]=g.geometry.y
    st.caption(f"데이터셋: {DATA_STEM} (포인트 {len(g)}개 · UTF-8 · 기본이름=지번)")
    return g[["jibun","name","lon","lat","geometry"]]

stops = load_stops()

@st.cache_data
def load_label_source() -> gpd.GeoDataFrame:
    g = _open_any()
    g = g.copy()
    g["name"] = g["jibun"].astype(str).str.strip() if "jibun" in g.columns else g.get("name","정류장")
    return g

label_gdf = load_label_source()

# ===================== Mapbox - Directions/Matrix =====================
def mapbox_route(lon1,lat1,lon2,lat2, profile="driving", token="", timeout=12):
    if not token: raise RuntimeError("MAPBOX_TOKEN 필요")
    url=f"https://api.mapbox.com/directions/v5/mapbox/{profile}/{lon1},{lat1};{lon2},{lat2}"
    params={"geometries":"geojson","overview":"full","access_token":token}
    r=requests.get(url,params=params,timeout=timeout)
    if r.status_code!=200: raise RuntimeError(f"Directions 오류 {r.status_code}: {r.text[:160]}")
    j=r.json(); routes=j.get("routes",[])
    if not routes: raise RuntimeError("경로가 반환되지 않았습니다.")
    rt=routes[0]; return rt["geometry"]["coordinates"], float(rt.get("duration",0.0)), float(rt.get("distance",0.0))

def mapbox_matrix(src_xy, dst_xy, profile="driving", token="", timeout=12):
    if not token: raise RuntimeError("MAPBOX_TOKEN 필요")
    coords = src_xy+dst_xy
    if len(coords)>MATRIX_MAX_COORDS: raise RuntimeError(f"Matrix 좌표 총합 {len(coords)}개 — {MATRIX_MAX_COORDS} 이하 필요")
    coord_str=";".join([f"{x},{y}" for x,y in coords])
    src_idx=";".join(map(str,range(len(src_xy))))
    dst_idx=";".join(map(str,range(len(src_xy),len(coords))))
    url=f"https://api.mapbox.com/directions-matrix/v1/mapbox/{profile}/{coord_str}"
    params={"access_token":token,"annotations":"duration,distance","sources":src_idx,"destinations":dst_idx}
    r=requests.get(url,params=params,timeout=timeout)
    if r.status_code!=200: raise RuntimeError(f"Matrix 오류 {r.status_code}: {r.text[:160]}")
    j=r.json(); return j.get("durations"), j.get("distances")

# ===================== 정류장명 자동 제안 =====================
def _mbx_geocode(lon, lat, types="poi,intersection,address", limit=10, language="ko") -> list[dict]:
    if not MAPBOX_TOKEN: return []
    url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{lon},{lat}.json"
    params = {
        "access_token": MAPBOX_TOKEN,
        "types": types,
        "limit": limit,
        "language": language,
        "proximity": f"{lon},{lat}"
    }
    try:
        r = requests.get(url, params=params, timeout=8)
        if r.status_code != 200:
            return []
        return r.json().get("features", []) or []
    except Exception:
        return []

# POI 카테고리 가중치 (필요시 추가)
POI_WEIGHTS = {
    "town hall": 95, "city hall": 95, "광장":90, "government":85,
    "university":85, "college":85, "school":80,
    "subway station":85, "train station":85, "bus station":80, "bus stop":80,
    "department store":85, "shopping mall":82, "supermarket":80, "emart":90, "homeplus":88, "lotte mart":88,
    "hospital":82, "clinic":80, "pharmacy":70,
    "park":70, "stadium":70, "library":70,
}

def _clean_text_ko(s: str) -> str:
    if not s: return ""
    s = str(s)
    # 불필요 상위 지명 제거
    for bad in ["대한민국","대한민국 ", "South Korea", "Republic of Korea"]:
        s = s.replace(bad,"")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _pick_intersection_name(text_ko: str) -> Optional[str]:
    # 'OO로 · XX로' 같은 교차로 명에서 대표 도로 + '사거리/삼거리' 생성
    if not text_ko: return None
    parts = re.split(r"[·/,&\-|]", text_ko)
    parts = [p.strip() for p in parts if p.strip()]
    if not parts: return None
    # 도로명 후보에 '대로|로|길' 우선
    road = None
    for p in parts:
        if re.search(r"(대로|로|길)$", p):
            road = p; break
    road = road or parts[0]
    # 교차 도로 수에 따라 사거리/삼거리
    deg = len(set(parts))
    suffix = "사거리" if deg>=3 else "교차로"
    return f"{road} {suffix}"

def suggest_name_from_mapbox(lon: float, lat: float) -> Optional[str]:
    feats = _mbx_geocode(lon, lat, types="poi,intersection,address", limit=10, language="ko")
    if not feats: return None

    best = None
    best_score = -1e9

    for f in feats:
        ptypes = f.get("place_type", [])
        text   = _clean_text_ko(f.get("text_ko") or f.get("text"))
        pname  = _clean_text_ko(f.get("place_name_ko") or f.get("place_name"))
        center = f.get("center")
        dist_score = 0.0
        if isinstance(center, list) and len(center)==2:
            try:
                dist = haversine((lon,lat),(center[0],center[1]))
                # 가까울수록 가점
                dist_score = max(0, 300 - dist) / 300.0
            except Exception:
                pass

        score = 0.0

        if "poi" in ptypes:
            cat = (f.get("properties",{}) or {}).get("category","").lower()
            # 브랜드/카테고리 가중치
            cat_score = 0
            for k,v in POI_WEIGHTS.items():
                if k in cat or k in text.lower():
                    cat_score = max(cat_score, v)
            # '이마트','홈플러스','롯데마트' 같은 한글 브랜드 직접 가점
            for k,v in [("이마트",90),("홈플러스",88),("롯데마트",88),("시청",95),("초등학교",80),("중학교",80),("고등학교",80),("대학교",85),("병원",82),("도서관",70),("공원",70)]:
                if k in text: cat_score = max(cat_score, v)
            score = 50 + cat_score + 5*dist_score
            cand = text

        elif "intersection" in ptypes:
            cand = _pick_intersection_name(text) or text
            score = 75 + 5*dist_score

        elif "address" in ptypes:
            # 주소는 낮은 우선순위, 도로명만 간략화
            road = re.sub(r"\s*\d.*$", "", text)  # 숫자(번지) 제거
            cand = f"{road} 교차로 인근"
            score = 35 + 3*dist_score

        else:
            cand = text
            score = 20

        if cand and score > best_score:
            best = cand; best_score = score

    return best

def suggest_name_from_osm(lon: float, lat: float) -> Optional[str]:
    """OSM으로 교차로 이름 추론 (토큰 없을 때 fallback)"""
    try:
        import osmnx as ox
        G = ox.graph_from_point((lat, lon), dist=220, network_type="drive")
        nn = ox.distance.nearest_nodes(G, lon, lat)
        # 해당 노드로 들어오는 엣지들의 도로명
        names = []
        for u,v,k,data in G.edges(nbunch=nn, keys=True, data=True):
            nm = data.get("name")
            if isinstance(nm, list): names += nm
            elif nm: names.append(nm)
        names = [n for n in names if n]
        if not names: return None
        # 대표 도로 선택: '대로|로|길' 우선
        road = None
        for n in names:
            if re.search(r"(대로|로|길)$", str(n)): road = str(n); break
        road = road or str(names[0])
        deg = len(set(names))
        suffix = "사거리" if deg>=3 else ("삼거리" if deg==2 else "교차로")
        return f"{road} {suffix}"
    except Exception:
        return None

def suggest_stop_name(lon: float, lat: float) -> Optional[str]:
    # 1) Mapbox로 시도
    nm = suggest_name_from_mapbox(lon, lat)
    if nm: return nm
    # 2) OSM fallback
    return suggest_name_from_osm(lon, lat)

# ===================== UI: 좌측 폼 =====================
col1, col2, col3 = st.columns([1.9,1.2,3.2], gap="large")

with col1:
    st.markdown('<div class="section-header">🚏 DRT 노선 추천 설정</div>', unsafe_allow_html=True)
    mode    = st.radio("운행 모드", ["차량(운행)","도보(승객 접근)"], horizontal=True)
    profile = "driving" if mode.startswith("차량") else "walking"

    all_names = stops["name"].tolist()
    starts = st.multiselect("출발(승차) 정류장", all_names, key="starts")
    ends   = st.multiselect("도착(하차) 정류장", all_names, key="ends")

    pairing = st.selectbox("매칭 방식", ["인덱스 쌍(1:1)","모든 조합"], index=1)
    top_k   = st.slider("과금보호: 최대 경로 수(N)", 1, 50, 5,
                        help="모든 조합을 Matrix로 평가 후 상위 N개만 Directions 호출 (Matrix 좌표 총합 25 제한)")

    # ---- 정류장명 자동 제안 ----
    st.markdown('<div class="section-header">📝 정류장명 자동 제안</div>', unsafe_allow_html=True)
    use_auto_names = st.checkbox("정류장명 자동 제안 사용 (POI/교차로 기반)", value=False,
                                 help="주소 대신 주변 POI/교차로를 보고 '시청, 이마트, ○○사거리' 형태로 이름을 제안합니다.")
    gen_clicked = st.button("선택 정류장에 대해 이름 제안 생성")
    apply_clicked = st.button("제안된 이름 일괄 적용")

    cA, cB, cC = st.columns(3)
    run_clicked   = cA.button("노선 추천")
    clear_clicked = cB.button("초기화")
    if cC.button("캐시 초기화"):
        st.cache_data.clear(); st.rerun()
    if clear_clicked:
        for k in ["order","duration","distance","suggested"]:
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

    def xy(nm: str):
        row = stops.loc[stops["name"]==nm]
        if row.empty: return None
        rr = row.iloc[0]; return float(rr["lon"]), float(rr["lat"])

    # ----- 정류장명 자동 제안 실행 -----
    if gen_clicked:
        if not starts and not ends:
            st.warning("이름을 제안할 정류장을 선택하세요(출발/도착 중 아무거나).")
        else:
            sel = list(dict.fromkeys((starts or []) + (ends or [])))  # 중복 제거, 순서 유지
            suggested = []
            for nm in sel:
                p = stops.loc[stops["name"]==nm].iloc[0]
                sname = suggest_stop_name(float(p["lon"]), float(p["lat"]))
                suggested.append({"기존이름": nm, "제안이름": sname or "(제안 없음)"})
            st.session_state["suggested"] = suggested

    if "suggested" in st.session_state and st.session_state["suggested"]:
        st.dataframe(pd.DataFrame(st.session_state["suggested"]))
        st.markdown('<div class="table-note">※ 제안이름은 인근 POI/교차로 기반입니다. 필요시 나중에 수동으로 수정하세요.</div>', unsafe_allow_html=True)

    if apply_clicked:
        if "suggested" not in st.session_state or not st.session_state["suggested"]:
            st.warning("먼저 '이름 제안 생성'을 실행하세요.")
        else:
            # 제안 이름을 stops.name에 반영
            sug_map = {row["기존이름"]: row["제안이름"] for row in st.session_state["suggested"] if row["제안이름"] and row["제안이름"]!="(제안 없음)"}
            if sug_map:
                stops["name"] = stops["name"].apply(lambda x: sug_map.get(x, x))
                st.success("제안된 정류장명으로 적용했습니다. (이 세션에서만 반영)")
            else:
                st.info("적용할 제안 이름이 없습니다.")

    # ----- 경로 생성 -----
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

                pairing = st.session_state.get("pairing") or st.session_state.update({"pairing":"모든 조합"})
                pairing = st.session_state.get("pairing","모든 조합")

                # 화면의 실제 선택값 다시 읽기
                pairing = st.session_state.get("pairing_sel") if "pairing_sel" in st.session_state else st.selectbox

                # 간단 처리: 인덱스 쌍 / 모든 조합
                if st.session_state.get("pairing_last") == "인덱스 쌍(1:1)":
                    pass
                # 위처럼 상태를 더 정교하게 관리하고 싶다면 state 키로 처리하세요.

                # 여기선 UI의 pairing 변수를 그대로 이용
                if st.session_state.get("pairing_last", None) is None:
                    st.session_state["pairing_last"] = st.session_state.get("pairing_last","모든 조합")

                # 실제 조건 분기
                if st.session_state["pairing_last"] == "인덱스 쌍(1:1)" or pairing == "인덱스 쌍(1:1)":
                    n = min(len(src_xy), len(dst_xy), top_k)
                    pairs_to_draw = [(i, i) for i in range(n)]
                else:
                    # 모든 조합 → Matrix로 상위 N 추림
                    if len(src_xy) + len(dst_xy) <= MATRIX_MAX_COORDS:
                        try:
                            durations, distances = mapbox_matrix(src_xy, dst_xy, profile=("driving" if mode.startswith("차량") else "walking"), token=MAPBOX_TOKEN)
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

                # Directions 호출 & 그리기
                for idx, (si,dj) in enumerate(pairs_to_draw):
                    sxy, exy = src_xy[si], dst_xy[dj]
                    s_label = starts[si]
                    e_label = ends[dj]
                    try:
                        coords, dur, dist = mapbox_route(sxy[0], sxy[1], exy[0], exy[1], profile=("driving" if mode.startswith("차량") else "walking"), token=MAPBOX_TOKEN)
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

                st.session_state["order"]    = [f"{starts[si]} → {ends[dj]}" for (si,dj) in pairs_to_draw]
                st.session_state["duration"] = total_min
                st.session_state["distance"] = total_km

    st_folium(m, height=560, returned_objects=[], use_container_width=True, key="main_map")
