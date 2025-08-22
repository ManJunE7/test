import streamlit as st
import geopandas as gpd
import pandas as pd
import folium
from folium.features import DivIcon
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from shapely.geometry import Point, LineString, MultiLineString
from shapely.ops import unary_union
from pathlib import Path
import requests
import math

# ─────────────────────────────────────────────
# 0) 기본 설정 & 토큰 (비워둠: st.secrets에서 읽음)
# ─────────────────────────────────────────────
st.set_page_config(page_title="DRT 최적경로 추천 (Mapbox)", layout="wide")
MAPBOX_TOKEN = st.secrets.get("MAPBOX_TOKEN", "pk.eyJ1IjoiZ3VyMDUxMDgiLCJhIjoiY21lZ2k1Y291MTdoZjJrb2k3bHc3cTJrbSJ9.DElgSQ0rPoRk1eEacPI8uQ")  # ← 네가 secrets.toml에 채워넣기

# ─────────────────────────────────────────────
# 1) 유틸: 벡터 레이어 자동 로더
#    - drt_1~drt_4: any of shp/geojson/gpkg
#    - poi 레이어: 기본 cb_tour.shp (Point)
# ─────────────────────────────────────────────
def _read_vector_any(basename: str):
    """basename(확장자 없이)로 shp/geojson/gpkg 탐색 후 첫 성공 레이어 리턴"""
    base = Path(".")
    patterns = [f"{basename}.shp", f"{basename}.geojson", f"{basename}.gpkg", f"{basename}.json"]
    for pat in patterns:
        for p in base.glob(f"**/{pat}"):
            try:
                gdf = gpd.read_file(p)
                return gdf
            except Exception:
                continue
    return None

@st.cache_data
def load_poi_layer():
    gdf = None
    # 프로젝트에 맞게 파일명 바꿔도 됨
    for name in ["cb_tour", "poi", "stops", "points"]:
        gdf = _read_vector_any(name)
        if gdf is not None:
            break
    if gdf is None:
        st.error("POI(Point) 레이어를 찾지 못했습니다. 예: cb_tour.shp")
        st.stop()
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    else:
        gdf = gdf.to_crs(epsg=4326)
    # lon/lat 컬럼 보장
    if "lon" not in gdf.columns or "lat" not in gdf.columns:
        gdf["lon"] = gdf.geometry.x
        gdf["lat"] = gdf.geometry.y
    # 이름 컬럼 추정
    name_col = None
    for c in ["name", "Name", "NAME", "title", "station", "st_name"]:
        if c in gdf.columns:
            name_col = c
            break
    if name_col is None:
        name_col = gdf.columns[0]  # 첫 컬럼을 이름처럼 사용 (원하면 바꿔도 됨)
    return gdf[[name_col, "lon", "lat", "geometry"]].rename(columns={name_col: "name"})

@st.cache_data
def load_drt_lines():
    drt = {}
    for i in range(1, 5):
        g = _read_vector_any(f"drt_{i}")
        if g is None:
            continue
        if g.crs is None:
            g = g.set_crs(epsg=4326)
        else:
            g = g.to_crs(epsg=4326)
        # 전체 라인 합치기
        geom = unary_union(g.geometry)
        if isinstance(geom, (LineString, MultiLineString)):
            drt[f"drt_{i}"] = geom
    return drt  # dict: {"drt_1": Line/MultiLine, ...}

# ─────────────────────────────────────────────
# 2) Mapbox Directions (실도로 라우팅)
# ─────────────────────────────────────────────
def mapbox_directions(lon1, lat1, lon2, lat2, profile="driving", token="", timeout=12):
    if not token:
        raise RuntimeError("MAPBOX_TOKEN이 설정되지 않았습니다. secrets.toml에 추가하세요.")
    url = f"https://api.mapbox.com/directions/v5/mapbox/{profile}/{lon1},{lat1};{lon2},{lat2}"
    params = {
        "geometries": "geojson",
        "overview": "full",
        "access_token": token
    }
    r = requests.get(url, params=params, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"Mapbox API 오류: {r.status_code} - {r.text[:180]}")
    j = r.json()
    routes = j.get("routes", [])
    if not routes:
        raise RuntimeError("Mapbox: 경로가 반환되지 않았습니다.")
    route = routes[0]
    coords = route["geometry"]["coordinates"]   # [[lon,lat], ...]
    duration = float(route.get("duration", 0.0))  # sec
    distance = float(route.get("distance", 0.0))  # m
    return coords, duration, distance

# ─────────────────────────────────────────────
# 3) 라우팅 결과를 DRT 라인과 매칭 (가까움 점수)
#    - 경로 좌표(경위도)를 일정 간격 샘플링 → 각 점과 라인의 거리 평균
#    - 거리는 투영좌표(UTM)로 변환 후 meter 단위 산출
# ─────────────────────────────────────────────
def route_drt_closeness_score(route_lonlat, drt_geom):
    """route_lonlat: [[lon,lat], ...], drt_geom: LineString/MultiLineString
       return: 평균거리(m) (낮을수록 '해당 DRT 라인과 가깝게 다닌다')"""
    if not route_lonlat:
        return float("inf")
    # GeoDataFrame으로 만들고 로컬 투영 좌표계로 변환 (UTM 자동 추정)
    r_gdf = gpd.GeoDataFrame(
        geometry=[Point(lon, lat) for lon, lat in route_lonlat],
        crs="EPSG:4326"
    )
    try:
        local_crs = r_gdf.estimate_utm_crs()
    except Exception:
        local_crs = "EPSG:32652"  # Korea 대략(Zone 52N), 필요시 조정
    r_xy = r_gdf.to_crs(local_crs)
    d_xy = gpd.GeoSeries([drt_geom], crs="EPSG:4326").to_crs(local_crs).iloc[0]
    # 모든 샘플 점에 대해 선까지의 거리(m) 평균
    dists = [pt.distance(d_xy) for pt in r_xy.geometry]
    return float(sum(dists) / len(dists))

def recommend_drt(route_lonlat, drt_dict):
    if not drt_dict:
        return None, {}
    scores = {}
    for k, geom in drt_dict.items():
        try:
            scores[k] = route_drt_closeness_score(route_lonlat, geom)
        except Exception:
            scores[k] = float("inf")
    # 최소 점수 = 가장 가까운 라인
    best = min(scores, key=scores.get)
    return best, scores

# ─────────────────────────────────────────────
# 4) 데이터 로드
# ─────────────────────────────────────────────
poi = load_poi_layer()
drt_lines = load_drt_lines()  # {"drt_1": geom, ...}

# 중심점 추정
try:
    c_lat = poi["lat"].astype(float).mean()
    c_lon = poi["lon"].astype(float).mean()
    if math.isnan(c_lat) or math.isnan(c_lon):
        raise ValueError
except Exception:
    c_lat, c_lon = 36.80, 127.15  # 충청권 대략값

# ─────────────────────────────────────────────
# 5) UI
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("옵션")
    profile = st.radio("이동 모드 (Mapbox)", ["driving", "walking"], horizontal=True)
    pairing = st.radio("매칭 방식", ["인덱스 쌍 매칭 (1:1)", "모든 조합"], index=0)
    max_routes = st.slider("최대 경로 수 제한", 1, 100, 20)
    run = st.button("경로 생성")

st.markdown("### 출발지 / 도착지 선택")
cols = st.columns(2)
with cols[0]:
    start_names = st.multiselect("출발지(여러 개 선택 가능)", poi["name"].tolist())
with cols[1]:
    end_names   = st.multiselect("도착지(여러 개 선택 가능)", poi["name"].tolist())

# ─────────────────────────────────────────────
# 6) 지도
# ─────────────────────────────────────────────
m = folium.Map(location=[c_lat, c_lon], zoom_start=12, tiles="CartoDB Positron", control_scale=True)

# POI 표시
mc = MarkerCluster().add_to(m)
for _, r in poi.iterrows():
    folium.Marker([r["lat"], r["lon"]], icon=folium.Icon(color="gray"), tooltip=str(r["name"])).add_to(mc)

# DRT 라인 표시
palette_drt = {"drt_1":"#9c27b0", "drt_2":"#00bcd4", "drt_3":"#8bc34a", "drt_4":"#ff9800"}
for k, geom in drt_lines.items():
    if isinstance(geom, LineString):
        segs = [geom.coords[:]]
    elif isinstance(geom, MultiLineString):
        segs = [list(ls.coords) for ls in geom.geoms]
    else:
        segs = []
    for seg in segs:
        folium.PolyLine([(lat, lon) for lon, lat in seg], color=palette_drt.get(k, "#555"), weight=3, opacity=0.6).add_to(m)

# ─────────────────────────────────────────────
# 7) 경로 생성
# ─────────────────────────────────────────────
results = []
route_palette = ["#4285f4", "#34a853", "#ea4335", "#fbbc04", "#7e57c2", "#26a69a", "#ef6c00", "#c2185b"]

def name_to_xy(name):
    row = poi.loc[poi["name"] == name]
    if row.empty:
        return None
    r = row.iloc[0]
    return float(r["lon"]), float(r["lat"])

if run:
    if not MAPBOX_TOKEN:
        st.error("MAPBOX_TOKEN이 없습니다. secrets.toml에 추가하세요.")
    elif not start_names or not end_names:
        st.warning("출발지와 도착지를 각각 1개 이상 선택하세요.")
    else:
        # 매칭 목록 구성
        od_pairs = []
        if pairing.startswith("인덱스"):
            n = min(len(start_names), len(end_names))
            for i in range(n):
                od_pairs.append((start_names[i], end_names[i]))
        else:  # 모든 조합
            for s in start_names:
                for e in end_names:
                    od_pairs.append((s, e))
        if len(od_pairs) > max_routes:
            st.info(f"경로가 {len(od_pairs)}건입니다. 상한 {max_routes}개만 처리합니다.")
            od_pairs = od_pairs[:max_routes]

        # 각 경로 계산 & 표시
        all_bounds = []
        for idx, (s_name, e_name) in enumerate(od_pairs):
            s_xy = name_to_xy(s_name)
            e_xy = name_to_xy(e_name)
            if s_xy is None or e_xy is None:
                st.warning(f"좌표를 찾을 수 없음: {s_name} → {e_name}")
                continue
            try:
                coords, dur, dist = mapbox_directions(s_xy[0], s_xy[1], e_xy[0], e_xy[1],
                                                      profile=profile, token=MAPBOX_TOKEN)
            except Exception as e:
                st.warning(f"Mapbox 실패: {s_name} → {e_name} / {e}")
                continue

            # 추천 DRT 계산
            best_drt, drt_scores = recommend_drt(coords, drt_lines)

            # 선 그리기
            color = route_palette[idx % len(route_palette)]
            latlon = [(c[1], c[0]) for c in coords]
            folium.PolyLine(latlon, color=color, weight=5, opacity=0.85).add_to(m)
            # 중간 라벨
            mid = latlon[len(latlon)//2]
            label = f"{idx+1}"
            if best_drt:
                label += f" · {best_drt}"
            folium.map.Marker(mid, icon=DivIcon(html=f"<div style='background:{color};color:#fff;"
                                                     "border-radius:50%;width:26px;height:26px;line-height:26px;"
                                                     "text-align:center;font-weight:700;'>{label}</div>")).add_to(m)
            # 시작/끝 마커
            folium.Marker([s_xy[1], s_xy[0]], icon=folium.Icon(color="red"), tooltip=f"Start: {s_name}").add_to(m)
            folium.Marker([e_xy[1], e_xy[0]], icon=folium.Icon(color="blue"), tooltip=f"End: {e_name}").add_to(m)

            # bounds 수집
            all_bounds += latlon

            # 결과 테이블용
            row = {
                "idx": idx+1,
                "start": s_name,
                "end": e_name,
                "profile": profile,
                "duration_min": round(dur/60, 1),
                "distance_km": round(dist/1000, 2),
                "recommend_drt": best_drt if best_drt else "-"
            }
            # 상위 4개 점수만 덧붙임
            for k in ["drt_1","drt_2","drt_3","drt_4"]:
                if k in drt_lines and k in (drt_scores or {}):
                    row[f"{k}_near_m"] = round(drt_scores[k], 1)
            results.append(row)

        # fit bounds
        if all_bounds:
            m.fit_bounds([
                [min(p[0] for p in all_bounds), min(p[1] for p in all_bounds)],
                [max(p[0] for p in all_bounds), max(p[1] for p in all_bounds)],
            ])

# 지도 출력
st_folium(m, width="100%", height=600, returned_objects=[], use_container_width=True)

# 결과 테이블
if results:
    st.markdown("### 경로 요약")
    df = pd.DataFrame(results)
    st.dataframe(df, use_container_width=True)
