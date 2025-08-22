import os, math, requests
import streamlit as st
import geopandas as gpd
import folium
from streamlit_folium import st_folium

# ── 기본 설정 ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="DRT 전체 정류장 · 실도로 최적 동선", layout="wide")

# Mapbox 토큰: (Secrets → env → 마지막 fallback)
MAPBOX_TOKEN = (st.secrets.get("MAPBOX_TOKEN", "") or os.getenv("MAPBOX_TOKEN", "")).strip()
if not MAPBOX_TOKEN:
    MAPBOX_TOKEN = "PUT_YOUR_MAPBOX_TOKEN_HERE"   # 배포땐 Secrets로 넣으세요!

# 정류장 shp 경로(포인트) – 파일명만 바꾸면 됨
STOP_SHP_CANDIDATES = ["drt1234.shp", "new_drt.shp"]

# ── 유틸 ───────────────────────────────────────────────────────────────────
def _ensure_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    try:
        if gdf.crs is None:
            # 좌표계 정보 없으면 WGS84로 가정(필요 시 수정)
            gdf.set_crs(epsg=4326, inplace=True)
        return gdf.to_crs(4326)
    except Exception:
        return gdf

@st.cache_data(show_spinner=False)
def load_all_stops():
    """정류장 포인트 SHP 로드: (lat, lon) + route/line 컬럼 추출"""
    shp = None
    for c in STOP_SHP_CANDIDATES:
        if os.path.exists(c):
            shp = c
            break
    if shp is None:
        raise FileNotFoundError("정류장 SHP를 찾지 못했습니다. (drt1234.shp / new_drt.shp 등)")

    gdf = _ensure_wgs84(gpd.read_file(shp))

    # 좌표 추출
    gdf["lat"] = gdf.geometry.y
    gdf["lon"] = gdf.geometry.x

    # 노선 컬럼 추정 (없으면 ALL)
    route_col_candidates = ["route", "line", "노선", "drt", "bus_line", "line_id"]
    route_col = next((c for c in route_col_candidates if c in gdf.columns), None)
    if route_col is None:
        gdf["route"] = "ALL"
        route_col = "route"

    # 정류장 ID 컬럼 추정 (없으면 인덱스로 대체)
    id_col_candidates = ["stop_id", "id", "정류장ID", "정류장id", "정류장", "bus_stop", "name"]
    id_col = next((c for c in id_col_candidates if c in gdf.columns), None)
    if id_col is None:
        gdf["stop_id"] = gdf.index.astype(str)
        id_col = "stop_id"

    # 라벨(보여주기용) – 이름은 빼달라고 하셔서, 라우트-순번 형태로
    # 만약 실제 순번 컬럼이 있다면 아래 candidates에 추가해서 우선 사용하세요.
    seq_col_candidates = ["seq", "순번", "order", "index"]
    seq_col = next((c for c in seq_col_candidates if c in gdf.columns), None)

    if seq_col:
        gdf["label"] = gdf[route_col].astype(str) + "-" + gdf[seq_col].astype(str).str.zfill(2)
    else:
        gdf["label"] = gdf[route_col].astype(str) + "-" + (gdf.groupby(route_col).cumcount() + 1).astype(str).str.zfill(2)

    return gdf, route_col, id_col

def mapbox_optimize(latlon_list, fix_first=True, fix_last=True):
    """Mapbox Optimization API: 실도로 최적 순서 + 전체 경로"""
    if len(latlon_list) < 2 or not MAPBOX_TOKEN or MAPBOX_TOKEN == "PUT_YOUR_MAPBOX_TOKEN_HERE":
        return None, [], 0.0, 0.0

    path = ";".join(f"{lon},{lat}" for (lat, lon) in latlon_list)
    params = {
        "geometries": "geojson",
        "overview": "full",
        "roundtrip": "false",
        "access_token": MAPBOX_TOKEN,
    }
    if fix_first:
        params["source"] = "first"
    if fix_last:
        params["destination"] = "last"

    url = f"https://api.mapbox.com/optimized-trips/v1/mapbox/driving/{path}"
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        j = r.json()
        trips = j.get("trips", [])
        if not trips:
            return None, [], 0.0, 0.0
        trip = trips[0]
        route_coords = [(lat, lon) for (lon, lat) in trip["geometry"]["coordinates"]]
        # 정렬된 방문 인덱스
        wps = j.get("waypoints", [])
        order = sorted(
            [(wp.get("waypoint_index", -1), i) for i, wp in enumerate(wps) if wp.get("waypoint_index", -1) >= 0],
            key=lambda x: x[0]
        )
        order_idx = [i for _, i in order]
        return route_coords, order_idx, trip.get("distance", 0.0), trip.get("duration", 0.0)
    except Exception:
        return None, [], 0.0, 0.0

def mapbox_directions(a_latlon, b_latlon):
    """단순 2점 경로(실도로) – 최적화 실패 시 fallback"""
    if not MAPBOX_TOKEN or MAPBOX_TOKEN == "PUT_YOUR_MAPBOX_TOKEN_HERE":
        return None, 0.0, 0.0
    (la1, lo1), (la2, lo2) = a_latlon, b_latlon
    url = f"https://api.mapbox.com/directions/v5/mapbox/driving/{lo1},{la1};{lo2},{la2}"
    params = {"geometries": "geojson", "overview": "full", "access_token": MAPBOX_TOKEN}
    try:
        r = requests.get(url, params=params, timeout=12)
        r.raise_for_status()
        j = r.json()
        if not j.get("routes"):
            return None, 0.0, 0.0
        rt = j["routes"][0]
        coords = [(lat, lon) for (lon, lat) in rt["geometry"]["coordinates"]]
        return coords, rt.get("distance", 0.0), rt.get("duration", 0.0)
    except Exception:
        return None, 0.0, 0.0

# ── 데이터 로드 ────────────────────────────────────────────────────────────
try:
    stops_gdf, ROUTE_COL, ID_COL = load_all_stops()
except Exception as e:
    st.error(f"정류장 데이터를 불러오지 못했습니다: {e}")
    st.stop()

# ── UI ─────────────────────────────────────────────────────────────────────
st.markdown("## 🚌 모든 정류장 기반 · 실도로 최적 동선")

left, right = st.columns([1.1, 2.9], gap="large")

with left:
    # 노선 필터
    routes = ["전체"] + sorted(stops_gdf[ROUTE_COL].astype(str).unique().tolist())
    sel_route = st.selectbox("노선 필터", routes)

    if sel_route == "전체":
        pool = stops_gdf.copy()
    else:
        pool = stops_gdf[stops_gdf[ROUTE_COL].astype(str) == sel_route].copy()

    # 선택 목록(라벨: route-순번 형태, 내부값: index)
    options = pool.index.tolist()
    option_labels = pool["label"].tolist()

    picks_idx = st.multiselect("승차 정류장 (여러 개 선택)", options, format_func=lambda i: option_labels[options.index(i)])
    drops_idx = st.multiselect("하차 정류장 (여러 개 선택)", options, format_func=lambda i: option_labels[options.index(i)])

    fix_first = st.checkbox("첫 정류장 고정(시작점)", True)
    fix_last  = st.checkbox("마지막 정류장 고정(종점)", True)

    run = st.button("최적 동선 계산", type="primary")

with right:
    # 초기 맵 중심
    if len(pool):
        center = [pool["lat"].mean(), pool["lon"].mean()]
    else:
        center = [36.815, 127.113]

    m = folium.Map(location=center, zoom_start=13, tiles="CartoDB Positron")

    # 모든 정류장 표시
    for _, r in pool.iterrows():
        folium.CircleMarker(
            [r.lat, r.lon],
            radius=4,
            color="#1e88e5",
            fill=True,
            fill_opacity=1,
            tooltip=r["label"],
        ).add_to(m)

    # 선택된 정류장 강조(색상 구분)
    for i in picks_idx:
        rr = pool.loc[i]
        folium.CircleMarker([rr.lat, rr.lon], radius=7, color="#43a047", fill=True, fill_opacity=1,
                            tooltip=f"승차: {rr['label']}").add_to(m)
    for i in drops_idx:
        rr = pool.loc[i]
        folium.CircleMarker([rr.lat, rr.lon], radius=7, color="#e53935", fill=True, fill_opacity=1,
                            tooltip=f"하차: {rr['label']}").add_to(m)

    # 실도로 최적 동선
    if run:
        sel_idx = list(dict.fromkeys(picks_idx + drops_idx))  # 중복 제거 + 순서 유지
        if len(sel_idx) < 2:
            st.warning("정류장을 최소 2개 이상 선택하세요.")
        elif MAPBOX_TOKEN == "PUT_YOUR_MAPBOX_TOKEN_HERE":
            st.error("MAPBOX_TOKEN이 설정되어 있지 않습니다. Secrets에 추가해 주세요.")
        else:
            latlon = [(pool.loc[i].lat, pool.loc[i].lon) for i in sel_idx]
            trip, order_idx, dist_m, dur_s = mapbox_optimize(latlon, fix_first, fix_last)

            if trip:
                folium.PolyLine(trip, color="#00c853", weight=7, opacity=0.95, tooltip="최적 동선").add_to(m)
                st.success(f"📏 {dist_m/1000:.2f} km  ·  ⏱ {dur_s/60:.1f} 분")
                st.markdown("**방문 순서(최적화 결과)**")
                for n, idx in enumerate(order_idx, 1):
                    lab = pool.loc[sel_idx[idx], "label"]
                    st.write(f"- {n}. {lab}")
            else:
                # 최적화 실패 시 선택 순서대로 실도로 연결
                total_d, total_t = 0.0, 0.0
                for a, b in zip(latlon[:-1], latlon[1:]):
                    line, d, t = mapbox_directions(a, b)
                    if line:
                        folium.PolyLine(line, color="#00c853", weight=7, opacity=0.95).add_to(m)
                        total_d += d; total_t += t
                st.info(f"(추정) 📏 {total_d/1000:.2f} km  ·  ⏱ {total_t/60:.1f} 분")

    st_folium(m, height=640, use_container_width=True)
