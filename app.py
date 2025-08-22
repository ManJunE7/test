# app.py — DRT 3/4, 페르소나 정류장만, 도로 경로
import os, math, requests
import streamlit as st
import geopandas as gpd
import folium
from shapely.geometry import LineString, MultiLineString
from streamlit_folium import st_folium

st.set_page_config(page_title="DRT 3·4 최적 동선(도로 경로)", layout="wide")

# ▣ Mapbox 토큰: secrets/env/하드코딩 순
MAPBOX_TOKEN = (st.secrets.get("MAPBOX_TOKEN", "") or os.getenv("MAPBOX_TOKEN", "")).strip()
if not MAPBOX_TOKEN:
    MAPBOX_TOKEN = "pk.eyJ1IjoiZ3VyMDUxMDgiLCJhIjoiY21lZ2k1Y291MTdoZjJrb2k3bHc3cTJrbSJ9.DElgSQ0rPoRk1eEacPI8uQ"

# ▣ 우리가 정한 페르소나 정류장만 사용 (좌표는 없음 → 라인상 보간으로 만듦)
STOPDATA = {
    "3번버스": [
        {"seq":1,  "name":"우성단 아파트 (RED)",                 "personas":["P3"]},
        {"seq":2,  "name":"북측 아파트 (RED)",                   "personas":["P2"]},
        {"seq":3,  "name":"동측 그린 빌라 (RED)",                "personas":["P1"]},
        {"seq":4,  "name":"산단 분홍존 → Gate 2 (GREEN)",       "personas":["P1"]},
        {"seq":5,  "name":"라인 아파트 (RED)",                   "personas":["P4"]},
        {"seq":6,  "name":"청수 쉼터 (RED)",                     "personas":["P5"]},
        {"seq":7,  "name":"중앙시장 주민 허브 (GREEN)",          "personas":["P5"]},
        {"seq":8,  "name":"산단 기업지원동 (GREEN)",             "personas":["P2"]},
        {"seq":9,  "name":"체육센터(수영장) (GREEN)",            "personas":["P3"]},
        {"seq":10, "name":"교회 앞 골목 (GREEN)",                "personas":["P4"]},
    ],
    "4번버스": [
        {"seq":1,  "name":"봉명동 빌딩 앞 (RED)",                "personas":["P2"]},
        {"seq":2,  "name":"상명대 인근 아파트 (RED)",            "personas":["P3"]},
        {"seq":3,  "name":"천안역 환승/출구 (GREEN)",            "personas":["P2"]},
        {"seq":4,  "name":"쌍용동 아파트 (RED)",                 "personas":["P4"]},
        {"seq":5,  "name":"병원 정문/외래 접수 (GREEN)",         "personas":["P3"]},
        {"seq":6,  "name":"성정남부 주택가 (RED)",               "personas":["P5"]},
        {"seq":7,  "name":"도솔/쌍용공원 입구 (GREEN)",          "personas":["P4"]},
        {"seq":8,  "name":"남부도서관 인근 주거지 (RED)",        "personas":["P5"]},
        {"seq":9,  "name":"이마트 정문 (GREEN)",                 "personas":["P5"]},
        {"seq":10, "name":"사찰/법당 앞 (GREEN)",                "personas":["P6"]},
    ],
}

ROUTE_FILES = {"3번버스": "drt_3.shp", "4번버스": "drt_4.shp"}

# ---------- Geo helpers ----------
def load_route_latlon(route_name):
    shp = ROUTE_FILES[route_name]
    gdf = gpd.read_file(shp).to_crs(4326)
    geom = gdf.geometry.iloc[0]
    if isinstance(geom, MultiLineString):
        geom = max(list(geom.geoms), key=lambda L: L.length)
    if not isinstance(geom, LineString):
        raise ValueError("라인 형식이 아님")
    lonlat = list(geom.coords)             # (lon,lat)
    return [(y, x) for (x, y) in lonlat]   # (lat,lon)

def hav(lat1, lon1, lat2, lon2):
    R=6371000.0
    from math import radians, sin, cos, asin, sqrt
    y1,x1,y2,x2 = map(radians,[lat1,lon1,lat2,lon2])
    dy,dx = y2-y1, x2-x1
    h = sin(dy/2)**2 + cos(y1)*cos(y2)*sin(dx/2)**2
    return 2*R*asin(sqrt(h))

def poly_len_m(coords):
    return sum(hav(*coords[i], *coords[i+1]) for i in range(len(coords)-1))

def cumulative(coords):
    acc=[0.0]
    for i in range(len(coords)-1):
        acc.append(acc[-1] + hav(*coords[i], *coords[i+1]))
    return acc

def point_at_length(coords, cum, target):
    if target<=0: return coords[0]
    if target>=cum[-1]: return coords[-1]
    for i in range(1,len(cum)):
        if cum[i] >= target:
            r=(target-cum[i-1])/max(cum[i]-cum[i-1],1e-9)
            lat = coords[i-1][0] + r*(coords[i][0]-coords[i-1][0])
            lon = coords[i-1][1] + r*(coords[i][1]-coords[i-1][1])
            return (lat,lon)
    return coords[-1]

def seq_to_pos(seq, N, total_m, coords, cum):
    t = (seq-1)/max(N-1,1)
    return point_at_length(coords, cum, t*total_m)

# ---------- Mapbox Directions (도로 경로) ----------
def directions_between(p1_latlon, p2_latlon):
    """(lat,lon),(lat,lon) -> (coords_latlon, dist_m, dur_s)"""
    if not MAPBOX_TOKEN: return None, None, None
    (lat1,lon1),(lat2,lon2) = p1_latlon, p2_latlon
    url = f"https://api.mapbox.com/directions/v5/mapbox/driving/{lon1},{lat1};{lon2},{lat2}"
    params = {"geometries":"geojson","overview":"full","access_token":MAPBOX_TOKEN}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data=r.json()
        if not data.get("routes"): return None, None, None
        route=data["routes"][0]
        coords = [(lat,lon) for (lon,lat) in route["geometry"]["coordinates"]]
        return coords, route.get("distance",0.0), route.get("duration",0.0)
    except Exception:
        return None, None, None

# ---------- UI ----------
left, mid, right = st.columns([1.1, 0.9, 3.0], gap="large")

with left:
    st.subheader("① 노선/정류장 선택")
    route = st.selectbox("노선", ["3번버스","4번버스"])
    persona = st.selectbox("페르소나 필터(선택)", ["ALL","P1","P2","P3","P4","P5","P6"])

    stops = STOPDATA[route]
    def seen(s): return True if persona=="ALL" else (persona in s["personas"])
    filtered = [s for s in stops if seen(s)] or stops
    labels = [f"{s['seq']:02d}. {s['name']}" for s in filtered]

    picks = st.multiselect("승차 정류장(여러 개)", labels, default=labels[:1])
    drops = st.multiselect("하차 정류장(여러 개)", labels, default=labels[-1:])

    direction = st.radio("방향", ["오름차순(순서↑)", "내림차순(순서↓)"], horizontal=True, index=0)
    go = st.button("경로 계산")

# 노선 라인 로드 + 정류장 좌표 보간
try:
    base_line = load_route_latlon(route)
except Exception as e:
    st.error(f"{route} 라인 로드 실패: {e}")
    st.stop()

total_m = poly_len_m(base_line)
cum = cumulative(base_line)
N = len(stops)
seq_xy = {s["seq"]: seq_to_pos(s["seq"], N, total_m, base_line, cum) for s in stops}

def seq_of(label): return int(str(label).split(".")[0])
sel_seqs = sorted(set([seq_of(x) for x in (picks+drops)]))

with mid:
    st.subheader("② 요약")
    if go and len(sel_seqs)>=2:
        order = sorted(sel_seqs) if direction.startswith("오름") else sorted(sel_seqs, reverse=True)
        tot_m = 0.0; tot_s = 0.0; legs = []
        for a,b in zip(order[:-1], order[1:]):
            c1, c2 = seq_xy[a], seq_xy[b]
            coords, d, t = directions_between(c1, c2)
            if coords is None:   # 실패 시 직선 대신 노선 라인 구간으로 대체
                # 노선 라인에서 비율 잘라서 표시
                a_m = (a-1)/max(N-1,1)*total_m; b_m=(b-1)/max(N-1,1)*total_m
                # 간단히 보간 두 점만 (fallback)
                coords = [point_at_length(base_line, cum, a_m), point_at_length(base_line, cum, b_m)]
                d = hav(*coords[0], *coords[1])
                t = d/ (25/3.6)  # 25km/h 가정
            legs.append(coords); tot_m += d; tot_s += t

        st.metric("📏 총 이동거리", f"{tot_m/1000:.2f} km")
        st.metric("⏱ 예상 소요시간", f"{tot_s/60:.1f} 분")
        st.caption(f"선택 정류장 수: {len(order)} · 구간 수: {len(order)-1}")
    else:
        st.info("승차/하차 정류장을 여러 개 선택한 뒤 **경로 계산**을 누르세요.")

with right:
    st.subheader("③ 경로 시각화")
    m = folium.Map(location=base_line[len(base_line)//2], zoom_start=13, tiles="CartoDB Positron")

    # 전체 노선(얇게)
    folium.PolyLine(base_line, color="#9aa0a6", weight=3, opacity=0.5, tooltip=f"{route} 라인").add_to(m)

    # 정류장(우리가 정한 것만) — 선택된 것은 색 강조
    pick_set = {seq_of(x) for x in picks}
    drop_set = {seq_of(x) for x in drops}
    focus_set = set(sel_seqs)

    for s in stops:
        lat,lon = seq_xy[s["seq"]]
        color = "#1e88e5"
        if s["seq"] in pick_set: color = "#43a047"  # 승차=초록
        if s["seq"] in drop_set: color = "#e53935"  # 하차=빨강
        radius = 6 if s["seq"] in focus_set else 4
        folium.CircleMarker([lat,lon], radius=radius, color=color, fill=True, fill_opacity=1.0,
                            tooltip=f"{s['seq']}. {s['name']}").add_to(m)

    # 도로 경로 그리기
    if go and len(sel_seqs)>=2:
        order = sorted(sel_seqs) if direction.startswith("오름") else sorted(sel_seqs, reverse=True)
        for a,b in zip(order[:-1], order[1:]):
            coords, _, _ = directions_between(seq_xy[a], seq_xy[b])
            if coords is None:
                # fallback: 두 점만
                coords = [seq_xy[a], seq_xy[b]]
            folium.PolyLine(coords, color="#00c853", weight=7, opacity=0.95,
                            tooltip=f"{a}→{b}").add_to(m)

    st_folium(m, height=620, use_container_width=True)
