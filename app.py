# app.py — 엑셀 없이 동작 (엑셀에서 추출한 drt_excel_meta.json 사용)
import json, math, os
import streamlit as st
import geopandas as gpd
import folium
from shapely.geometry import LineString
from streamlit_folium import st_folium

st.set_page_config(page_title="DRT 페르소나 최적 동선", layout="wide")

# 1) 노선 파일 매핑
ROUTE_FILES = {
    "1번버스": "drt_1.shp",
    "2번버스": "drt_2.shp",
    "3번버스": "drt_3.shp",
    "4번버스": "drt_4.shp",
}

# 2) 엑셀에서 뽑아둔 메타(JSON) 불러오기: route -> seq -> {name,time,note,persona,persona_code}
META_PATH = "drt_excel_meta.json"
with open(META_PATH, "r", encoding="utf-8") as f:
    EXCEL_META = json.load(f)

# 3) 유틸
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0
    from math import radians, sin, cos, asin, sqrt
    y1, x1, y2, x2 = map(radians, [lat1, lon1, lat2, lon2])
    dy, dx = y2 - y1, x2 - x1
    h = sin(dy/2)**2 + cos(y1)*cos(y2)*sin(dx/2)**2
    return 2*R*asin(sqrt(h))

def poly_length_km(coords_latlon):
    return sum(haversine(*coords_latlon[i], *coords_latlon[i+1]) for i in range(len(coords_latlon)-1)) / 1000.0

def point_at_fraction(coords_latlon, t):
    """폴리라인의 누적거리 비율 t(0~1) 지점 좌표 보간"""
    if t <= 0: return coords_latlon[0]
    if t >= 1: return coords_latlon[-1]
    # 각 세그먼트 길이
    seg_d = [haversine(*coords_latlon[i], *coords_latlon[i+1]) for i in range(len(coords_latlon)-1)]
    total = sum(seg_d)
    target = t * total
    acc = 0.0
    for i, d in enumerate(seg_d):
        if acc + d >= target:
            ratio = (target - acc) / max(d, 1e-9)
            lat = coords_latlon[i][0] + ratio * (coords_latlon[i+1][0] - coords_latlon[i][0])
            lon = coords_latlon[i][1] + ratio * (coords_latlon[i+1][1] - coords_latlon[i][1])
            return (lat, lon)
        acc += d
    return coords_latlon[-1]

# 4) 페르소나 설정 (요구: 3,4 중심) — 속도는 시간 계산용
PERSONAS = {
    "P3": {"label": "페르소나 3", "speed_kmh": 30},  # 예: 외래/수영회원 등
    "P4": {"label": "페르소나 4", "speed_kmh": 25},  # 예: 조깅/운동 선호
    "ALL": {"label": "전체 보기", "speed_kmh": 28},
}

# 5) 레이아웃
left, mid, right = st.columns([1.2, 1.0, 2.6], gap="large")

with left:
    st.subheader("① 승하차/페르소나")
    route = st.selectbox("운행 노선", list(ROUTE_FILES.keys()), index=2)  # 기본 3번버스
    persona_key = st.selectbox("페르소나", ["P3","P4","ALL"], format_func=lambda k: PERSONAS[k]["label"])
    speed_kmh = PERSONAS[persona_key]["speed_kmh"]

    # 메타에서 정류장 목록(순서) 가져오기
    meta = EXCEL_META.get(route, {})
    seqs = sorted(int(s) for s in meta.keys())
    # 페르소나 필터(3/4번 노선에서만 의미 있음; 그 외는 전체)
    def persona_match(v):
        if persona_key == "ALL": 
            return True
        code = (meta[v].get("persona_code") or "").upper()
        return code.startswith(persona_key)

    filtered = [s for s in seqs if persona_match(s)]
    if len(filtered) < 2:
        # 필터 결과가 너무 적으면 전체 사용
        filtered = seqs

    options = [f"{s:02d}. {meta[s]['name']}" for s in filtered]
    start = st.selectbox("출발 정류장", options, index=0)
    end   = st.selectbox("도착 정류장", options, index=len(options)-1)
    start_seq = int(start.split(".")[0])
    end_seq   = int(end.split(".")[0])

    ride_time = st.time_input("승차 시간", value=None, help="선택 시 화면 우측에 표시만 합니다.")
    go = st.button("최적 동선 생성")

# 6) 노선 라인 로드 + 좌표 준비
gdf = gpd.read_file(ROUTE_FILES[route]).to_crs(4326)
geom = gdf.geometry.iloc[0]
if isinstance(geom, LineString):
    coords_lonlat = list(geom.coords)              # (lon,lat)
    coords_latlon = [(y, x) for (x, y) in coords_lonlat]
else:
    # MultiLine이면 첫 라인만
    geom = list(geom.geoms)[0]
    coords_lonlat = list(geom.coords)
    coords_latlon = [(y, x) for (x, y) in coords_lonlat]

# EXCEL seq 개수 기준으로 폴리라인 상 위치를 균등 매핑 (좌표가 엑셀에 없으므로)
N = max(seqs) if seqs else 1
seq_to_xy = {}
for s in seqs:
    t = (s-1)/max(N-1,1)
    seq_to_xy[s] = point_at_fraction(coords_latlon, t)  # (lat, lon)

# 7) 계산/그리기
with mid:
    st.subheader("② 결과 요약")
    if go:
        a, b = min(start_seq, end_seq), max(start_seq, end_seq)
        t_a, t_b = (a-1)/max(N-1,1), (b-1)/max(N-1,1)
        # 전체 라인 길이를 비율로 잘라 거리/시간 계산(근사)
        total_km = poly_length_km(coords_latlon)
        seg_km = abs(t_b - t_a) * total_km
        seg_min = (seg_km / max(speed_kmh, 1e-6)) * 60.0

        st.metric("📏 이동거리", f"{seg_km:.2f} km")
        st.metric("⏱ 소요시간", f"{seg_min:.1f} 분")
        if ride_time:
            st.caption(f"승차 시간: **{ride_time.strftime('%H:%M')}**")
    else:
        st.info("좌측 설정 후 **최적 동선 생성**을 눌러주세요.")

with right:
    st.subheader("③ 지도")
    # 기본 지도
    center = coords_latlon[len(coords_latlon)//2]
    m = folium.Map(location=center, zoom_start=13, tiles="CartoDB Positron")

    # 전체 노선 라인
    folium.PolyLine(coords_latlon, color="#356df3", weight=6, opacity=0.8).add_to(m)

    # 정류장 마커(필터 반영)
    for s in filtered:
        lat, lon = seq_to_xy[s]
        label = f"{s}. {meta[s]['name']}"
        folium.CircleMarker([lat,lon], radius=4, color="#666", fill=True, fill_opacity=1,
                            tooltip=label).add_to(m)

    # 출발/도착 강조
    if go:
        for s, color, icon in [(start_seq,"green","play"), (end_seq,"red","stop")]:
            lat, lon = seq_to_xy[s]
            folium.Marker([lat,lon], tooltip=f"{s}. {meta[s]['name']}",
                          icon=folium.Icon(color=color, icon=icon)).add_to(m)

    st_folium(m, height=560, use_container_width=True)
