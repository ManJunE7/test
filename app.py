import os, math
import streamlit as st
import geopandas as gpd
import folium
from shapely.geometry import LineString, MultiLineString
from streamlit_folium import st_folium

st.set_page_config(page_title="DRT 3·4 페르소나 경로", layout="wide")

# ─────────────────────────────────────
# 0) 스샷 기반 정류장(순서/이름/페르소나) — 좌표는 없음
#    ※ 필요하면 문자열만 수정하세요.
# ─────────────────────────────────────
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

# 페르소나 별 속도(시간 추정용). 필요 시 조정하세요.
PERSONA_SPEED = {
    "P1": 32,  # 교대 근로자
    "P2": 30,  # 직장인/어르신
    "P3": 28,  # 외래/수영회원
    "P4": 24,  # 조깅러/성도
    "P5": 26,  # 주민/복지
    "P6": 25,  # 신도
    "ALL": 28,
}

# 노선 → shp 파일명 매핑 (레포 루트에 있어야 함)
ROUTE_FILES = {"3번버스": "drt_3.shp", "4번버스": "drt_4.shp"}

# ─────────────────────────────────────
# 1) 지오메트리 로딩 + 라인 좌표(lat,lon)
# ─────────────────────────────────────
def load_route_latlon(route_name):
    shp = ROUTE_FILES[route_name]
    if not os.path.exists(shp):
        st.error(f"{shp} 파일이 필요합니다. 레포 루트에 올려주세요.")
        st.stop()
    gdf = gpd.read_file(shp).to_crs(4326)
    geom = gdf.geometry.iloc[0]
    if isinstance(geom, MultiLineString):
        geom = max(list(geom.geoms), key=lambda L: L.length)
    if not isinstance(geom, LineString):
        st.error("라인 형식이 올바르지 않습니다.")
        st.stop()
    coords_lonlat = list(geom.coords)            # (lon,lat)
    coords_latlon = [(y, x) for (x, y) in coords_lonlat]
    return coords_latlon

def hav(lat1, lon1, lat2, lon2):
    R = 6371000.0
    from math import radians, sin, cos, asin, sqrt
    y1, x1, y2, x2 = map(radians, [lat1, lon1, lat2, lon2])
    dy, dx = y2 - y1, x2 - x1
    h = sin(dy/2)**2 + cos(y1)*cos(y2)*sin(dx/2)**2
    return 2*R*asin(sqrt(h))

def poly_length_m(coords):
    return sum(hav(*coords[i], *coords[i+1]) for i in range(len(coords)-1))

def cumulative_dist(coords):
    """각 꼭짓점까지의 누적거리(m)"""
    cum = [0.0]
    for i in range(len(coords)-1):
        cum.append(cum[-1] + hav(*coords[i], *coords[i+1]))
    return cum

def point_at_length(coords, cum, target):
    """누적거리 target(m) 지점의 보간 좌표(lat,lon)"""
    if target <= 0: return coords[0]
    if target >= cum[-1]: return coords[-1]
    for i in range(1, len(cum)):
        if cum[i] >= target:
            ratio = (target - cum[i-1]) / max(cum[i]-cum[i-1], 1e-9)
            lat = coords[i-1][0] + ratio*(coords[i][0]-coords[i-1][0])
            lon = coords[i-1][1] + ratio*(coords[i][1]-coords[i-1][1])
            return (lat, lon)
    return coords[-1]

def seq_to_position(seq, N, total_m, coords, cum):
    """정류장 seq(1..N)를 폴리라인의 균등 길이 비율로 매핑"""
    t = (seq-1)/max(N-1,1)
    return point_at_length(coords, cum, t*total_m)

def extract_segment(coords, cum, a_m, b_m):
    """누적거리 a~b 구간의 폴리라인을 추출"""
    if a_m > b_m: a_m, b_m = b_m, a_m
    seg = [point_at_length(coords, cum, a_m)]
    for i in range(1, len(cum)-1):
        if a_m < cum[i] < b_m:
            seg.append(coords[i])
    seg.append(point_at_length(coords, cum, b_m))
    return seg

# ─────────────────────────────────────
# 2) UI
# ─────────────────────────────────────
left, mid, right = st.columns([1.2, 1.0, 2.6], gap="large")

with left:
    st.subheader("① 노선/페르소나/승하차")
    route = st.selectbox("운행 노선", ["3번버스","4번버스"])
    persona_pick = st.selectbox("페르소나", ["P3","P4","ALL"])
    speed_kmh = PERSONA_SPEED.get(persona_pick, 28)

    stops = STOPDATA[route]
    N = len(stops)

    # 페르소나 필터(해당 페르소나 포함된 정류장 우선 표출)
    def visible(s):
        return True if persona_pick=="ALL" else (persona_pick in s["personas"])

    vis = [s for s in stops if visible(s)]
    if len(vis) < 2:         # 너무 적으면 전체 보여주기
        vis = stops

    label = lambda s: f"{s['seq']:02d}. {s['name']}"
    start = st.selectbox("출발 정류장", [label(s) for s in vis], index=0)
    end   = st.selectbox("도착 정류장", [label(s) for s in vis], index=len(vis)-1)

    start_seq = int(start.split(".")[0])
    end_seq   = int(end.split(".")[0])

    ride_time = st.time_input("승차 시간", value=None, help="입력 시 결과에 함께 표기됩니다.")
    go = st.button("최적 동선 생성")

# ─────────────────────────────────────
# 3) 계산
# ─────────────────────────────────────
coords = load_route_latlon(route)
total_m = poly_length_m(coords)
cum = cumulative_dist(coords)

# 모든 정류장의 보간 좌표
seq_xy = {s["seq"]: seq_to_position(s["seq"], N, total_m, coords, cum) for s in stops}

with mid:
    st.subheader("② 결과 요약")
    if go:
        # 선택 구간 길이/시간
        a = (start_seq-1)/max(N-1,1) * total_m
        b = (end_seq-1)/max(N-1,1) * total_m
        seg_m = abs(b-a)
        seg_km = seg_m / 1000.0
        seg_min = (seg_km / max(speed_kmh, 1e-6)) * 60.0

        # 방문 순서(시퀀스 방향)
        step = 1 if end_seq >= start_seq else -1
        visit_names = [next(s["name"] for s in stops if s["seq"]==i) 
                       for i in range(start_seq, end_seq+step, step)]

        st.markdown("**방문 순서**")
        for i, nm in enumerate(visit_names, 1):
            st.markdown(f"- {i}. {nm}")

        st.metric("📏 이동거리", f"{seg_km:.2f} km")
        st.metric("⏱ 소요시간", f"{seg_min:.1f} 분")
        if ride_time:
            st.caption(f"승차 시간: **{ride_time.strftime('%H:%M')}**")
    else:
        st.info("좌측에서 선택 후 **최적 동선 생성**을 눌러주세요.")

# ─────────────────────────────────────
# 4) 지도
# ─────────────────────────────────────
with right:
    st.subheader("③ 지도")
    center = coords[len(coords)//2]
    m = folium.Map(location=center, zoom_start=13, tiles="CartoDB Positron")

    # 전체 노선
    folium.PolyLine(coords, color="#356df3", weight=6, opacity=0.75, tooltip=f"{route}").add_to(m)

    # 정류장 마커(페르소나 우선 강조)
    for s in stops:
        lat, lon = seq_xy[s["seq"]]
        is_focus = (persona_pick=="ALL") or (persona_pick in s["personas"])
        color = "#ff7043" if is_focus else "#9aa0a6"
        folium.CircleMarker([lat,lon], radius=5, color=color, fill=True, fill_opacity=1.0,
                            tooltip=f"{s['seq']}. {s['name']} ({'/'.join(s['personas'])})").add_to(m)

    # 선택 구간 하이라이트
    if go:
        a = (start_seq-1)/max(N-1,1) * total_m
        b = (end_seq-1)/max(N-1,1) * total_m
        seg_line = extract_segment(coords, cum, a, b)
        folium.PolyLine(seg_line, color="#00c853", weight=8, opacity=0.95,
                        tooltip="선택 구간").add_to(m)

        for seq, (lat,lon) in [(start_seq, seq_xy[start_seq]), (end_seq, seq_xy[end_seq])]:
            nm = next(s["name"] for s in stops if s["seq"]==seq)
            icon = "play" if seq==start_seq else "stop"
            colr = "green" if seq==start_seq else "red"
            folium.Marker([lat,lon], tooltip=f"{seq}. {nm}",
                          icon=folium.Icon(color=colr, icon=icon)).add_to(m)

    st_folium(m, height=560, use_container_width=True)
