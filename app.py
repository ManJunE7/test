import os, math, requests
import streamlit as st
import geopandas as gpd
import folium
from shapely.geometry import LineString, MultiLineString
from streamlit_folium import st_folium

st.set_page_config(page_title="DRT 3·4 다중 승하차 최적 동선", layout="wide")

# ─────────────────────────────────────
# Mapbox 토큰: Secrets/환경변수/하드코딩(네가 준 값) 순
# ─────────────────────────────────────
MAPBOX_TOKEN = (st.secrets.get("MAPBOX_TOKEN", "") or os.getenv("MAPBOX_TOKEN", "")).strip()
if not MAPBOX_TOKEN:
    MAPBOX_TOKEN = "pk.eyJ1IjoiZ3VyMDUxMDgiLCJhIjoiY21lZ2k1Y291MTdoZjJrb2k3bHc3cTJrbSJ9.DElgSQ0rPoRk1eEacPI8uQ"

# ─────────────────────────────────────
# 스샷 기반 정류장 목록(좌표 없음 → 라인상 보간)
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

# 페르소나 속도(시간 추정, km/h) — 필요시 조정
PERSONA_SPEED = {"P1":32, "P2":30, "P3":28, "P4":24, "P5":26, "P6":25, "ALL":28}

# 노선 → shp 파일명
ROUTE_FILES = {"3번버스": "drt_3.shp", "4번버스": "drt_4.shp"}

# ─────────────────────────────────────
# Geo 유틸
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
    coords_lonlat = list(geom.coords)             # (lon,lat)
    coords_latlon = [(y, x) for (x, y) in coords_lonlat]
    return coords_latlon

def hav(lat1, lon1, lat2, lon2):
    R = 6371000.0
    from math import radians, sin, cos, asin, sqrt
    y1, x1, y2, x2 = map(radians, [lat1, lon1, lat2, lon2])
    dy, dx = y2 - y1, x2 - x1
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
    if target <= 0: return coords[0]
    if target >= cum[-1]: return coords[-1]
    for i in range(1, len(cum)):
        if cum[i] >= target:
            r = (target - cum[i-1]) / max(cum[i]-cum[i-1], 1e-9)
            lat = coords[i-1][0] + r*(coords[i][0]-coords[i-1][0])
            lon = coords[i-1][1] + r*(coords[i][1]-coords[i-1][1])
            return (lat, lon)
    return coords[-1]

def seq_to_pos(seq, N, total_m, coords, cum):
    t = (seq-1)/max(N-1,1)
    return point_at_length(coords, cum, t*total_m)

def extract_segment(coords, cum, a_m, b_m):
    """누적거리 a~b 구간 폴리라인 (a<=b 순으로 반환)"""
    if a_m > b_m: a_m, b_m = b_m, a_m
    seg = [point_at_length(coords, cum, a_m)]
    for i in range(1, len(cum)-1):
        if a_m < cum[i] < b_m:
            seg.append(coords[i])
    seg.append(point_at_length(coords, cum, b_m))
    return seg

# ─────────────────────────────────────
# Mapbox Map Matching (도로 스냅)
# ─────────────────────────────────────
def thin_coords(coords_latlon, max_pts=95):
    if len(coords_latlon) <= max_pts: return coords_latlon
    step = math.ceil(len(coords_latlon) / max_pts)
    return coords_latlon[::step] + [coords_latlon[-1]]

def map_match_path(coords_latlon):
    if not MAPBOX_TOKEN:
        return None
    pts = thin_coords(coords_latlon, max_pts=95)
    coord_str = ";".join([f"{lon:.6f},{lat:.6f}" for (lat,lon) in pts])  # lon,lat
    url = f"https://api.mapbox.com/matching/v5/mapbox/driving/{coord_str}"
    params = {"geometries":"geojson", "overview":"full", "tidy":"true", "access_token": MAPBOX_TOKEN}
    try:
        r = requests.get(url, params=params, timeout=12)
        r.raise_for_status()
        data = r.json()
        if not data.get("matchings"): return None
        coords = data["matchings"][0]["geometry"]["coordinates"]  # [ [lon,lat],... ]
        return [(lat,lon) for (lon,lat) in coords]
    except Exception:
        return None

# ─────────────────────────────────────
# UI
# ─────────────────────────────────────
left, mid, right = st.columns([1.25, 1.0, 2.6], gap="large")

with left:
    st.subheader("① 노선 · 선택옵션")
    route = st.selectbox("운행 노선", ["3번버스","4번버스"])
    persona_filter = st.selectbox("페르소나 필터(선택)", ["ALL","P1","P2","P3","P4","P5","P6"])
    speed_kmh = PERSONA_SPEED.get(persona_filter, 28)
    match_to_roads = st.toggle("도로를 따라 그리기 (Map Matching)", value=True)

    # 정류장 옵션 만들기(필터 적용)
    stops = STOPDATA[route]
    def visible(s): return True if persona_filter=="ALL" else (persona_filter in s["personas"])
    filtered = [s for s in stops if visible(s)] or stops
    opt = [f"{s['seq']:02d}. {s['name']}" for s in filtered]

    st.subheader("② 승하차 정류장 선택")
    picks = st.multiselect("승차 정류장(여러 개)", opt, default=opt[:1])
    drops = st.multiselect("하차 정류장(여러 개)", opt, default=opt[-1:])

    direction = st.radio("방향", ["자동(오름차순)", "오름차순", "내림차순"], horizontal=True, index=0)
    go = st.button("최적 동선 생성")

# 기본 데이터 준비
coords = load_route_latlon(route)
total_m = poly_len_m(coords)
cum = cumulative(coords)
N = len(stops)
seq_xy = {s["seq"]: seq_to_pos(s["seq"], N, total_m, coords, cum) for s in stops}

# 선택된 시퀀스 집합
def seq_of(label): return int(str(label).split(".")[0])
selected_seqs = sorted({seq_of(v) for v in (picks + drops)})

with mid:
    st.subheader("③ 결과 요약")
    if go:
        if len(selected_seqs) == 0:
            st.warning("정류장을 한 개 이상 선택하세요.")
        elif len(selected_seqs) == 1:
            st.metric("📏 이동거리", "0.00 km")
            st.metric("⏱ 소요시간", "0.0 분")
            st.info("선택 구간이 1곳이므로 이동이 없습니다.")
        else:
            # 방문 순서 정하기
            if direction == "내림차순":
                order = sorted(selected_seqs, reverse=True)
            else:  # 자동/오름차순 → 오름차순
                order = sorted(selected_seqs)

            # 전체 경로 폴리라인 구성 (선택 구간들을 차례로 연결)
            full_line = []
            total_len_m = 0.0
            for a, b in zip(order[:-1], order[1:]):
                a_m = (a-1)/max(N-1,1) * total_m
                b_m = (b-1)/max(N-1,1) * total_m
                seg = extract_segment(coords, cum, a_m, b_m)
                if a > b: seg = list(reversed(seg))        # 진행방향 정렬
                if full_line and seg:
                    if full_line[-1] == seg[0]: seg = seg[1:]
                full_line += seg

            # 도로 스냅
            matched = map_match_path(full_line) if match_to_roads else None
            line_for_calc = matched or full_line
            total_len_m = poly_len_m(line_for_calc)
            total_km = total_len_m / 1000.0
            total_min = (total_km / max(speed_kmh, 1e-6)) * 60.0

            # 방문 순서 출력
            st.markdown("**방문 순서**")
            for i, s in enumerate(order, 1):
                nm = next(x["name"] for x in stops if x["seq"]==s)
                tag = "승차" if s in {seq_of(v) for v in picks} else ("하차" if s in {seq_of(v) for v in drops} else "경유")
                st.markdown(f"- {i}. {nm} ({tag})")

            st.metric("📏 총 이동거리", f"{total_km:.2f} km")
            st.metric("⏱ 예상 소요시간", f"{total_min:.1f} 분")
    else:
        st.info("정류장들을 고른 뒤 **최적 동선 생성**을 눌러 주세요.")

with right:
    st.subheader("④ 지도")
    center = coords[len(coords)//2]
    m = folium.Map(location=center, zoom_start=13, tiles="CartoDB Positron")

    # 전체 노선
    folium.PolyLine(coords, color="#356df3", weight=5, opacity=0.7, tooltip=f"{route}").add_to(m)

    # 모든 정류장 표시(필터 반영)
    focus_set = set(selected_seqs)
    for s in stops:
        lat, lon = seq_xy[s["seq"]]
        pick = s["seq"] in {seq_of(v) for v in picks}
        drop = s["seq"] in {seq_of(v) for v in drops}
        color = "#1e88e5"
        if pick: color = "#43a047"   # 승차: green
        if drop: color = "#e53935"   # 하차: red
        radius = 6 if s["seq"] in focus_set else 4
        folium.CircleMarker([lat,lon], radius=radius, color=color, fill=True, fill_opacity=1.0,
                            tooltip=f"{s['seq']}. {s['name']}").add_to(m)

    # 선택 경로 그리기
    if go and len(selected_seqs) >= 2:
        if direction == "내림차순":
            order = sorted(selected_seqs, reverse=True)
        else:
            order = sorted(selected_seqs)

        full_line = []
        for a, b in zip(order[:-1], order[1:]):
            a_m = (a-1)/max(N-1,1) * total_m
            b_m = (b-1)/max(N-1,1) * total_m
            seg = extract_segment(coords, cum, a_m, b_m)
            if a > b: seg = list(reversed(seg))
            if full_line and seg and full_line[-1]==seg[0]: seg = seg[1:]
            full_line += seg

        matched = map_match_path(full_line) if match_to_roads else None
        draw_line = matched or full_line

        folium.PolyLine(draw_line, color="#00c853", weight=8, opacity=0.95,
                        tooltip="선택 구간(최적 동선)").add_to(m)

    st_folium(m, height=560, use_container_width=True)
