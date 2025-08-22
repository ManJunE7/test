import os, requests, math
import streamlit as st
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="DRT 3·4 최적 동선 (실도로)", layout="wide")

# ▣ Mapbox 토큰: secrets → env → (마지막) 하드코딩
MAPBOX_TOKEN = (st.secrets.get("MAPBOX_TOKEN", "") or os.getenv("MAPBOX_TOKEN", "")).strip()
if not MAPBOX_TOKEN:
    MAPBOX_TOKEN = "YOUR_MAPBOX_TOKEN"  # ← Streamlit Cloud에 Secrets로 넣는 걸 권장

# ▣ 페르소나 정류장 (이름만 정의. 좌표는 라인 위 보간)
STOPDATA = {
    "3번버스": [
        {"seq":1, "name":"우성단 아파트 (RED)"},
        {"seq":2, "name":"북측 아파트 (RED)"},
        {"seq":3, "name":"동측 그린 빌라 (RED)"},
        {"seq":4, "name":"산단 분홍존 → Gate 2 (GREEN)"},
        {"seq":5, "name":"라인 아파트 (RED)"},
        {"seq":6, "name":"청수 쉼터 (RED)"},
        {"seq":7, "name":"중앙시장 주민 허브 (GREEN)"},
        {"seq":8, "name":"산단 기업지원동 (GREEN)"},
        {"seq":9, "name":"체육센터(수영장) (GREEN)"},
        {"seq":10,"name":"교회 앞 골목 (GREEN)"},
    ],
    "4번버스": [
        {"seq":1, "name":"봉명동 빌딩 앞 (RED)"},
        {"seq":2, "name":"상명대 인근 아파트 (RED)"},
        {"seq":3, "name":"천안역 환승/출구 (GREEN)"},
        {"seq":4, "name":"쌍용동 아파트 (RED)"},
        {"seq":5, "name":"병원 정문/외래 접수 (GREEN)"},
        {"seq":6, "name":"성정남부 주택가 (RED)"},
        {"seq":7, "name":"도솔/쌍용공원 입구 (GREEN)"},
        {"seq":8, "name":"남부도서관 인근 주거지 (RED)"},
        {"seq":9, "name":"이마트 정문 (GREEN)"},
        {"seq":10,"name":"사찰/법당 앞 (GREEN)"},
    ],
}
ROUTE_FILES = {"3번버스": "drt_3.shp", "4번버스": "drt_4.shp"}

# ---------- 라인 읽고 (lat,lon) 리스트로 ----------
@st.cache_data(show_spinner=False)
def load_route_latlon(route_name: str):
    shp = ROUTE_FILES[route_name]
    gdf = gpd.read_file(shp).to_crs(4326)
    geom = gdf.geometry.iloc[0]
    if isinstance(geom, MultiLineString):
        geom = max(list(geom.geoms), key=lambda L: L.length)
    if not isinstance(geom, LineString):
        raise ValueError("라인 형식이 아닙니다.")
    lonlat = list(geom.coords)              # (lon,lat)
    return [(y, x) for (x, y) in lonlat]    # (lat,lon)

# ---------- 보조함수: 길이/보간 ----------
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

def point_at_length(coords, cum, target_m):
    if target_m <= 0: return coords[0]
    if target_m >= cum[-1]: return coords[-1]
    for i in range(1,len(cum)):
        if cum[i] >= target_m:
            r = (target_m - cum[i-1]) / max(cum[i] - cum[i-1], 1e-9)
            lat = coords[i-1][0] + r*(coords[i][0]-coords[i-1][0])
            lon = coords[i-1][1] + r*(coords[i][1]-coords[i-1][1])
            return (lat,lon)
    return coords[-1]

def seq_to_pos(seq, N, total_m, coords, cum):
    t = (seq-1)/max(N-1,1)
    return point_at_length(coords, cum, t*total_m)

# ---------- Mapbox Directions/Optimization ----------
def directions_between(a_latlon, b_latlon):
    """두 점을 도로로 연결 (fallback 용)"""
    (lat1,lon1),(lat2,lon2) = a_latlon, b_latlon
    url = f"https://api.mapbox.com/directions/v5/mapbox/driving/{lon1},{lat1};{lon2},{lat2}"
    params = {"geometries":"geojson","overview":"full","access_token":MAPBOX_TOKEN}
    try:
        r = requests.get(url, params=params, timeout=12)
        r.raise_for_status()
        j=r.json()
        if not j.get("routes"): return None,0.0,0.0
        rt=j["routes"][0]
        coords=[(lat,lon) for (lon,lat) in rt["geometry"]["coordinates"]]
        return coords, rt.get("distance",0.0), rt.get("duration",0.0)
    except Exception:
        return None,0.0,0.0

def optimize_trip(coords_latlon, fix_first=True, fix_last=True):
    """
    coords_latlon: [(lat,lon), ...]
    Mapbox Optimization API로 최적 순서 + 전체 경로 반환
    """
    if len(coords_latlon) < 2: return None,[],0.0,0.0
    path = ";".join(f"{lon},{lat}" for (lat,lon) in coords_latlon)
    params = {
        "geometries":"geojson",
        "overview":"full",
        "roundtrip":"false",
        "access_token":MAPBOX_TOKEN
    }
    if fix_first: params["source"]="first"
    if fix_last:  params["destination"]="last"

    url = f"https://api.mapbox.com/optimized-trips/v1/mapbox/driving/{path}"
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        j=r.json()
        trips=j.get("trips",[])
        if not trips: return None,[],0.0,0.0
        trip=trips[0]
        # 전체 경로
        trip_coords=[(lat,lon) for (lon,lat) in trip["geometry"]["coordinates"]]
        # 최적 순서(index)
        wps=j.get("waypoints",[])
        order = sorted(
            [ (wp.get("waypoint_index",-1), i) for i,wp in enumerate(wps) if wp.get("waypoint_index", -1) >= 0 ],
            key=lambda x: x[0]
        )
        ord_idx=[i for _,i in order]
        return trip_coords, ord_idx, trip.get("distance",0.0), trip.get("duration",0.0)
    except Exception:
        return None,[],0.0,0.0

# ---------------- UI ----------------
st.markdown("## 🚌 DRT 3·4 최적 동선 (실도로)")

left, mid, right = st.columns([1.3, 1.1, 2.6], gap="large")

with left:
    route = st.selectbox("운행 노선", ["3번버스", "4번버스"])
    stops = STOPDATA[route]
    labels = [f"{s['seq']:02d}. {s['name']}" for s in stops]

    st.markdown("**승차 정류장(여러 개 가능)**")
    picks = st.multiselect("", labels[:], default=labels[:1], key="pick")
    st.markdown("**하차 정류장(여러 개 가능)**")
    drops = st.multiselect("", labels[:], default=labels[-1:], key="drop")

    fix_first = st.checkbox("첫 정류장 고정(시작점)", True)
    fix_last  = st.checkbox("마지막 정류장 고정(종점)", True)

    go = st.button("최적 동선 계산", type="primary")

# 라인 로드 + 정류장 좌표 생성
try:
    base = load_route_latlon(route)
except Exception as e:
    st.error(f"{route} 라인 로드 실패: {e}")
    st.stop()

total_m = poly_len_m(base)
cum = cumulative(base)
N = len(stops)
seq_to_xy = {s["seq"]: seq_to_pos(s["seq"], N, total_m, base, cum) for s in stops}

def seq_of(label): return int(str(label).split(".")[0])

sel_seqs = sorted(set([seq_of(x) for x in (picks + drops)]))
sel_coords = [seq_to_xy[s] for s in sel_seqs]
sel_names  = [next(x["name"] for x in stops if x["seq"]==s) for s in sel_seqs]

with mid:
    st.markdown("### 📊 요약")
    if not MAPBOX_TOKEN or MAPBOX_TOKEN == "YOUR_MAPBOX_TOKEN":
        st.error("Mapbox 토큰이 설정되어 있지 않습니다. Secrets에 MAPBOX_TOKEN을 추가하세요.")
    elif go and len(sel_coords) >= 2:
        # Optimization API 호출
        trip_coords, order_idx, dist_m, dur_s = optimize_trip(sel_coords, fix_first, fix_last)

        if trip_coords:
            st.metric("📏 총 이동거리", f"{dist_m/1000:.2f} km")
            st.metric("⏱ 예상 소요시간", f"{dur_s/60:.1f} 분")
            st.markdown("**방문 순서(최적화 결과)**")
            for i, idx in enumerate(order_idx, 1):
                st.write(f"- {i}. {sel_names[idx]}")
        else:
            st.warning("최적화 호출 실패 → 구간별 일반 경로로 대체합니다.")
            # 간단 fallback: 선택 순서대로 directions 연결
            dsum, tsum = 0.0, 0.0
            for a,b in zip(sel_coords[:-1], sel_coords[1:]):
                _, d, t = directions_between(a,b)
                dsum += d; tsum += t
            st.metric("📏 총 이동거리(추정)", f"{dsum/1000:.2f} km")
            st.metric("⏱ 예상 소요시간(추정)", f"{tsum/60:.1f} 분")
            st.markdown("**방문 순서(선택 순서)**")
            for i, nm in enumerate(sel_names, 1):
                st.write(f"- {i}. {nm}")
    else:
        st.info("정류장을 고른 뒤 **최적 동선 계산**을 누르세요.")

with right:
    st.markdown("### 🗺️ 경로 시각화")
    m = folium.Map(location=base[len(base)//2], zoom_start=13, tiles="CartoDB Positron")

    # 전체 라인(얇게)
    folium.PolyLine(base, color="#9aa0a6", weight=3, opacity=0.5, tooltip=f"{route} 라인").add_to(m)

    # 정류장(우리가 정한 것만)
    pick_set = {seq_of(x) for x in picks}
    drop_set = {seq_of(x) for x in drops}
    focus    = set(sel_seqs)

    for s in stops:
        lat, lon = seq_to_xy[s["seq"]]
        color = "#1e88e5"
        if s["seq"] in pick_set: color = "#43a047"  # 승차: 초록
        if s["seq"] in drop_set: color = "#e53935"  # 하차: 빨강
        radius = 6 if s["seq"] in focus else 4
        folium.CircleMarker([lat,lon], radius=radius, color=color, fill=True, fill_opacity=1.0,
                            tooltip=f"{s['seq']}. {s['name']}").add_to(m)

    # 최적 경로 라인
    if go and len(sel_coords) >= 2 and MAPBOX_TOKEN and MAPBOX_TOKEN != "YOUR_MAPBOX_TOKEN":
        trip_coords, order_idx, _, _ = optimize_trip(sel_coords, fix_first, fix_last)
        if trip_coords:
            folium.PolyLine(trip_coords, color="#00c853", weight=7, opacity=0.95,
                            tooltip="최적 동선").add_to(m)
        else:
            # fallback: 선택 순서대로 도로 경로 그리기
            for a,b in zip(sel_coords[:-1], sel_coords[1:]):
                coords,_,_ = directions_between(a,b)
                if coords:
                    folium.PolyLine(coords, color="#00c853", weight=7, opacity=0.95).add_to(m)

    st_folium(m, height=620, use_container_width=True)
