import streamlit as st
import geopandas as gpd
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from folium.features import DivIcon
from shapely.geometry import Point
import osmnx as ox
import requests
from streamlit_folium import st_folium
import math

# ──────────────────────────────
# ✅ 데이터 로드
# ──────────────────────────────
@st.cache_data
def load_data():
    try:
        stops = gpd.read_file("./new_drt.shp").to_crs(epsg=4326)
        stops["lon"], stops["lat"] = stops.geometry.x, stops.geometry.y

        bus_data = {}
        for i in range(1, 5):
            bus_data[f"drt_{i}"] = gpd.read_file(f"./drt_{i}.shp").to_crs(epsg=4326)

        return stops, bus_data
    except Exception as e:
        st.error(f"❌ 데이터 로드 실패: {str(e)}")
        return None, None

st.set_page_config(
    page_title="천안 DRT 최적 노선",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<div class="header-container" style="text-align:center; margin-bottom:1rem;">
    <h1 style="font-size:2.2rem; font-weight:700; color:#202124;">🚌 천안 DRT 최적 노선</h1>
</div>
""", unsafe_allow_html=True)

stops, bus_data = load_data()
if stops is None:
    st.stop()

# ──────────────────────────────
# ✅ 레이아웃
# ──────────────────────────────
col1, col2, col3 = st.columns([1.3, 1.2, 3], gap="large")

# ------------------------------
# [좌] 출발/도착 선택
# ------------------------------
with col1:
    st.markdown("### 🚗 추천경로 설정")
    start = st.selectbox("출발 정류장", stops["name"].unique())
    end = st.selectbox("도착 정류장", stops["name"].unique())
    time = st.time_input("승차 시간", value=pd.to_datetime("07:30").time())
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        create_clicked = st.button("경로 생성")
    with col_btn2:
        clear_clicked = st.button("초기화")

# ------------------------------
# [중간] 정보 출력
# ------------------------------
with col2:
    st.markdown("### 📍 정류장 순서")
    if "order" not in st.session_state:
        st.session_state["order"] = []
    if "duration" not in st.session_state:
        st.session_state["duration"] = 0.0
    if "distance" not in st.session_state:
        st.session_state["distance"] = 0.0

    if st.session_state["order"]:
        for i, name in enumerate(st.session_state["order"], 1):
            st.markdown(f"- {i}. {name}")
    else:
        st.info("경로 생성 후 표시됩니다")

    st.metric("⏱️ 소요시간", f"{st.session_state['duration']:.1f}분")
    st.metric("📏 이동거리", f"{st.session_state['distance']:.2f}km")

# ------------------------------
# [우] 지도
# ------------------------------
with col3:
    st.markdown("### 🗺️ 추천경로 지도시각화")

    clat, clon = stops["lat"].mean(), stops["lon"].mean()
    m = folium.Map(location=[clat, clon], zoom_start=13, tiles="CartoDB Positron")

    # 모든 정류장 표시
    mc = MarkerCluster().add_to(m)
    for _, row in stops.iterrows():
        folium.Marker([row.lat, row.lon],
                      popup=row["name"],
                      tooltip=row["name"],
                      icon=folium.Icon(color="blue", icon="bus", prefix="fa")
        ).add_to(mc)

    # 경로 생성 시각화
    if create_clicked:
        try:
            order = [start, end]
            st.session_state["order"] = order
            st.session_state["duration"] = 12.3   # 예시값
            st.session_state["distance"] = 5.8    # 예시값

            # 출발지/도착지 강조
            srow = stops[stops["name"] == start].iloc[0]
            erow = stops[stops["name"] == end].iloc[0]
            folium.Marker([srow.lat, srow.lon], icon=folium.Icon(color="green", icon="play")).add_to(m)
            folium.Marker([erow.lat, erow.lon], icon=folium.Icon(color="red", icon="stop")).add_to(m)

            # 버스별 샘플 노선 표시 (drt_1.shp ~ drt_4.shp)
            colors = ["#4285f4", "#ea4335", "#34a853", "#fbbc04"]
            for i, (bus, gdf) in enumerate(bus_data.items()):
                if not gdf.empty:
                    folium.PolyLine([(y, x) for x, y in gdf.geometry.iloc[0].coords],
                                    color=colors[i], weight=5, opacity=0.7,
                                    tooltip=f"{bus} 노선").add_to(m)

            st.success("✅ 경로가 생성되었습니다!")
        except Exception as e:
            st.error(f"경로 생성 오류: {str(e)}")

    st_folium(m, width="100%", height=520)
