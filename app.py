# app_cheonan_drt_cost_protected_rg.py
# ---------------------------------------------------------
# 천안 DRT - Matrix로 상위 N개만 Directions 호출(과금보호) + 역지오코딩으로 이름 매핑
# ---------------------------------------------------------
import math
from pathlib import Path
from glob import glob
import geopandas as gpd
import pandas as pd
import numpy as np
import requests
import streamlit as st
from folium.features import DivIcon
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import folium

APP_TITLE = "천안 DRT - 맞춤형 AI기반 스마트 교통 가이드"
LOGO_URL  = "https://raw.githubusercontent.com/JeongWon4034/cheongju/main/cheongpung_logo.png"
st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="collapsed")

# (동일 스타일) ---------------------------------------------------------------
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

# ----- 토큰/상수 -------------------------------------------------------------
MAPBOX_TOKEN = "여기에_네_토큰"   # ← 실제 토큰으로 교체
PALETTE = ["#4285f4","#34a853","#ea4335","#fbbc04","#7e57c2","#26a69a","#ef6c00","#c2185b"]
MATRIX_MAX_COORDS = 25

# ----- 좌표계 보정/로더(위 코드와 동일) --------------------------------------
KOREA_CRS_CANDIDATES = ["EPSG:5179","EPSG:5181","EPSG:5186","EPSG:5187","EPSG:2097","EPSG:32651","EPSG:32652","EPSG:32653"]
def _looks_like_korea(gdf_wgs84: gpd.GeoDataFrame)->bool:
    if gdf_wgs84.empty: return False
    xs, ys = gdf_wgs84.geometry.x, gdf_wgs84.geometry.y
    return (((xs>120)&(xs<135)&(ys>30)&(ys<45)).mean()>0.9)
def to_wgs84_auto(gdf): 
    g=gdf.copy()
    if g.crs:
        try:
            w=g.to_crs(epsg=4326)
            if _looks_like_korea(w): return w
        except: pass
    for cand in KOREA_CRS_CANDIDATES:
        try:
            tmp=g.copy().set_crs(cand,allow_override=True).to_crs(epsg=4326)
            if _looks_like_korea(tmp): return tmp
        except: continue
    return g.set_crs(epsg=4326) if not g.crs else g.to_crs(epsg=4326)
def _is_text_series(s: pd.Series)->bool:
    return s.dtype=="object" or s.dtype.name.startswith("string")
def _find_first(pat:str):
    try: return next(Path(".").rglob(pat))
    except StopIteration: return None
def _pick_name_col(df: pd.DataFrame):
    for c in ["name","정류장명","정류장","stop_name","station","st_name","poi_name","title","NAME","Name"]:
        if c in df.columns and _is_text_series(df[c]): return c
    return None

@st.cache_data
def load_stops():
    ff=_find_first("ff_drt_dh.shp"); g=None
    if ff:
        g0=gpd.read_file(ff)
        use=None
        cand=[c for c in g0.columns if c.lower()=="bus_stops"]
        if cand:
            c=cand[0]; use=g0[g0[c].astype(str).str.strip().str.lower().isin(["1","true","y","yes","bus_stops"])]
        if use is None or use.empty:
            cat_cols=[c for c in g0.columns if c.lower() in ("layer","type","category","class","feature","theme","kind","group","분류","구분","시설구분")]
            for c in cat_cols:
                m=g0[c].astype(str).str.lower().str.contains(r"bus[\s_\-]*stop",na=False)
                if m.any(): use=g0[m]; break
        if use is None or use.empty:
            use=g0[g0.geom_type.astype(str).str.contains("Point",case=False,na=False)]
        if use.empty: st.error("ff_drt_dh.shp에서 bus_stops 포인트를 찾지 못했습니다."); st.stop()
        g=to_wgs84_auto(use)
    if g is None:
        cands=[]
        for bn in ["cb_tour","stops","poi","bus_stops","drt_points"]:
            cands+=glob(f"**/{bn}.shp",recursive=True)
            cands+=glob(f"**/{bn}.geojson",recursive=True)
            cands+=glob(f"**/{bn}.gpkg",recursive=True)
            cands+=glob(f"**/{bn}.json",recursive=True)
        for p in sorted(set(cands)):
            try:
                g0=gpd.read_file(p)
                pts=g0[g0.geom_type.astype(str).str.contains("Point",case=False,na=False)]
                if pts.empty: continue
                g=to_wgs84_auto(pts); break
            except: continue
    if g is None or g.empty: st.error("정류장(POINT) 레이어를 찾지 못했습니다."); st.stop()
    name_col=_pick_name_col(g)
    if name_col is None:
        g["name"]=[f"정류장_{i+1}" for i in range(len(g))]; name_col="name"
    g=g.rename(columns={name_col:"name"})
    g["lon"],g["lat"]=g.geometry.x,g.geometry.y
    if not ((g["lon"].between(-180,180))&(g["lat"].between(-90,90))).all():
        st.error("좌표 변환 실패(경위도 범위 벗어남)."); st.stop()
    return g[["name","lon","lat","geometry"]]

@st.cache_data
def load_boundary():
    for nm in ["cb_shp","boundary","admin_boundary","cheonan_boundary"]:
        for ext in ["shp","geojson","gpkg","json"]:
            p=_find_first(f"**/{nm}.{ext}")
            if p:
                g0=gpd.read_file(p)
                return g0.to_crs(epsg=4326) if g0.crs else to_wgs84_auto(g0)
    return None

stops=load_stops()
boundary=load_boundary()

ctr_lat=float(stops["lat"].mean()); ctr_lon=float(stops["lon"].mean())
if math.isnan(ctr_lat) or math.isnan(ctr_lon): ctr_lat,ctr_lon=36.80,127.15

# ----- Mapbox API ------------------------------------------------------------
def mapbox_route(lon1,lat1,lon2,lat2,profile="driving",token="",timeout=12):
    if not token: raise RuntimeError("MAPBOX_TOKEN 필요")
    url=f"https://api.mapbox.com/directions/v5/mapbox/{profile}/{lon1},{lat1};{lon2},{lat2}"
    params={"geometries":"geojson","overview":"full","access_token":token}
    r=requests.get(url,params=params,timeout=timeout)
    if r.status_code!=200: raise RuntimeError(f"Directions 오류 {r.status_code}: {r.text[:160]}")
    j=r.json(); routes=j.get("routes",[])
    if not routes: raise RuntimeError("경로가 반환되지 않았습니다.")
    rt=routes[0]
    return rt["geometry"]["coordinates"], float(rt.get("duration",0.0)), float(rt.get("distance",0.0))

def mapbox_matrix(sources_xy,destinations_xy,profile="driving",token="",timeout=12):
    if not token: raise RuntimeError("MAPBOX_TOKEN 필요")
    coords=sources_xy+destinations_xy
    if len(coords)>MATRIX_MAX_COORDS:
        raise RuntimeError(f"Matrix 좌표 총합 {len(coords)}개 — {MATRIX_MAX_COORDS}개 이하로 줄여주세요.")
    coord_str=";".join([f"{x},{y}" for x,y in coords])
    src_idx=";".join(map(str, range(len(sources_xy))))
    dst_idx=";".join(map(str, range(len(sources_xy), len(coords))))
    url=f"https://api.mapbox.com/directions-matrix/v1/mapbox/{profile}/{coord_str}"
    params={"access_token":token,"annotations":"duration,distance","sources":src_idx,"destinations":dst_idx}
    r=requests.get(url,params=params,timeout=timeout)
    if r.status_code!=200: raise RuntimeError(f"Matrix 오류 {r.status_code}: {r.text[:160]}")
    j=r.json()
    return j.get("durations"), j.get("distances")

@st.cache_data(show_spinner=False)
def mapbox_reverse_name(lon, lat, token="", lang="ko"):
    # 주변 POI/주소/도로명을 한국어로 우선 반환 (캐싱)
    try:
        url=f"https://api.mapbox.com/geocoding/v5/mapbox.places/{lon},{lat}.json"
        params={
            "access_token": token,
            "language": lang,
            "types": "poi,address,street,place,neighborhood",
            "limit": 5,
            "country": "KR",
            "proximity": f"{lon},{lat}"
        }
        r=requests.get(url, params=params, timeout=10)
        if r.status_code!=200: return None
        feats=r.json().get("features",[])
        if not feats: return None
        order={"poi":0,"address":1,"street":2,"place":3,"neighborhood":4}
        feats.sort(key=lambda ft: order.get((ft.get("place_type") or [""])[0], 9))
        ft=feats[0]
        label=ft.get("text") or ft.get("place_name")
        # 주소번호가 있으면 살짝 붙이기
        addr=ft.get("address")
        if addr and label and addr not in label:
            label=f"{label} {addr}"
        return label
    except Exception:
        return None

# ----- UI --------------------------------------------------------------------
col1, col2, col3 = st.columns([1.7,1.1,3.2], gap="large")

with col1:
    st.markdown('<div class="section-header">🚏 DRT 노선 추천 설정</div>', unsafe_allow_html=True)
    mode = st.radio("운행 모드", ["차량(운행)","도보(승객 접근)"], horizontal=True)
    profile = "driving" if mode.startswith("차량") else "walking"
    starts = st.multiselect("출발(승차) 정류장", stops["name"].tolist(), key="starts")
    ends   = st.multiselect("도착(하차) 정류장", stops["name"].tolist(), key="ends")
    pairing = st.selectbox("매칭 방식", ["인덱스 쌍(1:1)","모든 조합"], index=1)
    top_k = st.slider("과금보호: 최대 경로 수(N)", 1, 50, 5,
                      help="모든 조합을 Matrix 1회로 평가 후, 소요시간이 짧은 상위 N개만 Directions로 실제 경로 요청.")
    cA, cB = st.columns(2)
    run_clicked   = cA.button("노선 추천")
    clear_clicked = cB.button("초기화")
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
            st.error("MAPBOX_TOKEN을 코드 상단에 입력하세요.")
        else:
            src_xy = [xy(nm) for nm in starts if xy(nm)]
            dst_xy = [xy(nm) for nm in ends if xy(nm)]
            if not src_xy or not dst_xy:
                st.warning("유효한 좌표가 없습니다.")
            else:
                total_min, total_km = 0.0, 0.0
                pairs_to_draw = []

                if pairing.startswith("인덱스"):
                    n = min(len(src_xy), len(dst_xy), top_k)
                    pairs_to_draw = [(i, i) for i in range(n)]
                else:
                    pair_count = len(src_xy) * len(dst_xy)
                    if pair_count == 1:
                        pairs_to_draw = [(0,0)]
                    else:
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
                            def hav(xy1, xy2):
                                R=6371000.0
                                lon1,lat1,lon2,lat2 = map(np.radians,[xy1[0],xy1[1],xy2[0],xy2[1]])
                                dlon=lon2-lon1; dlat=lat2-lat1
                                a=np.sin(dlat/2)**2+np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
                                return 2*R*np.arcsin(np.sqrt(a))
                            scored=[]
                            for i,s in enumerate(src_xy):
                                for j,d in enumerate(dst_xy):
                                    scored.append((hav(s,d), i, j))
                            scored.sort(key=lambda x: x[0])
                            pairs_to_draw = [(i,j) for _,i,j in scored[:top_k]]

                # Directions: 상위 N개만 호출 + 역지오코딩 라벨
                for idx, (si, dj) in enumerate(pairs_to_draw):
                    sxy, exy = src_xy[si], dst_xy[dj]
                    s_label = mapbox_reverse_name(sxy[0], sxy[1], token=MAPBOX_TOKEN) or starts[si]
                    e_label = mapbox_reverse_name(exy[0], exy[1], token=MAPBOX_TOKEN) or ends[dj]
                    try:
                        coords, dur, dist = mapbox_route(sxy[0], sxy[1], exy[0], exy[1], profile=profile, token=MAPBOX_TOKEN)
                        ll = [(c[1], c[0]) for c in coords]
                        folium.PolyLine(ll, color=PALETTE[idx % len(PALETTE)], weight=5, opacity=0.9).add_to(m)
                        mid = ll[len(ll)//2]
                        folium.map.Marker(mid, icon=DivIcon(html=f"<div style='background:{PALETTE[idx%len(PALETTE)]};color:#fff;border-radius:50%;width:26px;height:26px;line-height:26px;text-align:center;font-weight:700;'>{idx+1}</div>")).add_to(m)
                        folium.Marker([sxy[1], sxy[0]], icon=folium.Icon(color="red"),  tooltip=f"승차: {s_label}").add_to(m)
                        folium.Marker([exy[1], exy[0]], icon=folium.Icon(color="blue"), tooltip=f"하차: {e_label}").add_to(m)
                        total_min += dur/60; total_km += dist/1000
                    except Exception as e:
                        st.warning(f"{s_label}→{e_label} Directions 실패: {e}")

                st.session_state["order"]    = [f"{mapbox_reverse_name(xy(starts[si])[0], xy(starts[si])[1], token=MAPBOX_TOKEN) or starts[si]} → {mapbox_reverse_name(xy(ends[dj])[0], xy(ends[dj])[1], token=MAPBOX_TOKEN) or ends[dj]}" for (si,dj) in pairs_to_draw]
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
