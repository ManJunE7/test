# app.py
# ---------------------------------------------------------
# 천안 DRT - 맞춤형 AI기반 스마트 교통 가이드
# (노선 추천 + 커버리지 비교(기존 SHP 사용), Fiona 미의존 / pyogrio 경로)
#
# - 기존 DRT:  천안콜 버스 정류장(v250730)_4326.shp  (WGS84, EPSG:4326)
# - 후보(추가) DRT: new_new_drt_full_utf8.(shp/gpkg/geojson)  (WGS84, EPSG:4326)
# - 라우팅: Mapbox Directions
# - 커버리지: 반경 버퍼(기본 100m) 합집합 면적(km²) 비교 + 폴리곤 시각화
# ---------------------------------------------------------

import os, math
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.ops import unary_union
import requests
import streamlit as st
import folium
from folium.plugins import MarkerCluster
from folium.features import DivIcon
from streamlit_folium import st_folium

# ===================== 경로/상수 =====================
EXISTING_SHP   = "천안콜 버스 정류장(v250730)_4326.shp"   # 같은 폴더에 두세요 (EPSG:4326)
CANDIDATE_STEM = "new_new_drt_full_utf8"                  # .shp/.gpkg/.geojson 중 하나

MAPBOX_TOKEN = os.getenv("MAPBOX_TOKEN", "")
if not MAPBOX_TOKEN:
    try:
        MAPBOX_TOKEN = st.secrets["MAPBOX_TOKEN"]
    except Exception:
        pass
if not MAPBOX_TOKEN:
    # 데모용 토큰 (가능하면 본인 토큰으로 교체)
    MAPBOX_TOKEN = "pk.eyJ1IjoiZ3VyMDUxMDgiLCJhIjoiY21lbWppYjByMDV2ajJqcjQyYXUxdzY3byJ9.yLBRJK_Ib6W3p9f16YlIKQ"

PALETTE = ["#e74c3c","#8e44ad","#3498db","#e67e22","#16a085","#2ecc71","#1abc9c","#d35400"]

# ===================== UI 스타일 =====================
st.set_page_config(page_title="천안 DRT - 스마트 교통 가이드", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Noto Sans KR', -apple-system, BlinkMacSystemFont, sans-serif; }
.header{display:flex;align-items:center;gap:12px;margin:8px 0 12px}
.title{font-size:1.6rem;font-weight:800}
.section{font-weight:800;border-bottom:2px solid #f3f4f6;padding-bottom:6px;margin:10px 0}
.legend-chip{display:inline-flex;align-items:center;gap:6px;margin-right:10px}
.legend-dot{width:10px;height:10px;border-radius:50%}
.visit-card{display:flex;align-items:center;gap:10px;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:#fff;border-radius:12px;padding:8px 10px;margin-bottom:6px}
.visit-num{background:#fff;color:#667eea;width:22px;height:22px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:800;font-size:.75rem}
.empty{color:#9ca3af;background:linear-gradient(135deg,#ffecd2 0%,#fcb69f 100%);border-radius:12px;padding:18px 12px;text-align:center}
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="header"><div class="title">천안 DRT - 맞춤형 AI기반 스마트 교통 가이드</div></div>', unsafe_allow_html=True)

# ===================== (선택) 사이드바 리셋 버튼 =====================
with st.sidebar:
    if st.button("🔄 캐시/세션 초기화 후 재실행"):
        try: st.cache_data.clear()
        except: pass
        try: st.cache_resource.clear()
        except: pass
        for k in list(st.session_state.keys()):
            try: del st.session_state[k]
            except: pass
        st.rerun()

# ===================== 파일 로드 유틸 =====================
def read_shp_with_encoding(path: Path) -> gpd.GeoDataFrame:
    """Shapefile을 pyogrio로 읽되, 인코딩 후보를 순차 시도한다."""
    try:
        from pyogrio import read_dataframe as pio
    except Exception:
        st.error("pyogrio가 필요합니다. requirements.txt에 'pyogrio'를 추가하세요.")
        raise

    enc_candidates = []
    # cpg 파일 우선
    try:
        cpg = path.with_suffix(".cpg")
        if cpg.exists():
            enc = cpg.read_text(encoding="ascii", errors="ignore").strip()
            if enc:
                enc_candidates.append(enc.lower())
    except Exception:
        pass

    # 한국어/일반 후보
    enc_candidates += ["cp949", "euc-kr", "utf-8", "latin1", None]
    # 중복 제거
    seen = set(); enc_candidates = [e for e in enc_candidates if not (e in seen or seen.add(e))]

    last_err = None
    for enc in enc_candidates:
        try:
            g = pio(path, encoding=enc)
            return gpd.GeoDataFrame(g, geometry="geometry", crs=getattr(g, "crs", None))
        except Exception as e:
            last_err = e
            continue

    st.error(f"Shapefile 인코딩 해석 실패: {path.name}  (시도: {enc_candidates})")
    if last_err:
        st.exception(last_err)
    raise RuntimeError("Shapefile read failed")

def read_any_vector(path_stem: str) -> gpd.GeoDataFrame:
    """stem을 받아 .shp/.gpkg/.geojson 중 존재하는 것 읽기 (WGS84로 반환, 포인트 아니면 대표점)."""
    for ext in (".shp", ".gpkg", ".geojson"):
        p = Path(f"./{path_stem}{ext}")
        if p.exists():
            if ext == ".shp":
                g = read_shp_with_encoding(p)
            else:
                g = gpd.read_file(p)

            try:
                if g.crs and g.crs.to_epsg() != 4326:
                    g = g.to_crs(epsg=4326)
            except Exception:
                pass

            # 포인트가 아니면 대표점 사용
            if not g.geom_type.astype(str).str.contains("Point", case=False, na=False).any():
                g = g.copy(); g["geometry"] = g.geometry.representative_point()
            return g
    st.error(f"'{path_stem}.shp/.gpkg/.geojson' 파일을 같은 폴더에 두세요."); st.stop()

def read_existing_shp(path: str) -> gpd.GeoDataFrame:
    """기존 DRT Shapefile을 인코딩 자동판별로 읽고, 이름/좌표 컬럼 구성."""
    p = Path(path)
    if not p.exists():
        st.error(f"기존 DRT shapefile이 보이지 않습니다: {path}")
        st.stop()

    if p.suffix.lower() == ".shp":
        g = read_shp_with_encoding(p)
    else:
        g = gpd.read_file(p)

    try:
        if g.crs and g.crs.to_epsg() != 4326:
            g = g.to_crs(epsg=4326)
    except Exception:
        pass

    # 이름 컬럼 추정(없으면 자동 생성)
    name_col = None
    for c in ["정류장명", "정류소명", "name", "NAME", "정류장", "정류소"]:
        if c in g.columns:
            name_col = c; break
    if name_col is None:
        g["name"] = [f"기존DRT_{i+1}" for i in range(len(g))]
    else:
        g["name"] = g[name_col].astype(str)

    # 포인트가 아니면 대표점 사용
    if not g.geom_type.astype(str).str.contains("Point", case=False, na=False).any():
        g = g.copy(); g["geometry"] = g.geometry.representative_point()

    g["lon"] = g.geometry.x
    g["lat"] = g.geometry.y
    return g[["name","lon","lat","geometry"]]

# ===================== 라우팅/순회 유틸 =====================
def haversine(xy1, xy2):
    R=6371000.0
    lon1,lat1,lon2,lat2 = map(np.radians,[xy1[0],xy1[1],xy2[0],xy2[1]])
    dlon=lon2-lon1; dlat=lat2-lat1
    a=np.sin(dlat/2)**2+np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

def mapbox_route(lon1,lat1,lon2,lat2, profile="driving", token="", timeout=12):
    if not token: raise RuntimeError("MAPBOX_TOKEN 필요")
    url=f"https://api.mapbox.com/directions/v5/mapbox/{profile}/{lon1},{lat1};{lon2},{lat2}"
    params={"geometries":"geojson","overview":"full","access_token":token}
    r=requests.get(url,params=params,timeout=timeout)
    if r.status_code!=200:
        raise RuntimeError(f"Directions 오류 {r.status_code}: {r.text[:160]}")
    j=r.json(); routes=j.get("routes",[])
    if not routes: raise RuntimeError("경로가 반환되지 않았습니다.")
    rt=routes[0]; return rt["geometry"]["coordinates"], float(rt.get("duration",0.0)), float(rt.get("distance",0.0))

def greedy_pairing(src_xy: List[Tuple[float,float]], dst_xy: List[Tuple[float,float]]) -> List[int]:
    m, n = len(src_xy), len(dst_xy)
    if n == 0: return []
    used = set(); mapping = [-1]*m
    for i in range(m):
        dists = [(haversine(src_xy[i], dst_xy[j]), j) for j in range(n) if j not in used]
        dists.sort(key=lambda x: x[0])
        if dists:
            j = dists[0][1]; mapping[i] = j; used.add(j)
    unused = [j for j in range(n) if j not in used]; ui = 0
    for i in range(m):
        if mapping[i] == -1 and ui < len(unused): mapping[i] = unused[ui]; ui += 1
    return mapping

def build_single_vehicle_steps(starts: List[str], ends: List[str], stops_df: pd.DataFrame) -> List[dict]:
    def xy(label):
        r = stops_df.loc[stops_df["name"]==label]
        if r.empty: return None
        rr = r.iloc[0]; return (float(rr["lon"]), float(rr["lat"]))
    src_xy = [xy(nm) for nm in starts if xy(nm)]
    dst_xy = [xy(nm) for nm in ends if xy(nm)]
    if not src_xy or not dst_xy: return []
    mapping = greedy_pairing(src_xy, dst_xy)
    remaining = list(range(len(src_xy))); order=[]
    cur_i = 0; remaining.remove(cur_i)
    order += [{"kind":"pickup","name":starts[cur_i],"xy":src_xy[cur_i]},
              {"kind":"drop","name":ends[mapping[cur_i]],"xy":dst_xy[mapping[cur_i]]}]
    current_point = dst_xy[mapping[cur_i]]
    while remaining:
        nxt = min(remaining, key=lambda i: haversine(current_point, src_xy[i]))
        remaining.remove(nxt)
        order.append({"kind":"pickup","name":starts[nxt],"xy":src_xy[nxt]})
        order.append({"kind":"drop","name":ends[mapping[nxt]],"xy":dst_xy[mapping[nxt]]})
        current_point = dst_xy[mapping[nxt]]
    return order

# ===================== 커버리지(버퍼→합집합→면적) =====================
def coverage_union_and_area(points_gdf: gpd.GeoDataFrame, radius_m: int = 100):
    """포인트 GeoDataFrame → 3857 버퍼 → unary_union → 면적(km²) + WGS84 폴리곤"""
    if points_gdf.empty:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"), 0.0
    g = points_gdf.to_crs(epsg=3857)
    unioned = unary_union(g.buffer(radius_m))
    area_km2 = float(gpd.GeoSeries([unioned], crs="EPSG:3857").area.iloc[0] / 1_000_000)
    out = gpd.GeoDataFrame(geometry=[unioned], crs="EPSG:3857").to_crs(epsg=4326)
    return out, area_km2

# ===================== 데이터 로드 =====================
@st.cache_data
def load_existing_candidates():
    existing = read_existing_shp(EXISTING_SHP)   # 기존 DRT
    cand     = read_any_vector(CANDIDATE_STEM)   # 후보(추가)

    # 후보 이름/좌표 보정
    if "jibun" in cand.columns and "name" not in cand.columns:
        cand["name"] = cand["jibun"].astype(str)
    else:
        cand["name"] = cand.get("name", cand.get("jibun", pd.Series([f"후보_{i+1}" for i in range(len(cand))]))).astype(str)
    cand["lon"] = cand.geometry.x
    cand["lat"] = cand.geometry.y
    cand = cand[["name","lon","lat","geometry"]]

    return existing, cand

existing_gdf, cand_gdf = load_existing_candidates()

# ===================== 라우팅/경로 추천 (후보 데이터 기준) =====================
st.markdown('<div class="section">🚏 노선 추천</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns([1.8,1.2,3.2], gap="large")

with c1:
    st.caption(f"후보 정류장(추가): {len(cand_gdf)}개  |  기존 정류장: {len(existing_gdf)}개")
    mode    = st.radio("운행 모드", ["차량(운행)","도보(승객 접근)"], horizontal=True)
    profile = "driving" if mode.startswith("차량") else "walking"
    all_names = cand_gdf["name"].tolist()
    starts = st.multiselect("출발(승차) 정류장", all_names)
    ends   = st.multiselect("도착(하차) 정류장", all_names)
    route_mode = st.radio("노선 모드", ["개별쌍(모든 조합)","단일 차량(연속 경로)"], index=1)
    st.markdown(
        '<span class="legend-chip"><span class="legend-dot" style="background:#e74c3c"></span>첫 승차</span>'
        '<span class="legend-chip"><span class="legend-dot" style="background:#8e44ad"></span>중간 승차</span>'
        '<span class="legend-chip"><span class="legend-dot" style="background:#3498db"></span>하차</span>',
        unsafe_allow_html=True
    )
    b_run = st.button("노선 추천 실행")

with c2:
    st.markdown("**방문 순서**")
    ord_list = st.session_state.get("order", [])
    if ord_list:
        for i, nm in enumerate(ord_list, 1):
            st.markdown(f'<div class="visit-card"><div class="visit-num">{i}</div><div>{nm}</div></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="empty">경로 생성 후 표시됩니다</div>', unsafe_allow_html=True)
    st.metric("소요시간(합)", f"{st.session_state.get('duration', 0.0):.1f}분")
    st.metric("이동거리(합)", f"{st.session_state.get('distance', 0.0):.2f}km")

with c3:
    ctr_lat = float(cand_gdf["lat"].mean()) if len(cand_gdf) else float(existing_gdf["lat"].mean())
    ctr_lon = float(cand_gdf["lon"].mean()) if len(cand_gdf) else float(existing_gdf["lon"].mean())
    if math.isnan(ctr_lat) or math.isnan(ctr_lon): ctr_lat, ctr_lon = 36.80, 127.15

    m = folium.Map(location=[ctr_lat, ctr_lon], zoom_start=12, tiles="CartoDB Positron", control_scale=True)
    mc = MarkerCluster().add_to(m)
    for _, r in cand_gdf.iterrows():
        folium.Marker([r["lat"], r["lon"]], tooltip=f"[후보]{r['name']}", icon=folium.Icon(color="gray")).add_to(mc)

    if b_run:
        if not starts or not ends:
            st.warning("출발/도착 정류장을 각각 1개 이상 선택하세요.")
        elif not MAPBOX_TOKEN:
            st.error("MAPBOX_TOKEN을 설정하세요.")
        else:
            def xy_from(df, nm):
                row = df.loc[df["name"]==nm]
                if row.empty: return None
                rr = row.iloc[0]; return (float(rr["lon"]), float(rr["lat"]))

            total_min, total_km = 0.0, 0.0
            order_names = []

            if route_mode.startswith("개별쌍"):
                for i, s in enumerate(starts):
                    for j, e in enumerate(ends):
                        sxy, exy = xy_from(cand_gdf, s), xy_from(cand_gdf, e)
                        if not sxy or not exy: continue
                        try:
                            coords, dur, dist = mapbox_route(sxy[0], sxy[1], exy[0], exy[1], profile=profile, token=MAPBOX_TOKEN)
                            ll = [(c[1], c[0]) for c in coords]
                            folium.PolyLine(ll, color=PALETTE[(i+j) % len(PALETTE)], weight=5, opacity=0.9).add_to(m)
                            total_min += dur/60; total_km += dist/1000
                            order_names.append(f"{s} → {e}")
                        except Exception as e:
                            st.warning(f"{s}→{e} 실패: {e}")
            else:
                steps = build_single_vehicle_steps(starts, ends, cand_gdf)

                def badge(n, color):
                    return ("<div style='background:"+color+";color:#fff;"
                            "border:2px solid #fff;border-radius:50%;width:30px;height:30px;"
                            "line-height:30px;text-align:center;font-weight:800;"
                            "box-shadow:0 2px 6px rgba(0,0,0,.35);font-size:13px;'>"+str(n)+"</div>")

                prev = None
                for idx, step in enumerate(steps, start=1):
                    lon, lat = step["xy"]; name = step["name"]
                    color = "#e74c3c" if (step["kind"]=="pickup" and idx==1) else ("#8e44ad" if step["kind"]=="pickup" else "#3498db")
                    folium.Marker([lat, lon], tooltip=f"{idx}. {step['kind']} : {name}",
                                  icon=DivIcon(html=badge(idx, color))).add_to(m)
                    if prev is not None:
                        try:
                            coords, dur, dist = mapbox_route(prev[0], prev[1], lon, lat, profile=profile, token=MAPBOX_TOKEN)
                            ll = [(c[1], c[0]) for c in coords]
                            folium.PolyLine(ll, color=PALETTE[(idx-1) % len(PALETTE)], weight=5, opacity=0.9).add_to(m)
                            total_min += dur/60; total_km += dist/1000
                        except Exception as e:
                            st.warning(f"세그먼트 {idx-1}→{idx} 실패: {e}")
                    prev = (lon, lat); order_names.append(name)

            st.session_state["order"]    = order_names
            st.session_state["duration"] = total_min
            st.session_state["distance"] = total_km

    st_folium(m, height=510, returned_objects=[], use_container_width=True, key="routing_map")

# ===================== 커버리지 비교(선택과 무관, 전체 기준) =====================
st.markdown('<div class="section">🗺️ 커버리지 비교 (반경 100m · 전체 기준)</div>', unsafe_allow_html=True)

radius_m = st.slider("커버리지 반경(미터)", min_value=50, max_value=300, value=100, step=10)

# 1) 기존/후보 포인트 준비
exist_pts = existing_gdf[["name","lon","lat","geometry"]].copy()
cand_pts  = cand_gdf[["name","lon","lat","geometry"]].copy()

# 2) 면적 계산 (기존 / 기존+후보)
base_poly, base_km2 = coverage_union_and_area(exist_pts, radius_m=radius_m)
prop_poly, prop_km2 = coverage_union_and_area(pd.concat([exist_pts, cand_pts], ignore_index=True), radius_m=radius_m)
delta_area = prop_km2 - base_km2
inc_rate   = (delta_area / base_km2 * 100) if base_km2 > 0 else (100.0 if prop_km2 > 0 else 0.0)

# 3) 메트릭
mc1, mc2, mc3, mc4 = st.columns(4)
mc1.metric("기존 커버 면적", f"{base_km2:.3f} km²")
mc2.metric("제안(기존+추가) 면적", f"{prop_km2:.3f} km²")
mc3.metric("면적 증가", f"{delta_area:+.3f} km²")
mc4.metric("증가율", f"{inc_rate:+.1f}%")

# 4) 지도 표시 (기존=빨강 폴리곤, 제안=초록 폴리곤, 점: 기존/후보)
ctr_lat2 = float(pd.concat([exist_pts["lat"], cand_pts["lat"]]).mean())
ctr_lon2 = float(pd.concat([exist_pts["lon"], cand_pts["lon"]]).mean())
if math.isnan(ctr_lat2) or math.isnan(ctr_lon2): ctr_lat2, ctr_lon2 = 36.80, 127.15

m2 = folium.Map(location=[ctr_lat2, ctr_lon2], zoom_start=12, tiles="CartoDB Positron", control_scale=True)

fg_exist = folium.FeatureGroup(name=f"기존 정류장({len(exist_pts)})", show=False).add_to(m2)
for _, r in exist_pts.iterrows():
    folium.CircleMarker([r["lat"], r["lon"]], radius=5, color="#b91c1c", fill=True, fill_color="#ef4444",
                        fill_opacity=0.9, tooltip=f"[기존] {r['name']}").add_to(fg_exist)

fg_cand = folium.FeatureGroup(name=f"추가(후보) 정류장({len(cand_pts)})", show=False).add_to(m2)
for _, r in cand_pts.iterrows():
    folium.CircleMarker([r["lat"], r["lon"]], radius=5, color="#1e3a8a", fill=True, fill_color="#3b82f6",
                        fill_opacity=0.9, tooltip=f"[후보] {r['name']}").add_to(fg_cand)

if not base_poly.empty:
    folium.GeoJson(base_poly.__geo_interface__, name="기존 커버",
                   style_function=lambda x: {"color":"#ef4444","fillColor":"#ef4444","fillOpacity":0.15,"weight":2}).add_to(m2)
if not prop_poly.empty:
    folium.GeoJson(prop_poly.__geo_interface__, name="제안 커버",
                   style_function=lambda x: {"color":"#10b981","fillColor":"#10b981","fillOpacity":0.15,"weight":2}).add_to(m2)

folium.LayerControl(collapsed=True).add_to(m2)
st_folium(m2, height=560, returned_objects=[], use_container_width=True, key="coverage_map_all")
