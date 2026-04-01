import streamlit as st
import requests
import time
import pandas as pd
from diskcache import Cache
import psycopg2
from psycopg2.extras import RealDictCursor
from util.get_db_config import get_db_config
# gemma_search 내부에 extract_keywords_with_llm가 포함되어 있으므로 하나만 호출 준비
from llm_search_api import gemma_search

# --- 1. 초기 설정 및 캐시 로드 ---
# 캐시 디렉토리 설정 (86400초 = 24시간)
cache = Cache("./search_cache")
st.set_page_config(page_title="AI 지능형 검색 데모", layout="wide")

# DB 설정
DB_CONFIG = get_db_config("localDB.properties")

# --- 2. 핵심 함수 정의 ---
def get_java_search_results(query):
    """기존 자바 기반 키워드 검색 API 호출"""
    try:
        response = requests.get(f"https://www.daisomall.co.kr/ssn/search/SearchGoods?searchTerm={query}", timeout=2)
        res_json = response.json()
        results_list = res_json.get("resultSet", {}).get("result", [])
        if results_list:
            documents = results_list[1].get("resultDocuments", [])
            return documents
    except Exception as e:
        st.error(f"Legacy API 호출 중 오류 발생: {e}")
        return []
    return []

# --- 3. Streamlit UI 구성 ---
st.sidebar.title("⚙️ 시스템 설정")
if st.sidebar.button("캐시 초기화"):
    cache.clear()
    st.sidebar.success("캐시가 비워졌습니다.")

st.title("🌟 다이소 AI 지능형 검색 엔진 데모")
st.markdown("---")

# 검색창
query = st.text_input("검색어를 입력하세요", placeholder="예: 봄나들이 갈 때 필요한 물건")

if query:
    # --- 데이터 처리 시작 ---
    start_time_total = time.time()
    is_cached = query in cache

    # 1. AI 검색 데이터 확보 (캐시 또는 신규 호출)
    if is_cached:
        with st.status("🚀 캐시에서 데이터를 즉시 불러오는 중...", expanded=False) as status:
            ai_search_res = cache[query]
            time.sleep(0.1) # 시각적 효과
            status.update(label="✅ 캐시 데이터 로드 완료!", state="complete")
    else:
        with st.status("🧠 gemma-3-1b-it 가 의도를 분석하고 상품을 찾는 중...", expanded=True) as status:
            # gemma_search 호출 (이 안에서 LLM 키워드 추출 + 벡터 검색이 모두 수행됨)
            ai_search_res = gemma_search(query)
            # 결과 전체를 캐싱 (24시간)
            cache.set(query, ai_search_res, expire=86400)
            status.update(label="✅ AI 분석 및 벡터 검색 완료!", state="complete", expanded=False)
            # 상태 바 바로 아래에 모델 정보 표시
            st.markdown(
                f"""
                <div style='padding-left: 10px; margin-top: -10px; margin-bottom: 10px;'>
                    <span style='color: #444; font-weight: 600; font-size: 0.9rem;'>
                        🤖 사용된 모델: <code style='color: #ff4b4b;'>gemma-3-1b-it</code> |
                        📂 임베딩 모델: <code style='color: #1f77b4;'>BGE-M3-ko</code>
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )

    # 결과 데이터 분해
    keywords = ai_search_res.get("llm_recoommed_keywords", [])
    all_ai_results = ai_search_res.get("search_results", [])
    kw_runtime = ai_search_res.get("kw_runtime", 0)
    total_ai_runtime = time.time() - start_time_total
    # 화면 분할
    col1, col2 = st.columns(2)

    # --- 좌측: 기존 검색 엔진 (Legacy) ---
    with col1:
        st.subheader("📊 Legacy: 기존 키워드 검색")
        st.info("검색어 일치 기반 (Java API)")

        t_start_java = time.time()
        java_res = get_java_search_results(query)
        t_end_java = time.time()

        st.metric("응답 속도", f"{t_end_java - t_start_java:.3f} s")

        if java_res:
            df_legacy = pd.DataFrame(java_res)
            target_cols = ["pdNo", "exhPdNm"]
            existing_cols = [c for c in target_cols if c in df_legacy.columns]

            df_display_legacy = df_legacy[existing_cols].rename(columns={
                "pdNo": "품번",
                "exhPdNm": "상품명"
            })
            st.table(df_display_legacy.head(10))
        else:
            st.write("검색 결과가 없습니다.")

    # --- 우측: AI 확장 검색 (Vector) ---
    with col2:
        st.subheader("💡 AI: 지능형 의도 분석 검색")
        # gemma_search에서 가져온 키워드 리스트 표시
        keyword_list = keywords if isinstance(keywords, list) else [keywords]
        keyword_str = ", ".join(keywords) if isinstance(keywords, list) else str(keywords)
        st.success(f"확장 키워드: {keyword_str}")

        # 시간 측정: 캐시 적중 시 0에 수렴, 신규 쿼리 시 전체 수행 시간 계산
        # total_ai_runtime = time.time() - start_time_total
        st.metric("AI 검색 총 소요 시간", f"{total_ai_runtime:.3f} s", delta="-95% (Cache Hit)" if is_cached else None)
        # LLM 분석 시간이 전체에서 차지하는 비율 계산
        ratio = (float(kw_runtime) / float(total_ai_runtime)) * 100

        st.write(f"✅ 벡터 유사도 기반 상위 {len(all_ai_results)}개의 연관 상품")
        st.caption("※ 유사도가 '1'에 가까울수록 검색 의도와 일치합니다.")

        if all_ai_results:
            df_ai = pd.DataFrame(all_ai_results)

            # pgvector 결과 컬럼명 매핑 (pd_no, pd_nm, similarity)
            rename_dict = {
                "pd_no": "상품번호",
                "pd_nm": "상품명",
                "similarity": "유사도 점수"
            }

            df_ai_display = df_ai.rename(columns=rename_dict)

            # 유사도 점수 포맷팅 (소수점 3자리)
            if "유사도 점수" in df_ai_display.columns:
                df_ai_display["유사도 점수"] = df_ai_display["유사도 점수"].apply(lambda x: f"{float(x):.3f}")

            st.table(df_ai_display)
        else:
            st.warning("DB 내에 연관된 상품 결과가 없습니다.")


        # --- 🚀 새로 추가 : 확장 키워드 기반 Java API 호출 및 중복 제거 ---
        st.markdown("---")
        st.subheader("🔗 확장 키워드 실제 검색 결과 (다이퀘스트 API)")

        all_legacy_from_keywords = []
        seen_pd_nos = set() # 중복 체크를 위한 셋

        with st.spinner('확장 키워드로 실제 상품을 불러오는 중...'):
            for kw in keyword_list:
                # 위에서 정의한 get_java_search_results 재사용
                raw_res = get_java_search_results(kw)

                for item in raw_res:
                    pd_no = item.get("pdNo")
                    # 상품번호가 중복되지 않은 경우만 리스트에 추가
                    if pd_no and pd_no not in seen_pd_nos:
                        seen_pd_nos.add(pd_no)
                        all_legacy_from_keywords.append({
                            "품번": pd_no,
                            "상품명": item.get("exhPdNm"),
                            "출처키워드": kw  # 어떤 키워드 때문에 나왔는지 표시 (선택 사항)
                        })

        if all_legacy_from_keywords:
            df_ext_legacy = pd.DataFrame(all_legacy_from_keywords)
            st.write(f"✨ 중복 제거 후 총 **{len(df_ext_legacy)}**개의 상품을 찾았습니다.")
            # 데이터가 많을 수 있으므로 table 대신 dataframe(스크롤 지원) 사용 권장
            st.dataframe(df_ext_legacy, use_container_width=True)
        else:
            st.warning("확장 키워드에 대한 실제 검색 결과가 없습니다.")
    # --- 하단 분석 섹션 ---
    st.markdown("---")
    st.caption(f"시스템 알림: {'캐시 적중! 즉시 응답 모드입니다.' if is_cached else '신규 쿼리 분석 모드입니다.'} (LLM 분석 소요: {kw_runtime}s)")