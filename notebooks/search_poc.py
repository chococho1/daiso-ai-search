import streamlit as st
import requests
import time
import pandas as pd
from diskcache import Cache
import psycopg2
from psycopg2.extras import RealDictCursor
from util.get_db_config import get_db_config
from llm_search_api import extract_keywords_with_ollama, gemma_search


# --- 1. 초기 설정 및 캐시 로드 ---
cache = Cache("./search_cache")
st.set_page_config(page_title="AI 지능형 검색 데모", layout="wide")

# DB 설정 (기존에 사용하시던 설정 적용)
DB_CONFIG = get_db_config("localDB.properties")

# --- 2. 핵심 함수 정의 ---
# 검색엔진 API 호출
def get_java_search_results(query):
    response = requests.get(f"https://www.daisomall.co.kr/ssn/search/SearchGoods?searchTerm={query}", timeout=2)
    res_json = response.json()
    try:
        results_list = res_json.get("resultSet", {}).get("result", [])
        if results_list:
            documents = results_list[1].get("resultDocuments", [])
            return documents
    except Exception as e:
        print(f"파싱 에러: {e}")
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

    # 1. 키워드 추출 (캐시 적용)
    start_llm = time.time()
    is_cached = query in cache

    # 1. 상태 표시 객체 생성
    status_text = "🚀 캐시에서 데이터를 즉시 불러오는 중..." if is_cached else "🧠 Gemma 2:2B가 의도를 분석하고 키워드를 생성 중..."
    with st.status(status_text, expanded=not is_cached) as status:
        if is_cached:
            keywords = cache[query]
            time.sleep(0.1)
        else:
            keywords = extract_keywords_with_ollama(query)
            cache.set(query, keywords, expire=86400)

        # --- 핵심: 작업 완료 후 상태 업데이트 ---
        status.update(label="✅ 키워드 생성 완료!", state="complete", expanded=False)

    llm_runtime = time.time() - start_llm

    # 화면 분할
    col1, col2 = st.columns(2)

    # --- 좌측: 기존 검색 엔진 ---
    with col1:
        st.subheader("📊 Legacy: 기존 키워드 검색")
        st.info("검색어 일치 기반 (Java API)")

        t_start = time.time()
        java_res = get_java_search_results(query)
        t_end = time.time()

        st.metric("응답 속도", f"{t_end - t_start:.3f} s")
        if java_res:
            df = pd.DataFrame(java_res)
            target_cols = ["pdNo", "exhPdNm", "pdPrc"]

            existing_cols = [c for c in target_cols if c in df.columns]
            df_filtered = df[existing_cols]

            rename_dict = {
                "pdNo": "품번",
                "exhPdNm": "상품명",
                "pdPrc": "가격"
            }

            df_display = df_filtered.rename(columns=rename_dict)
            st.table(df_display.head(10))
        else:
            st.write("결과가 없습니다.")

    # --- 우측: AI 확장 검색 ---
    with col2:
        st.subheader("💡 AI: 지능형 의도 분석 검색")
        st.success(f"확장 키워드: {', '.join(keywords)}")

        t_start = time.time()
        all_ai_results = [] # 모든 키워드의 결과를 담을 리스트
        seen_pd_nos = set()  # 중복 체크를 위한 상품번호 저장소

        # 1. 추출된 각 키워드마다 자바 API 호출
        with st.spinner("연관 상품들을 불러오는 중..."):
            for kw in keywords:
                # 기존에 만든 자바 API 호출 함수 활용
                res = get_java_search_results(kw)
                for item in res:
                    # 상품번호(pdNo)를 기준으로 중복 체크
                    if item['pdNo'] not in seen_pd_nos:
                        all_ai_results.append(item)
                        seen_pd_nos.add(item['pdNo'])

        t_end = time.time()

        # 2. 메트릭 표시 (LLM 시간 + API 호출 합산 혹은 별도 표기)
        st.metric("AI 검색 총 소요 시간", f"{llm_runtime + (t_end - t_start):.3f} s", delta="-90% (Cache)" if is_cached else None)

        # 3. 결과 출력 (원하는 컬럼만 추출)
        st.write(f"✅ 총 {len(all_ai_results)}개의 연관 상품을 찾았습니다.")

        if all_ai_results:
            df_ai = pd.DataFrame(all_ai_results)

            # 필터링 및 한글화
            target_cols = ["pdNo", "exhPdNm", "pdPrc"]
            existing_cols = [c for c in target_cols if c in df_ai.columns]

            df_ai_display = df_ai[existing_cols].rename(columns={
                "pdNo": "상품번호", "exhPdNm": "상품명", "pdPrc": "가격"
            })

            # 상위 15개 정도 보여주기
            st.table(df_ai_display.head(15))
        else:
            st.warning("연관 상품 결과가 없습니다.")


    # --- 하단 분석 섹션 ---
    st.markdown("---")
    st.caption(f"시스템 알림: {'캐시 적중! 즉시 응답 모드입니다.' if is_cached else '신규 쿼리 분석 모드입니다.'}")