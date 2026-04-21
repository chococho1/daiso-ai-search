import os
import re
import requests
import json
import torch
import time
import psycopg2
import sqlite3
import logging
import numpy as np
import streamlit as st
from psycopg2 import pool
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
from psycopg2.extras import RealDictCursor
from google import genai
from google.genai import types
from util.get_db_config import get_db_config
from util.constants import system_instruction

# --- 1. 모델 및 리소스 설정 ---
# 10코어 사양 최적화: 임베딩 모델에 4개 코어 할당
torch.set_num_threads(4)

# 성능과 속도의 밸런스가 가장 좋은 Flash 모델 사용
# client = genai.Client(api_key="AIzaSyCCRWloF0cRPHTBNsvoHJ6uWQI6qGeHE5w")
# client = genai.Client(api_key="")
client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
print("BGE-M3 임베딩 모델 로딩 중...")
embed_model = SentenceTransformer('dragonkue/BGE-m3-ko', device='cpu')

print("--- [검증] 내 API 키로 접근 가능한 모델 목록 ---")
try:
    # 최신 SDK는 리스트를 가져올 때 속성 접근 방식을 다르게 합니다.
    for m in client.models.list():
        print(f"✅ 사용 가능 모델 ID: {m.name}")
except Exception as e:
    print(f"❌ 모델 목록 조회 실패: {e}")
print("------------------------------------------")
app = FastAPI()
DB_CONFIG = get_db_config("localDB.properties")

# --- 2. Connection Pool 초기화 ---
# minconn을 5로 높여서 서버 시작 시 미리 통로를 열어둡니다.
db_pool = psycopg2.pool.SimpleConnectionPool(
    minconn=5,
    maxconn=10,
    **DB_CONFIG
)
# --- llm 사용 config ---
sys_config = types.GenerateContentConfig(
    # system_instruction="쇼핑몰 검색 엔진용 키워드 추출기입니다. 설명 없이 쉼표로 구분된 키워드 10개만 출력하세요.",
    temperature=0.3, # 0.2보다 약간 높여야 키워드가 중복되지 않고 풍부해집니다.
    max_output_tokens=1000,
    safety_settings=[
        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
    ]
)


logger = logging.getLogger("uvicorn")
def extract_keywords_with_llm(user_query: str):
    # 1. Few-shot 프롬프트 (예시를 주면 10개를 꽉 채워서 답변합니다)
    prompt = f"""
    당신은 이커머스 쇼핑 키워드 매핑 전문가입니다.
    사용자의 검색어(특히 서술형, 주관적 표현, 롱테일)에 담긴 '의도 및 의미'를 파악하여,
    실제 쇼핑몰에서 사용되는 구체적인 상품, 상황, 대상 키워드를 10개로 변환하여 추출하세요.

    [키워드 추출 및 변환 규칙]
    1. 주관적/서술형 표현 치환: 검색어의 주관적 표현(낮은, 부드러운, 고정하는 등)을 날려버리지 말고,
    해당 속성을 나타내는 실제 쇼핑 매장용 '복합 명사'로 변환할 것.
    (예: 낮은 -> 틈새, 언더베드 / 부드러운 -> 애착, 극세사)
    2. 언어 규칙: 한국어(가-힣)와 아라비아 숫자(0-9) 혼용을 허용하며 쉼표로 구분할 것. (설명 금지)
    3. "추천", "상품", "꿀템", "인기" 등 마케팅 수식어 포함 금지할 것.
    4. 모든 키워드는 띄어쓰기 없이 하나의 단어 형태로 통일하고, 대표 하나만 남길 것 (중복 제거)
    (예: 침대 밑 수납장 -> 침대밑수납장)
    5. 상품 속성값은 필수로 포함하거나 유사 의미로 키워드 구성할 것.
    6. 실제 이커머스 검색엔진 및 카테고리에 존재하는(검색 가능한) 키워드만 허용할 것.
    7. 반드시 정확히 10개만 출력할 것.
    8. 실제 사용자가 많이 검색하는 키워드 순위를 반영하여 10개를 추출할 것.

    [유효성 필터 규칙 - 매우 중요]
    - 하나의 키워드는 한글(가-힣)과 숫자(0-9)로만 구성 (영문 및 특수기호 혼용 절대 불가)
    - "어주기", "하기", "되기", "용", "것" 등 불완전한 형태 제거
    - 의미 없는 부분 문자열 추출 절대 금지
    - 사람이 읽었을 때 자연스럽게 이해 가능한 명사/복합명사만 허용
    - [숫자+단위 결합 규칙]: 실제 고객의 검색 패턴을 반영하여, 아라비아 숫자(예: 3단, 3칸)와
    한글 수사(예: 삼단, 세칸) 표기법을 모두 실제 검색 가능한 유효한 키워드로 인정하여 다양하게 도출할 것.

    [예시]
    1. 주관적인 키워드
    입력: 낮은 리빙 박스
    출력: 언더베드수납장, 틈새수납장, 침대밑수납장, 슬라이딩리빙박스, 서랍장, 납작수납함, 플라스틱정리함, 옷정리함, 수납함, 리빙박스

    2. 롱테일 키워드
    입력: 침대 옆에 고정하는
    출력: 자바라거치대, 침대거치대, 스마트폰거치대, 태블릿거치대, 클립거치대, 침대협탁, 베드트레이, 무드등, 수면등, 침대선반

    3. 숫자와 단위의 현실적인 변환 (숫자+한글 혼용)
    입력: 미니 3단 서랍장
    출력: 미니3칸서랍장, 미니삼단서랍장, 세칸서랍장, 3단서랍소형, 미니3단서랍, 3단수납장, 책상위3단서랍, 탁상용3단서랍, 미니수납함3단, 3단정리함

    [실제 작업]
    입력: {user_query}
    출력:"""



    try:
        # 1. Gemini 호출 (contents를 리스트로 감싸서 전달)
        response = client.models.generate_content(
            model='models/gemma-3-1b-it', # 혹은 리스트에서 확인하신 정확한 ID
            contents=[prompt],
            config=sys_config
        )

        # 2. 텍스트 추출 (가장 안전한 방식)
        raw_result = ""
        try:
            raw_result = response.text
        except:
            if response.candidates and response.candidates[0].content.parts:
                raw_result = response.candidates[0].content.parts[0].text

        raw_result = (raw_result or "").strip()

        print("\n" + "🚀" * 20)
        print(f"입력 검색어: {user_query}")
        print(f"Gemini 원본 응답: [{raw_result}]")
        print("🚀" * 20 + "\n")

        if not raw_result:
            return [user_query]

        # 4. 정제 로직 (불필요한 문구 제거 및 분리)
        # '출력:', '결과:' 등이 섞여 나올 경우를 대비
        clean_text = re.sub(r'^(결과|출력|키워드)\s*[:：]\s*', '', raw_result, flags=re.IGNORECASE)

        # 쉼표로 분리 후 정제
        keywords = [k.strip() for k in clean_text.split(',')]

        # 특수문자 제거 및 한글/숫자만 유지 (2글자 이상)
        refined_keywords = []
        for k in keywords:
            # 한글, 숫자, 공백만 남기기
            clean_k = re.sub(r'[^가-힣0-9\s]', '', k).strip()
            if len(clean_k) >= 2 and clean_k != user_query:
                refined_keywords.append(clean_k)

        # 최종 10개 반환 (없으면 원본 쿼리라도 반환)
        return refined_keywords[:10] if refined_keywords else [user_query]

    except Exception as e:
        print(f"Gemini API Error: {e}")
        return [user_query]

# 사용자 질의에서 카테고리 추출
def extract_category_from_query(query: str):
    prompt = f"""
    사용자 검색어: "{query}"
    위 검색어에서 가장 적합한 상품 카테고리를 아래 리스트 중에서 하나만 선택해줘.
    카테고리: [국민득템, 뷰티/위생, 주방용품, 청소/욕실, 수납/정리, 문구/팬시, 인테리어/원예, 공구/디지털, 식품, 스포츠/레저/취미, 패션/잡화, 반려동물, 유아/완구, 시즌/시리즈]
    결과는 반드시 제시한 카테고리명으로만 답해줘. 다른 설명은 하지마.
    """

    # 1. Gemini 호출 (contents를 리스트로 감싸서 전달)
    response = client.models.generate_content(
        model='models/gemma-3-1b-it', # 혹은 리스트에서 확인하신 정확한 ID
        contents=[prompt],
        config=sys_config
    )

    # 2. 카테고리 추출
    raw_result = ""
    try:
        raw_result = response.text
    except:
        if response.candidates and response.candidates[0].content.parts:
            raw_result = response.candidates[0].content.parts[0].text

    raw_result = (raw_result or "").strip()

    print("\n" + "🚀" * 20)
    print(f"입력 검색어: {query}")
    print(f"Gemini 원본 응답: [{raw_result}]")
    print("🚀" * 20 + "\n")

    return raw_result

# --- 3. 벡터 유사도 검색 함수 (Numpy 기반) ---
def embed_search(query_vector, target_category, limit=50):
    conn = sqlite3.connect('search_data.db')
    cur = conn.cursor()
    # SQLite에서 전체 데이터 로드 (데이터가 아주 많지 않을 때 효율적)
    cur.execute("SELECT pd_no, pd_nm, item_vector, exh_ctgr FROM embedding_test")
    rows = cur.fetchall()
    conn.close()

    results = []
    q_vec = np.array(query_vector)

    for row in rows:
        pd_no, pd_nm, v_blob, exh_ctgr = row
        if v_blob:
            # BLOB 데이터를 다시 float32 배열로 복원
            i_vec = np.frombuffer(v_blob, dtype=np.float32)

            # 코사인 유사도 계산
            norm_q = np.linalg.norm(q_vec)
            norm_i = np.linalg.norm(i_vec)
            if norm_q > 0 and norm_i > 0:
                similarity = np.dot(q_vec, i_vec) / (norm_q * norm_i)
            else:
                similarity = 0.0

            # 가중치 로직 적용 -- 분석한 카테고리와 상품의 카테고리가 일치하면 가산점
            exh_large_ctgr_nm = exh_ctgr.split('>')[0]
            if target_category and (target_category in str(exh_large_ctgr_nm)):
                similarity += 0.2
            results.append({
                "pd_no": pd_no,
                "pd_nm": pd_nm,
                "large_category": exh_large_ctgr_nm,
                "similarity": float(similarity),
            })

    # 유사도 높은 순으로 정렬 후 상위 N개 반환
    results.sort(key=lambda x: x['similarity'], reverse=True)
    return results[:limit]


@app.get("/gemma-search")
def gemma_search(query: str):
    conn = None
    start_time = time.time()
    try:
        # 1. 키워드 추출
        kw_start = time.time()
        keywords = extract_keywords_with_llm(query)
        target_category = extract_category_from_query(query)
        kw_runtime = round(time.time() - kw_start, 3)

        # 2. 벡터화
        keyword_sentence = ", ".join(keywords)
        query_vector = embed_model.encode(keyword_sentence).tolist()

        # 3. SQLite 파일에서 유사도 검색 수행
        embed_search_start = time.time()
        search_results = embed_search(query_vector, target_category)
        embed_search_end = time.time()
        # # 3. DB 검색 (Pool 활용)
        # conn = db_pool.getconn() # 풀에서 대기 없이 가져옴
        # with conn.cursor(cursor_factory=RealDictCursor) as cur:
        #     # 인덱스 스캔 강제 설정 (세션별 적용)
        #     cur.execute("SET enable_seqscan = off;")

        #     search_sql = """
        #     SELECT pd_no, pd_nm, (1 - (item_vector <=> %s::vector)) AS similarity
        #     FROM embedding_test
        #     ORDER BY item_vector <=> %s::vector
        #     LIMIT 10;
        #     """
        #     cur.execute(search_sql, (query_vector, query_vector))
        #     results = cur.fetchall()

        total_runtime = round(time.time() - start_time, 3)
        return {
            "input_query": query,
            "llm_recoommed_keywords": keywords,
            "kw_runtime": kw_runtime,
            "embed_search_runtime": round(embed_search_end - embed_search_start, 3),
            "total_runtime": total_runtime,
            "target_category": target_category,
            "search_results": search_results
        }

    except Exception as e:
        return {"error": str(e)}
    finally:
        # [중요] 사용한 커넥션은 반드시 풀에 반납해야 합니다.
        if conn:
            db_pool.putconn(conn)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)