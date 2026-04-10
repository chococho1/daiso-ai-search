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
    당신은 쇼핑 키워드 추출기입니다. 사용자의 검색어와 연관된 상품, 상황, 대상 키워드를 설명없이, 반드시 10개만 추출하세요.

    키워드 추출 규칙:

    - 반드시 한국어로만 작성
    - 쉼표로 구분
    - 중복 제거
    - 설명 금지
    - "추천", "상품", "꿀템", "인기" 포함 금지
    - 명사 형태만 허용
    - 한국어 사전에 존재하는 단어만 사용
    - 모든 키워드는 띄어쓰기 없이 하나의 단어 형태로 통일하라



    [유효성 필터 규칙 - 매우 중요]

    다음 조건을 만족하지 않는 키워드는 절대 포함하지 마세요:

    1. 하나의 키워드는 오직 한글(가-힣)로만 구성되어야 한다
    → 한글 이외의 문자(영문, 숫자, 아랍어 등)가 포함되면 제거

    2. 의미가 완전한 명사만 허용한다
    → "어주기", "하기", "되기", "용", "것" 등 불완전한 형태는 제거

    3. 입력값의 일부를 잘라 만든 비정상 단어 금지
    → 의미 없는 부분 문자열 추출 금지

    4. 사람이 읽었을 때 자연스럽게 이해 가능한 단어만 허용

    5. 실제 쇼핑 검색에서 사용 가능한 키워드만 허용

    6. 최종 출력 전에 전체 키워드를 다시 검토하여
    중복 및 유사 키워드를 제거하고 반드시 10개만 출력하라

    7. 의미가 동일하거나 유사한 키워드는 하나만 남겨라
    (예: "강아지간식" / "강아지 간식" → 하나만 선택)

    [예시]
    입력: 강아지 간식
    출력: 개껌, 져키, 수제간식, 노령견, 칭찬용, 대용량, 덴탈껌, 연어트릿, 고단백, 강아지간식

    입력: 꽃구경 갈 때 쓰기 좋은 상품
    출력: 담요, 돗자리, 피크닉, 바구니, 피크닉가방, 도시락, 양산, 봄나들이, 벚꽃, 꽃놀이

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