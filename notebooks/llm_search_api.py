import os
import re
import requests
import json
import torch
import time
import psycopg2
import logging
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
client = genai.Client(api_key="AIzaSyB_9bT_R8cTmYMKmhGiOZVeNcfsobT7Cew")
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
logger = logging.getLogger("uvicorn")
def extract_keywords_with_ollama(user_query: str):
    # 1. Few-shot 프롬프트 (예시를 주면 10개를 꽉 채워서 답변합니다)
    prompt = f"""
    당신은 쇼핑 키워드 추출기입니다. 사용자의 검색어와 연관된 상품, 상황, 대상 키워드를 설명없이, 반드시 10개만 추출하세요.

    [예시]
    입력: 강아지 간식
    출력: 개껌, 져키, 수제간식, 노령견, 칭찬용, 대용량, 덴탈껌, 연어트릿, 고단백, 강아지간식추천

    [실제 작업]
    입력: {user_query}
    출력:"""

    sys_config = types.GenerateContentConfig(
        # system_instruction="쇼핑몰 검색 엔진용 키워드 추출기입니다. 설명 없이 쉼표로 구분된 키워드 10개만 출력하세요.",
        temperature=0.1, # 0.2보다 약간 높여야 키워드가 중복되지 않고 풍부해집니다.
        max_output_tokens=1000,
        safety_settings=[
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
        ]
    )

    try:
        # 2. Gemini 호출 (contents를 리스트로 감싸서 전달)
        response = client.models.generate_content(
            model='models/gemma-3-1b-it', # 혹은 리스트에서 확인하신 정확한 ID
            contents=[prompt],
            config=sys_config
        )

        # 3. 텍스트 추출 (가장 안전한 방식)
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

@app.get("/gemma-search")
def gemma_search(query: str):
    conn = None
    start_time = time.time()
    try:
        # 1. Ollama 키워드 추출
        ollama_start = time.time()
        keywords = extract_keywords_with_ollama(query)
        ollama_runtime = round(time.time() - ollama_start, 3)

        # 2. 벡터화
        keyword_sentence = ", ".join(keywords)
        query_vector = embed_model.encode(keyword_sentence).tolist()

        # 3. DB 검색 (Pool 활용)
        conn = db_pool.getconn() # 풀에서 대기 없이 가져옴
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # 인덱스 스캔 강제 설정 (세션별 적용)
            cur.execute("SET enable_seqscan = off;")

            search_sql = """
            SELECT pd_no, pd_nm, (1 - (item_vector <=> %s::vector)) AS similarity
            FROM embedding_test
            ORDER BY item_vector <=> %s::vector
            LIMIT 10;
            """
            cur.execute(search_sql, (query_vector, query_vector))
            results = cur.fetchall()

        total_runtime = round(time.time() - start_time, 3)
        return {
            "input_query": query,
            "llm_recoommed_keywords": keywords,
            "ollama_runtime": f"{ollama_runtime} s",
            "total_runtime": f"{total_runtime} s",
            "search_results": results
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