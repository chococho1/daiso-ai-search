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
from util.get_db_config import get_db_config
from util.constants import system_instruction

# --- 1. 모델 및 리소스 설정 ---
# 10코어 사양 최적화: 임베딩 모델에 4개 코어 할당
torch.set_num_threads(4)

print("BGE-M3 임베딩 모델 로딩 중...")
embed_model = SentenceTransformer('dragonkue/BGE-m3-ko', device='cpu')

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
    url = "http://localhost:11434/api/generate"
    prompt = f"""
        [쇼핑 키워드 추출기]
        규칙: 주어진 문장에 어울리는 연관 키워드 10개만 쉼표로 구분하여 출력한다. 본문 내용은 결과에 절대 포함하지 않는다.
        [예시]
        입력: 고양이 간식 추천
        출력: 츄르, 고양이캔, 연어트릿, 고양이우유, 북어트릿, 칭찬용, 노령묘간식, 수분보충, 영양공급, 기호성테스트

        ---
        입력: {user_query}
        출력:"""

    payload = {
        "model": "gemma2:2b",
        "prompt": prompt,
        "system": system_instruction,
        "stream": False,
        "keep_alive": "0",
        "options": {
            "num_thread": 6,      # [핵심] 10에서 6으로 하향 (임베딩 자원 확보)
            "num_predict": 30,    # [핵심] 더 짧게 끊어서 생성 시간 단축
            "temperature": 0,
            "top_k": 1,
            "repeat_penalty": 1.2,
            "stop": ["대상문장:", "입력:", "\n", user_query] # [추가] 질문이 나오는 순간 중단
        }
    }

    try:
        response = requests.post(url, json=payload, timeout=2)
        response_json = response.json()
        raw_result = response_json.get("response", "").strip()
        print(raw_result)
        # [디버깅] 터미널에 확실히 찍히도록 강제 출력
        print("\n" + "="*50)
        print(f"DEBUG - Raw LLM Response: [{raw_result}]")
        print("="*50 + "\n")
        # 만약 아무것도 안 왔다면?
        if not raw_result:
            print("⚠️ Ollama가 빈 응답을 보냈습니다. 모델 로드 상태를 확인하세요.")
            return [user_query]
        # 질문 도려내기 로직 (질문 따라하기 방지용)
        if user_query in raw_result:
            raw_result = raw_result.split(user_query)[-1].strip()

        keywords = re.findall(r'[^,\n]+', raw_result)
        refined_keywords = [re.sub(r'[^가-힣0-9\s]', '', k).strip() for k in keywords if len(k.strip()) > 1]
        return refined_keywords[:10]
    except Exception as e:
        print(f"Ollama Error: {e}")
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