import os
import re
import requests
import json
import torch
import time
import psycopg2
from fastapi import FastAPI
from transformers import pipeline, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from psycopg2.extras import RealDictCursor
from util.get_db_config import get_db_config
# 이 파일 실행 후
# 브라우저에서 http://localhost:8000/search?query=(자연어)  로 테스트.

# 1. 모델 로드 (Gemma 2 - 2b 또는 9b 모델 사용 가능)
# 메모리 사양에 따라 'google/gemma-2-2b-it'를 추천합니다.
print("gemma2:2b 모델 로딩 중... (시간이 다소 걸릴 수 있습니다)")
# 2. 임베딩 모델 로드
print("BGE-M3 임베딩 모델 로딩 중...")
# embed_model = SentenceTransformer('jhgan/ko-sbert-sts')
embed_model = SentenceTransformer('dragonkue/BGE-m3-ko', device='cpu') # 추천받은 한국어 특화 임베딩 모델

app = FastAPI()
DB_CONFIG = get_db_config("localDB.properties")

def extract_keywords_with_ollama(user_query: str):
    """Ollama를 통해 llm 에게 키워드 추출 요청 및 정규식 보정"""
    url = "http://localhost:11434/api/generate"
    prompt = f"""Task: Extract 10 shopping keywords.
    Example: 캠핑 용품 추천 -> 텐트, 침낭, 랜턴, 취사도구, 캠핑의자, 매트, 아이스박스, 타프, 버너, 망치
    Input: {user_query}
    Keywords:"""
    system_instruction = "Role: Product Keyword Extractor. Rule 1: Print only comma-separated keywords. Rule 2: Do not cut words. Rule 3: No explanation. Rule 4: Print only Korean. However, exclude the keyword '추천','상품'. Role 5: Do not extract English and emoticons."
    payload = {
        "model": "gemma2:2b",
        "prompt": prompt,
        "system": system_instruction,
        "stream": False,
        "keep_alive": "5m",
        "options": {
          "num_thread": 10,
          "num_predict": 40,
          "temperature": 0.4,
          "top_k": 1,
        }
    }

    try:
        response = requests.post(url, json=payload, timeout=10)
        raw_result = response.json().get("response", "").strip()

        # 1. 정규 표현식으로 보정
        # [^,\n]+ : 쉼표나 줄바꿈이 아닌 문자들이 연속된 덩어리를 찾습니다.
        # 이를 통해 '1. 키워드', '- 키워드', '키워드\n' 등 다양한 형태를 대응합니다.
        keywords = re.findall(r'[^,\n]+', raw_result)

        # 2. 전처리: 불필요한 공백, 번호(1. ), 특수문자(- ) 제거
        refined_keywords = []
        for k in keywords:
            clean_k = re.sub(r'^[\d\s\-\.\*]+', '', k).strip() # 앞쪽 숫자, 점, 대시 제거
            if len(clean_k) > 1: # '사', '의' 처럼 한 글자만 남은(잘린) 키워드는 제외
                refined_keywords.append(clean_k)

        # 3. 최대 10개까지만 반환
        return refined_keywords[:10]

    except Exception as e:
        print(f"Ollama Error: {e}")
        return [user_query] # 에러 발생 시 입력값 그대로 사용


@app.get("/gemma-search")
def gemma_search(query: str):
    start_time = time.time()
    try:
        ollama_start_time = time.time()
        # STEP 1: Gemma 2로 키워드 추출
        keywords = extract_keywords_with_ollama(query)
        ollama_runtime = round(time.time() - ollama_start_time, 3)
        keyword_sentence = ", ".join(keywords)

        # STEP 2: 추출된 키워드 뭉치를 벡터화
        query_vector = embed_model.encode(keyword_sentence).tolist()

        # STEP 3: pgvector 검색
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                search_sql = """
                SELECT pd_no, pd_nm, 1 - (item_vector <=> %s::vector) AS similarity
                FROM embedding_test
                WHERE item_vector is not null
                ORDER BY similarity DESC
                LIMIT 10;
                """
                cur.execute(search_sql, (query_vector,))
                results = cur.fetchall()

        total_runtime = round(time.time() - start_time, 3) # 소수점 3자리까지. (밀리초 단위)
        return {
            "input_query": query,
            "llm_recoommed_keywords": keywords,
            "ollama_runtime": f"{ollama_runtime} s",
            "total_runtime": f"{total_runtime} s",
            "search_results": results
        }

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)