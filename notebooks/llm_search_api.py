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
from util.constants import system_instruction
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
    prompt = f"""
    입력: 고양이 간식 추천
    출력: 츄르, 캣잎, 닭가슴살, 동결건조간식, 고양이캔, 연어트릿, 스틱간식, 필레, 고양이우유, 북어트릿

    입력: {user_query}
    출력:"""
    payload = {
        "model": "gemma2:2b",
        "prompt": prompt,
        "system": system_instruction,
        "stream": False,
        "keep_alive": "5m",
        "options": {
          "num_thread": 10,
          "num_predict": 40,
          "temperature": 0.2,
          "top_k": 1,
        }
    }

    try:
        response = requests.post(url, json=payload, timeout=10)
        raw_result = response.json().get("response", "").strip()
        # [추가] 모델이 유저의 질문을 그대로 복사했는지 체크하여 강제 삭제
        if user_query in raw_result:
            raw_result = raw_result.replace(user_query, "").strip()
        # 1. 정규 표현식으로 보정
        # [^,\n]+ : 쉼표나 줄바꿈이 아닌 문자들이 연속된 덩어리를 찾습니다.
        # 이를 통해 '1. 키워드', '- 키워드', '키워드\n' 등 다양한 형태를 대응합니다.
        keywords = re.findall(r'[^,\n]+', raw_result)

        # 2. 전처리: 불필요한 공백, 번호(1. ), 특수문자(- ) 제거
        refined_keywords = []
        seen_keywords = set()  # 중복 체크를 위한 집합

        # 사용자가 입력한 검색어도 중복으로 간주하여 제외 (확장 키워드만 보여주기 위함)
        input_query_clean = re.sub(r'[^가-힣0-9\s]', '', user_query).replace("추천", "").strip()
        seen_keywords.add(input_query_clean)
        for k in keywords:
            clean_k = re.sub(r'^[\d\s\-\.\*]+', '', k).strip() # 앞쪽 숫자, 점, 대시 제거
            clean_k = re.sub(r'[^가-힣0-9\s]', '', clean_k).strip() # 한글과 숫자만 남기고 나머지는 모두 제거
            #  중복 제거 및 유효성 검사
            # - 이미 추가된 단어가 아니고
            # - 2글자 이상이며
            # - 입력한 검색어와 완전히 같지 않은 경우만 추가
            if clean_k not in seen_keywords and len(clean_k) > 1:
                refined_keywords.append(clean_k)
                seen_keywords.add(clean_k) # 중복 목록에 추가

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