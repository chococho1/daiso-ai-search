import os
from fastapi import FastAPI
import psycopg2
from psycopg2.extras import RealDictCursor
from sentence_transformers import SentenceTransformer

# 로그 메시지 숨기기 (선택 사항)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = FastAPI()

# 1. 모델 로드 (서버 시작 시 한 번만 로드하여 메모리에 상주)
print("검색 모델 로딩 중...")
model = SentenceTransformer('dragonkue/BGE-m3-ko')

# 2. DB 설정
DB_CONFIG = {
    "host": "localhost",
    "database": "postgres",
    "user": "postgres",
    "password": "chdnjs6918^^",
    "port": 5432
}

def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)

@app.get("/search")
def search_products(query: str, limit: int = 5):
    """
    사용자의 검색어를 벡터로 변환하여 유사한 상품을 검색합니다.
    """
    try:
        # 1. 사용자의 검색어 임베딩
        query_vector = model.encode(query).tolist()

        # 2. DB 연결 및 검색
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # pgvector 코사인 유사도 검색 쿼리 (<=> 는 코사인 거리)
                # 1 - 거리를 하면 유사도 점수(0~1)가 나옵니다.
                search_query = """
                SELECT pd_no, pd_nm, 1 - (item_vector <=> %s::vector) AS similarity
                FROM embedding_test
                ORDER BY similarity DESC
                LIMIT %s;
                """
                cur.execute(search_query, (query_vector, limit))
                results = cur.fetchall()

        return {
            "query": query,
            "results": results
        }

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    # 서버 실행 (기본 포트 8000)
    uvicorn.run(app, host="0.0.0.0", port=8000)