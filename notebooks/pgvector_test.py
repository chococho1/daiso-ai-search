import os
import psycopg2
from util.get_db_config import get_db_config
from sentence_transformers import SentenceTransformer
import time

# 1. DB 연결 설정
DB_CONFIG = get_db_config("localDB.properties")
BATCH_SIZE = 100

print("모델 로딩 중...")
model = SentenceTransformer('dragonkue/BGE-m3-ko')

def batch_embedding_update():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # item_vector가 비어있는 데이터 개수 확인
    cur.execute("SELECT count(*) FROM embedding_test WHERE item_vector IS NULL")
    total_count = cur.fetchone()[0]
    print(f"남은 데이터: 총 {total_count}건")

    processed = 0
    start_time = time.time()

    try:
        while True:
            # 1. 배치 단위 데이터 가져오기 (상품명, 대>중>소카테고리, 가격)
            cur.execute("""
                SELECT pd_no, pd_nm, exh_ctgr, pd_prc
                FROM embedding_test
                WHERE item_vector IS NULL
                LIMIT %s
            """, (BATCH_SIZE,))
            rows = cur.fetchall()

            if not rows:
                break

            pd_nos = []
            combined_texts = []

            for row in rows:
                pd_no, pd_nm, exh_ctgr, pd_prc = row

                # 정보를 하나로 합칩니다. (예: [주방용품] 세라믹 칼 가격: 15000원)
                combined_text = f"[{exh_ctgr}] {pd_nm} 가격:{pd_prc}원"

                pd_nos.append(pd_no)
                combined_texts.append(combined_text)

            # 2. 합쳐진 텍스트를 통째로 임베딩
            embeddings = model.encode(combined_texts).tolist()

            # 3. 배치 업데이트 실행
            for i in range(len(pd_nos)):
                cur.execute(
                    "UPDATE embedding_test SET item_vector = %s WHERE pd_no = %s",
                    (embeddings[i], pd_nos[i])
                )

            # 4. 배치마다 커밋
            conn.commit()

            processed += len(rows)
            elapsed = time.time() - start_time
            print(f"[{processed}/{total_count}] 완료... (({elapsed:.2f} s 소요됨)")

    except Exception as e:
        print(f"에러 발생: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()
        print("모든 작업이 종료되었습니다.")

if __name__ == "__main__":
    batch_embedding_update()