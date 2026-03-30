import psycopg2
import sqlite3
import json
import numpy as np
from get_db_config import get_db_config

def export_to_sqlite():
    # 1. PostgreSQL 연결 설정 (localDB.properties에서 로드)
    pg_config = get_db_config("localDB.properties")

    try:
        # PostgreSQL 연결
        pg_conn = psycopg2.connect(**pg_config)
        pg_cur = pg_conn.cursor()

        # 2. 데이터 조회 (public.vector 타입 포함)
        query = "SELECT pd_no, pd_nm, item_vector, pd_prc, exh_ctgr FROM embedding_test"
        pg_cur.execute(query)
        rows = pg_cur.fetchall()

        # 3. SQLite 연결 및 테이블 생성
        sl_conn = sqlite3.connect('search_data.db')
        sl_cur = sl_conn.cursor()

        # 기존 테이블 삭제 후 재생성 (BLOB 타입으로 벡터 저장)
        sl_cur.execute("DROP TABLE IF EXISTS embedding_test")
        sl_cur.execute("""
            CREATE TABLE embedding_test (
                pd_no TEXT,
                pd_nm TEXT,
                item_vector BLOB,
                pd_prc REAL,
                exh_ctgr TEXT
            )
        """)

        # 4. 데이터 삽입 루프
        for row in rows:
            pd_no = str(row[0])
            pd_nm = row[1]
            # numeric 타입을 위해 float으로 변환
            pd_prc = float(row[3]) if row[3] is not None else 0.0
            exh_ctgr = row[4]

            vector_data = row[2]  # PostgreSQL의 public.vector 데이터
            final_blob = None

            if vector_data is not None:
                try:
                    # [pgvector 핵심 처리]
                    # pgvector 객체는 list()로 감싸면 순수 숫자 리스트로 추출됩니다.
                    if isinstance(vector_data, str):
                        v_list = json.loads(vector_data)
                    else:
                        # pgvector 전용 객체 혹은 numpy array 대응
                        v_list = list(vector_data)

                    # float32 타입의 numpy 배열로 만든 뒤 bytes로 변환 (SQLite BLOB 최적화)
                    final_blob = np.array(v_list, dtype=np.float32).tobytes()
                except Exception as ve:
                    print(f"ID {pd_no} 벡터 변환 실패: {ve}")

            # SQLite 실행 (바이너리 데이터를 직접 전달)
            sl_cur.execute(
                "INSERT INTO embedding_test (pd_no, pd_nm, item_vector, pd_prc, exh_ctgr) VALUES (?, ?, ?, ?, ?)",
                (pd_no, pd_nm, final_blob, pd_prc, exh_ctgr)
            )

        sl_conn.commit()
        print(f"✅ 총 {len(rows)}건의 데이터가 'search_data.db'로 성공적으로 변환되었습니다.")

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
    finally:
        if 'pg_conn' in locals(): pg_conn.close()
        if 'sl_conn' in locals(): sl_conn.close()

if __name__ == "__main__":
    export_to_sqlite()