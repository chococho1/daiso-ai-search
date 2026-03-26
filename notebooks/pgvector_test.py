from sentence_transformers import SentenceTransformer
import psycopg2

# 1. 모델 로드 (한국어 특화 무료 모델)
model = SentenceTransformer('dragonkue/BGE-m3-ko')

# 2. DB 연결
conn = psycopg2.connect("host=localhost dbname=postgres user=postgres password=chdnjs6918^^")
cur = conn.cursor()

# 3. 벡터값이 없는 상품 가져오기
cur.execute("select pd_no, pd_nm from EMBEDDING_TEST where item_vector is null")
rows = cur.fetchall()

for pd_no, pd_nm in rows:
    # 상품명을 1024차원 벡터로 변환
    embedding = model.encode(pd_nm).tolist()

    # DB 업데이트
    cur.execute("UPDATE EMBEDDING_TEST SET item_vector = %s WHERE pd_no = %s", (embedding, pd_no))

conn.commit()
cur.close()
conn.close()
print("☑️모든 상품의 벡터화가 완료되었습니다!")