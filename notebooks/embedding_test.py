from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("dragonkue/BGE-m3-ko")  #문장 임베딩

sentences1 = "화장실에 놓을 물건 추천"

sentences2 = [
    "일본제 키링 디스플레이 케이스",
    "일본제 포토 카드 디스플레이 케이스",
    "일본제 원터치 바늘 6개입",
    "일본제 모노톤 체크리스트 스티커 2매입",
    "일본제 마그넷 스윙 케이스 직사각 화이트",
    "일본제 냉장고 정리 트레이 350 ml 캔 음료용",
    "클리어 수납함 25.2X14.9X14.3cm",
    "진공 저장 용기 1480 ml",
    "고주파 냉감 반달 쿠션 45 X 25 cm 베이지",
    "템포롤화장지(24 m)",
    "뽑아쓰는키친타월(150매입)",
    "쿤달 리치퍼퓸 디퓨저 250 ml 퓨어솝 향"
]


embeddings1 = model.encode(sentences1)
embeddings2 = model.encode(sentences2)

similarities = model.similarity(embeddings1,embeddings2)
print(similarities.shape)

result = []
for i in range(len(embeddings2)):
    similarities = cosine_similarity([embeddings1], [embeddings2[i]])[0][0]
    result.append((sentences2[i], similarities))

result = sorted(result, key=lambda x: x[1], reverse=True)   ## 오름차순
print(result)