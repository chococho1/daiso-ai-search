import psycopg2
import json

conn = psycopg2.connect(
    host="127.0.0.1",
    port="40004",
    dbname="diapostgres",
    user="querypie",
    password="1e33f347d1443f82"
)
cursor = conn.cursor()
with open("../data/idx_srch_goods.sql", "r", encoding="utf-8") as file:
    sql_script = file.read()

cursor.execute(sql_script)


rows = cursor.fetchall()


with open("../data/train.jsonl", "w", encoding="utf-8") as f:
    for row in rows:
        pd_no,exh_pd_nm,pd_prc,new_pd_yn,group_brand,group_pd_type,group_takbae_type,pkup_or_psbl_yn,pds_or_psbl_yn,mass_or_psbl_yn,fdrm_or_psbl_yn,quick_or_psbl_yn,only_quick_yn,avg_stsc_val,revw_cnt,pdimgurl,only_onl_yn,onl_stck_qy,soldoutyn,inct = row
        prompt = (
            f"전시상품명: {exh_pd_nm}\n"
            f"가격: {pd_prc}\n"
            f"신상품 여부: {new_pd_yn}\n"
            f"브랜드코드>브랜드명: {group_brand}\n"
            f"상품 유형: {group_pd_type}\n"
            f"택배 가능 여부 그룹핑: {group_takbae_type}\n"
            f"매장픽업 가능 여부: {pkup_or_psbl_yn}\n"
            f"택배배송 가능 여부: {pds_or_psbl_yn}\n"
            f"대량 구매 가능 여부: {mass_or_psbl_yn}\n"
            f"오늘 배송 가능 여부: {quick_or_psbl_yn}\n"
            f"오늘배송여부: {only_quick_yn}\n"
            f"리뷰 평점: {avg_stsc_val}\n"
            f"리뷰 수: {revw_cnt}\n"
            f"이미지 URL: {pdimgurl}\n"
            f"온라인 전용 상품 여부: {only_onl_yn}\n"
            f"온라인 재고 수량: {onl_stck_qy}\n"
            f"품절 여부: {soldoutyn}\n"
        )
        data = {
            "prompt": prompt,
            "completion": f"전시상품명: {exh_pd_nm}\n가격: {pd_prc}\n신상품 여부: {new_pd_yn}\n브랜드코드>브랜드명: {group_brand}\n상품 유형: {group_pd_type}\n택배 가능 여부: {group_takbae_type}\n픽업 가능 여부: {pkup_or_psbl_yn}\n배송 가능 여부: {pds_or_psbl_yn}\n대량 구매 가능 여부: {mass_or_psbl_yn}\n냉동 배송 가능 여부: {fdrm_or_psbl_yn}\n오늘 배송 가능 여부: {quick_or_psbl_yn}\n오늘배송여부: {only_quick_yn}\n리뷰 평점: {avg_stsc_val}\n리뷰 수: {revw_cnt}\n이미지 URL: {pdimgurl}\n온라인 전용 상품 여부: {only_onl_yn}\n온라인 재고 수량: {onl_stck_qy}\n품절 여부: {soldoutyn}"
        }
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

cursor.close()
conn.close()
print("☑️   train.jsonl 파일로 저장 완료오~!")