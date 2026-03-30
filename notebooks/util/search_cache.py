from diskcache import Cache
from llm_search_api import extract_keywords_with_llm
import os

# 현재 폴더에 'search_cache' 폴더 생성 후 데이터 저장
cache = Cache("./search_cache")

def get_smart_keywords(query):
    # 캐시에 있으면 가져오고, 없으면 함수를 실행해 저장함 (데코레이터처럼 동작)
    if query in cache:
        return cache[query]

    # 캐시 없을 때만 실행
    result = extract_keywords_with_llm(query)
    cache.set(query, result, expire=86400) # 24시간 동안 유지
    return result