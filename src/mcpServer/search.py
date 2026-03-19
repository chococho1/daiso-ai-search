from mcp.server.fastmcp import FastMCP
from prompt.search_prompt import TOOL_DESCRIPTIONS
import requests


mcp = FastMCP("--- DaisoMall Search Mcp ---")


@mcp.tool()
def prod_search(query: str) -> list:
  url = f"https://prdm.daisomall.co.kr/ssn/search/SearchGoods?searchTerm={query}"

  try:
    response = requests.get(url)
    response.raise_for_status() # 200 아니면 예외발생.

    data = response.json()
    result_set = data.get("resultSet", {})
    results = result_set.get("result", [])

    if len(results) > 0:
      product_list = results[1].get("resultDocuments", [])
      final_results = []
      for product in product_list:
        pdNo = product.get("pdNo")
        exhPdNm = product.get("exhPdNm")
        pdPrc = product.get("pdPrc")

        # 상품상세 url
        prod_detail_url = f"https://www.daisomall.co.kr/pd/pdr/SCR_PDR_0001?pdNo={pdNo}&recmYn=N"

        #필요한 정보만 넣기
        final_results.append({
          "상품명" : exhPdNm,
          "품번" : pdNo,
          "가격" : f"{pdPrc}원",
          "링크" : f"[상품상세]({prod_detail_url})"
        })
      return final_results
    else:
      return []
  except  Exception as e:
    print("에러났음!!  -- {e}")
  return

prod_search.__doc__ = TOOL_DESCRIPTIONS["prod_search"]

@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
  """ 하잉! 하고 인사함!! """
  return f"하이, {name} !!"


if __name__ == "__main__":
  mcp.run()