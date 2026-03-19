from mcp.server.fastmcp import FastMCP
import requests


mcp = FastMCP("--- DaisoMall Search Mcp ---")


@mcp.tool()
def prod_search(query: str) -> list:
  """ DaisoMall Prod Search (다이소몰 상품 통합검색)
    query: 검색할 상품명 또는 키워드 또는 품번
  """
  url = f"https://prdm.daisomall.co.kr/ssn/search/SearchGoods?searchTerm={query}"

  try:
    response = requests.get(url)
    response.raise_for_status() # 200 아니면 예외발생.

    data = response.json()
    result_set = data.get("resultSet", {})
    results = result_set.get("result", [])

    if len(results) > 0:
      product_list = results[1].get("resultDocuments", [])
      return product_list
    else:
      return []
  except  Exception as e:
    print("에러났음!!  -- {e}")
  return



@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
  """ 하잉! 하고 인사함!! """
  return f"하이, {name} !!"


if __name__ == "__main__":
  mcp.run()