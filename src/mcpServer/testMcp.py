from mcp.server.fastmcp import FastMCP

mcp = FastMCP("chococho1 test MCP")


@mcp.tool()
def add(a: int, b: int) -> int:
  """ Add two numbers """
  return a+b

@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
  """ 하잉! 하고 인사함!! """
  return f"하이, {name} !!"


if __name__ == "__main__":
  mcp.run()