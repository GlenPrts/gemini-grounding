import asyncio
import os
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def run():
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    server_script = os.path.join(current_dir, "mcp_server.py")

    print(f"Starting MCP server: {server_script}")

    env = os.environ.copy()

    server_params = StdioServerParameters(
        command="uv", args=["run", server_script], env=env
    )

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                print("\n=== Listing Tools ===")
                tools = await session.list_tools()
                for tool in tools.tools:
                    print(f"Tool: {tool.name}")
                    print(f"Description: {tool.description}")
                    print("---")

                query = "Python 的最新版本是多少"
                print(f"\n=== Calling Tool: google_search (query='{query}') ===")

                result = await session.call_tool(
                    "google_search", arguments={"query": query}
                )

                print("\n=== Result ===")
                for content in result.content:
                    if content.type == "text":
                        print(content.text)
                    else:
                        print(f"[{content.type} content]: {content}")

    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(run())
