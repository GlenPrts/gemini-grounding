import os
import sys
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
env_path = os.path.join(project_root, ".env")
if os.path.exists(env_path):
    load_dotenv(env_path)

sys.path.append(current_dir)

try:
    from search import search
except ImportError:
    from .search import search

mcp = FastMCP("gemini-grounding")


@mcp.tool()
def google_search(
    query: str,
    model: str = "gemini-2.5-flash",
    retry_count: int = 3,
    retry_delay: float = 5.0,
    search_delay_min: float = 0.0,
    search_delay_max: float = 0.0,
) -> str:
    """
    Perform a Google search using Gemini Grounding.
    """
    try:
        result = search(
            query,
            model=model,
            retry_count=retry_count,
            retry_delay=retry_delay,
            search_delay_min=search_delay_min,
            search_delay_max=search_delay_max,
        )

        output = result["text"]
        if result["sources"]:
            output += "\n\n## Sources\n"
            for src in result["sources"]:
                output += f"{src['id']}. [{src['title']}]({src['url']})\n"

        return output
    except Exception as e:
        return f"Error performing search: {str(e)}"


if __name__ == "__main__":
    mcp.run()
