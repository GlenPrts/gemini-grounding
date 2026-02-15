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
    利用 Google 搜索 (Gemini Grounding) 获取带有来源引用的实时事实信息。

    适用于：
    - 时事新闻：新闻、发布日期、体育比分、近期动态。
    - 事实核查：验证特定声明、统计数据或历史细节。
    - 外部知识：可能超出你训练截止日期或专有文档的信息。

    Args:
        query: 搜索关键词。建议将对话式问题转换为关键词查询以获得更好结果 (例如: "Python 最新版本 发布日期" 而非 "Python的最新版本是多少")。
        model: 指定 Gemini 模型 (默认: gemini-2.5-flash)。
        retry_count: 失败重试次数 (默认: 3)。
        retry_delay: 重试等待时间(秒) (默认: 5.0)。
        search_delay_min: 搜索前最小随机延迟(秒) (默认: 0.0)。
        search_delay_max: 搜索前最大随机延迟(秒) (默认: 0.0)。
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
