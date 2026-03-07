import os
import re
import sys
import requests
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
env_path = os.path.join(project_root, ".env")
if os.path.exists(env_path):
    load_dotenv(env_path)

sys.path.append(current_dir)

try:
    from search import resolve_search_options, search
except ImportError:
    from .search import resolve_search_options, search

mcp = FastMCP("gemini-grounding")


@mcp.tool()
def google_search(
    query: str,
    model: str | None = None,
    retry_count: int | None = None,
    retry_delay: float | None = None,
    search_delay_min: float | None = None,
    search_delay_max: float | None = None,
    retry_until_success: bool | None = None,
) -> str:
    """
    利用 Google 搜索 (Gemini Grounding) 获取带有来源引用的实时事实信息。

    适用于：
    - 时事新闻：新闻、发布日期、体育比分、近期动态。
    - 事实核查：验证特定声明、统计数据或历史细节。
    - 外部知识：可能超出你训练截止日期或专有文档的信息。

    为了获得最佳搜索结果，请务必拆分并优化搜索语句。建议针对单一特定的信息点进行搜索，宁可进行多次精准搜索，也不要尝试一次性搜索过多复杂内容。

    在 MCP 集成场景中，建议默认将 `GEMINI_RETRY_UNTIL_SUCCESS=true` 传入运行环境，
    以便在遇到限流、瞬时网络错误或偶发空结果时持续重试直到拿到非空结果。

    Args:
        query: 搜索关键词。建议将对话式问题转换为关键词查询以获得更好结果 (例如: "Python 最新版本 发布日期" 而非 "Python的最新版本是多少")。
        model: 指定 Gemini 模型 (默认: 读取 GEMINI_MODEL，否则 gemini-2.5-flash)。
        retry_count: 失败重试次数 (默认: 读取 GEMINI_RETRY_COUNT，否则 3)。
        retry_delay: 重试等待时间(秒) (默认: 读取 GEMINI_RETRY_DELAY，否则 5.0)。
        search_delay_min: 搜索前最小随机延迟(秒) (默认: 读取 GEMINI_SEARCH_DELAY_MIN，否则 0.0)。
        search_delay_max: 搜索前最大随机延迟(秒) (默认: 读取 GEMINI_SEARCH_DELAY_MAX，否则 0.0)。
        retry_until_success: 是否持续重试直到拿到非空结果 (默认: 读取 GEMINI_RETRY_UNTIL_SUCCESS，否则 False；MCP / Skill 场景建议设为 True)。
    """
    try:
        options = resolve_search_options(
            model=model,
            retry_count=retry_count,
            retry_delay=retry_delay,
            search_delay_min=search_delay_min,
            search_delay_max=search_delay_max,
            retry_until_success=retry_until_success,
        )

        result = search(
            query,
            model=options["model"],
            retry_count=options["retry_count"],
            retry_delay=options["retry_delay"],
            search_delay_min=options["search_delay_min"],
            search_delay_max=options["search_delay_max"],
            retry_until_success=options["retry_until_success"],
        )

        output = result["text"]
        if result["sources"]:
            output += "\n\n## Sources\n"
            for src in result["sources"]:
                output += f"{src['id']}. [{src['title']}]({src['url']})\n"

        return output
    except ValueError as e:
        return f"参数错误: {str(e)}"
    except requests.RequestException as e:
        # 脱敏：移除错误信息中的 URL，防止泄露 base_url
        sanitized = re.sub(r"https?://\S+", "[REDACTED_URL]", str(e))
        return f"网络请求失败: {sanitized}"
    except Exception as e:
        # 脱敏：移除错误信息中的 URL，防止泄露 base_url
        sanitized = re.sub(r"https?://\S+", "[REDACTED_URL]", str(e))
        return f"搜索失败: {sanitized}"


if __name__ == "__main__":
    mcp.run()
