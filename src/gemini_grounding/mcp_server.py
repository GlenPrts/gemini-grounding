import asyncio
import re

from mcp.server.fastmcp import FastMCP, Context
from mcp.server.fastmcp.exceptions import ToolError

try:
    from .search import resolve_search_options, search, ensure_initialized
except ImportError:
    from search import resolve_search_options, search, ensure_initialized

_MAX_QUERY_LENGTH = 2000

mcp = FastMCP("gemini-grounding")
mcp._mcp_server.version = "0.1.0"


@mcp.tool()
async def google_search(
    query: str,
    model: str | None = None,
    retry_count: int | None = None,
    retry_delay: float | None = None,
    search_delay_min: float | None = None,
    search_delay_max: float | None = None,
    retry_until_success: bool | None = None,
    ctx: Context | None = None,
) -> str:
    """
    利用 Google 搜索 (Gemini Grounding) 获取带有来源引用的实时事实信息。

    适用于：
    - 时事新闻：新闻、发布日期、体育比分、近期动态。
    - 事实核查：验证特定声明、统计数据或历史细节。
    - 外部知识：可能超出你训练截止日期或专有文档的信息。

    为了获得最佳搜索结果，请务必拆分并优化搜索语句。建议针对单一特定的信息点进行搜索，宁可进行多次精准搜索，也不要尝试一次性搜索过多复杂内容。

    Args:
        query: 搜索关键词。建议将对话式问题转换为关键词查询以获得更好结果 (例如: "Python 最新版本 发布日期" 而非 "Python的最新版本是多少")。
        model: 指定 Gemini 模型 (默认: 读取 GEMINI_MODEL，否则 gemini-2.5-flash)。
        retry_count: 失败重试次数 (默认: 读取 GEMINI_RETRY_COUNT，否则 3)。
        retry_delay: 重试等待时间(秒) (默认: 读取 GEMINI_RETRY_DELAY，否则 5.0)。
        search_delay_min: 搜索前最小随机延迟(秒) (默认: 读取 GEMINI_SEARCH_DELAY_MIN，否则 0.0)。
        search_delay_max: 搜索前最大随机延迟(秒) (默认: 读取 GEMINI_SEARCH_DELAY_MAX，否则 0.0)。
        retry_until_success: 是否持续重试直到拿到非空结果 (默认: 读取 GEMINI_RETRY_UNTIL_SUCCESS，否则 False；MCP / Skill 场景建议设为 True)。
    """
    if not query or not query.strip():
        raise ToolError("搜索查询不能为空")

    if len(query) > _MAX_QUERY_LENGTH:
        raise ToolError(f"查询长度 ({len(query)}) 超过最大限制 ({_MAX_QUERY_LENGTH})")

    if (
        search_delay_min is not None
        and search_delay_max is not None
        and search_delay_min > search_delay_max
    ):
        raise ToolError(
            f"search_delay_min ({search_delay_min}) 不能大于 search_delay_max ({search_delay_max})"
        )

    try:
        ensure_initialized()

        options = resolve_search_options(
            model=model,
            retry_count=retry_count,
            retry_delay=retry_delay,
            search_delay_min=search_delay_min,
            search_delay_max=search_delay_max,
            retry_until_success=retry_until_success,
        )

        if ctx:
            await ctx.info(
                f"Searching: {query!r} (model={options['model']}, "
                f"retry_until_success={options['retry_until_success']})"
            )

        # Run blocking search in a thread to avoid blocking the MCP event loop
        result = await asyncio.to_thread(
            search,
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

    except ToolError:
        raise
    except ValueError as e:
        raise ToolError(f"参数错误: {e}") from e
    except Exception as e:
        # Sanitize: remove URLs to prevent leaking base_url / api_key
        sanitized = re.sub(r"https?://\S+", "[REDACTED_URL]", str(e))
        raise ToolError(f"搜索失败: {sanitized}") from e


if __name__ == "__main__":
    mcp.run()
