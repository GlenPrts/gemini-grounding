"""
MCP 服务压测脚本

功能：通过 MCP 协议对 gemini-grounding 服务进行多维度压测
测试项：并发调用、缓存命中、错误处理、响应时间分布
"""

import asyncio
import os
import sys
import time
import statistics
import traceback
from dataclasses import dataclass, field
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# ============================================================
# 数据结构
# ============================================================


@dataclass
class CallResult:
    """单次调用结果"""

    query: str
    success: bool
    duration: float
    error: str = ""
    response_len: int = 0
    has_sources: bool = False


@dataclass
class StressReport:
    """压测报告"""

    results: list = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def total(self):
        return len(self.results)

    @property
    def successes(self):
        return [r for r in self.results if r.success]

    @property
    def failures(self):
        return [r for r in self.results if not r.success]

    @property
    def durations(self):
        return [r.duration for r in self.successes]


# ============================================================
# 工具函数
# ============================================================


def format_duration(seconds):
    """
    格式化耗时为可读字符串

    参数:
        seconds: 秒数
    返回值:
        格式化后的字符串，如 "1.23s"
    """
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.2f}s"


def print_separator(title=""):
    """
    打印分隔线

    参数:
        title: 分隔线标题
    返回值:
        无
    """
    if title:
        print(f"\n{'=' * 20} {title} {'=' * 20}")
    else:
        print("=" * 60)


def compute_stats(durations):
    """
    计算耗时统计数据

    参数:
        durations: 耗时列表（秒）
    返回值:
        dict: 包含 min/max/avg/median/p95/p99/stdev 的字典
    """
    if not durations:
        return {}

    sorted_d = sorted(durations)
    n = len(sorted_d)

    result = {
        "min": sorted_d[0],
        "max": sorted_d[-1],
        "avg": statistics.mean(sorted_d),
        "median": statistics.median(sorted_d),
        "p95": sorted_d[int(n * 0.95)] if n >= 2 else sorted_d[-1],
        "p99": sorted_d[int(n * 0.99)] if n >= 2 else sorted_d[-1],
    }

    if n >= 2:
        result["stdev"] = statistics.stdev(sorted_d)

    return result


# ============================================================
# 压测用例
# ============================================================

# 不同查询用于测试并发去重和缓存
QUERIES_UNIQUE = [
    "Python 最新版本发布时间",
    "Rust 2025 新特性",
    "TypeScript 5.x 变更",
    "Linux kernel 最新版本",
    "Go 1.23 发布说明",
]

# 重复查询用于验证缓存命中
QUERY_CACHED = "Python 最新版本发布时间"

# 边界测试查询
QUERIES_EDGE = [
    "",  # 空查询
    "a",  # 极短查询
    "x " * 500,  # 超长查询
    "🔥🎉💻🐍",  # 纯 emoji
    "<script>alert(1)</script>",  # XSS 注入
    "' OR 1=1 --",  # SQL 注入
]


# ============================================================
# 核心测试逻辑
# ============================================================


async def call_tool(session, query, tool_args=None):
    """
    调用 MCP 工具并记录结果

    参数:
        session: MCP 客户端会话
        query: 搜索查询
        tool_args: 额外的工具参数字典
    返回值:
        CallResult: 调用结果
    """
    args = {"query": query}
    if tool_args:
        args.update(tool_args)

    start = time.monotonic()
    try:
        result = await asyncio.wait_for(
            session.call_tool("google_search", arguments=args),
            timeout=60,
        )
        elapsed = time.monotonic() - start

        text = ""
        for content in result.content:
            if content.type == "text":
                text += content.text

        has_sources = "## Sources" in text
        ERROR_PREFIXES = (
            "Error performing search:",
            "搜索失败:",
            "参数错误:",
        )
        is_error = any(text.startswith(p) for p in ERROR_PREFIXES)

        return CallResult(
            query=query[:50],
            success=not is_error,
            duration=elapsed,
            response_len=len(text),
            has_sources=has_sources,
            error=text[:200] if is_error else "",
        )
    except asyncio.TimeoutError:
        elapsed = time.monotonic() - start
        return CallResult(
            query=query[:50],
            success=False,
            duration=elapsed,
            error="超时 (60s)",
        )
    except Exception as e:
        elapsed = time.monotonic() - start
        return CallResult(
            query=query[:50],
            success=False,
            duration=elapsed,
            error=str(e)[:200],
        )


async def test_sequential(session, report):
    """
    顺序调用测试：逐个发送请求，测量基准延迟

    参数:
        session: MCP 客户端会话
        report: 压测报告对象
    返回值:
        无
    """
    print_separator("测试1: 顺序调用 (基准延迟)")
    for q in QUERIES_UNIQUE[:3]:
        r = await call_tool(session, q)
        report.results.append(r)
        status = "✓" if r.success else "✗"
        print(
            f"  {status} [{format_duration(r.duration)}] "
            f'q="{r.query}" '
            f"len={r.response_len} "
            f"sources={r.has_sources}"
        )
        if not r.success:
            print(f"    错误: {r.error}")


async def test_concurrent(session, report):
    """
    并发调用测试：同时发送多个不同请求

    参数:
        session: MCP 客户端会话
        report: 压测报告对象
    返回值:
        无
    """
    print_separator("测试2: 并发调用 (5个不同查询)")
    tasks = [call_tool(session, q) for q in QUERIES_UNIQUE]
    results = await asyncio.gather(*tasks)
    for r in results:
        report.results.append(r)
        status = "✓" if r.success else "✗"
        print(
            f"  {status} [{format_duration(r.duration)}] "
            f'q="{r.query}" '
            f"len={r.response_len}"
        )
        if not r.success:
            print(f"    错误: {r.error}")


async def test_cache(session, report):
    """
    缓存命中测试：重复相同查询，验证缓存加速

    参数:
        session: MCP 客户端会话
        report: 压测报告对象
    返回值:
        无
    """
    print_separator("测试3: 缓存命中 (同一查询3次)")
    cache_results = []
    for i in range(3):
        r = await call_tool(session, QUERY_CACHED)
        report.results.append(r)
        cache_results.append(r)
        status = "✓" if r.success else "✗"
        print(
            f"  第{i + 1}次: {status} "
            f"[{format_duration(r.duration)}] "
            f"len={r.response_len}"
        )

    # 分析缓存效果
    if len(cache_results) >= 2:
        first = cache_results[0].duration
        rest_avg = statistics.mean([r.duration for r in cache_results[1:]])
        if first > 0:
            speedup = first / rest_avg if rest_avg > 0 else float("inf")
            print(f"  缓存加速比: {speedup:.1f}x")


async def test_concurrent_same_query(session, report):
    """
    并发同查询测试：同时发送相同查询，检查竞态

    参数:
        session: MCP 客户端会话
        report: 压测报告对象
    返回值:
        无
    """
    print_separator("测试4: 并发同查询 (竞态检测)")
    fresh_query = "Node.js 最新 LTS 版本"
    tasks = [call_tool(session, fresh_query) for _ in range(5)]
    results = await asyncio.gather(*tasks)

    responses = set()
    for i, r in enumerate(results):
        report.results.append(r)
        status = "✓" if r.success else "✗"
        print(
            f"  副本{i + 1}: {status} "
            f"[{format_duration(r.duration)}] "
            f"len={r.response_len}"
        )
        if r.success:
            responses.add(r.response_len)

    if len(responses) > 1:
        print("  ⚠ 警告: 同查询返回了不同长度的响应，可能存在竞态问题")


async def test_edge_cases(session, report):
    """
    边界条件测试：特殊输入的处理能力

    参数:
        session: MCP 客户端会话
        report: 压测报告对象
    返回值:
        无
    """
    print_separator("测试5: 边界条件")
    labels = [
        "空查询",
        "极短查询",
        "超长查询",
        "纯emoji",
        "XSS注入",
        "SQL注入",
    ]
    for label, q in zip(labels, QUERIES_EDGE):
        r = await call_tool(session, q)
        report.results.append(r)
        status = "✓" if r.success else "✗"
        print(
            f"  {status} [{label}] "
            f"dur={format_duration(r.duration)} "
            f"len={r.response_len}"
        )
        if not r.success:
            print(f"    错误: {r.error}")


async def test_invalid_params(session, report):
    """
    无效参数测试：非法参数组合的容错能力

    参数:
        session: MCP 客户端会话
        report: 压测报告对象
    返回值:
        无
    """
    print_separator("测试6: 无效参数")
    invalid_cases = [
        (
            "不存在的模型",
            {"model": "nonexistent-model-xyz"},
        ),
        (
            "负数重试",
            {"retry_count": -1},
        ),
        (
            "超大重试延迟",
            {"retry_delay": 99999},
        ),
    ]
    for label, extra_args in invalid_cases:
        r = await call_tool(session, "test query", tool_args=extra_args)
        report.results.append(r)
        status = "✓" if r.success else "✗"
        print(
            f"  {status} [{label}] "
            f"dur={format_duration(r.duration)} "
            f"err={r.error[:80] if r.error else 'none'}"
        )


async def test_burst(session, report):
    """
    突发流量测试：短时间内发送大量请求

    参数:
        session: MCP 客户端会话
        report: 压测报告对象
    返回值:
        无
    """
    print_separator("测试7: 突发流量 (10个并发请求)")
    burst_queries = [f"技术问题 {i}" for i in range(10)]
    tasks = [call_tool(session, q) for q in burst_queries]

    start = time.monotonic()
    results = await asyncio.gather(*tasks)
    wall_time = time.monotonic() - start

    ok = sum(1 for r in results if r.success)
    fail = sum(1 for r in results if not r.success)

    for r in results:
        report.results.append(r)

    print(f"  总耗时: {format_duration(wall_time)}")
    print(f"  成功: {ok}, 失败: {fail}")
    print(f"  吞吐量: {len(results) / wall_time:.2f} req/s")

    # 打印失败详情
    for r in results:
        if not r.success:
            print(f'  ✗ q="{r.query}" err={r.error[:80]}')


# ============================================================
# 报告生成
# ============================================================


def print_report(report):
    """
    输出压测汇总报告

    参数:
        report: StressReport 对象
    返回值:
        无
    """
    print_separator("压测汇总报告")

    total_time = report.end_time - report.start_time
    print(f"总耗时: {format_duration(total_time)}")
    print(
        f"总请求: {report.total} | "
        f"成功: {len(report.successes)} | "
        f"失败: {len(report.failures)}"
    )

    if report.total > 0:
        rate = len(report.successes) / report.total * 100
        print(f"成功率: {rate:.1f}%")

    durations = report.durations
    if durations:
        stats = compute_stats(durations)
        print(f"\n响应时间分布 (仅成功请求):")
        print(f"  最小: {format_duration(stats['min'])}")
        print(f"  最大: {format_duration(stats['max'])}")
        print(f"  平均: {format_duration(stats['avg'])}")
        print(f"  中位: {format_duration(stats['median'])}")
        print(f"  P95:  {format_duration(stats['p95'])}")
        print(f"  P99:  {format_duration(stats['p99'])}")
        if "stdev" in stats:
            print(f"  标准差: {format_duration(stats['stdev'])}")

    # 打印所有失败详情
    if report.failures:
        print(f"\n失败详情 ({len(report.failures)} 个):")
        for r in report.failures:
            print(
                f'  ✗ q="{r.query}" '
                f"dur={format_duration(r.duration)} "
                f"err={r.error[:100]}"
            )

    # 无来源的成功请求
    no_src = [r for r in report.successes if not r.has_sources]
    if no_src:
        print(f"\n⚠ {len(no_src)} 个成功请求无来源引用:")
        for r in no_src:
            print(f'  q="{r.query}"')

    print_separator()


# ============================================================
# 主入口
# ============================================================


async def run():
    """
    压测主流程：启动 MCP 服务并依次执行各测试用例

    参数: 无
    返回值: 无
    """
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    server_script = os.path.join(current_dir, "mcp_server.py")

    print(f"MCP 服务脚本: {server_script}")

    env = os.environ.copy()
    server_params = StdioServerParameters(
        command="uv",
        args=["run", server_script],
        env=env,
    )

    report = StressReport()

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # 验证工具列表
                tools = await session.list_tools()
                tool_names = [t.name for t in tools.tools]
                print(f"可用工具: {tool_names}")

                if "google_search" not in tool_names:
                    print("错误: google_search 工具未找到")
                    return

                report.start_time = time.monotonic()

                # 依次执行测试
                await test_sequential(session, report)
                await test_concurrent(session, report)
                await test_cache(session, report)
                await test_concurrent_same_query(session, report)
                await test_edge_cases(session, report)
                await test_invalid_params(session, report)
                await test_burst(session, report)

                report.end_time = time.monotonic()

    except Exception as e:
        print(f"\n致命错误: {e}")
        traceback.print_exc()
        report.end_time = time.monotonic()

    print_report(report)


if __name__ == "__main__":
    asyncio.run(run())
