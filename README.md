# Gemini Grounding MCP Server & Search Skill

这是一个基于 Google Gemini API 和 Grounding（搜索溯源）功能的 **Model Context Protocol (MCP)** 服务。它允许 AI 助手（如 Claude Desktop）执行联网搜索，并返回带有准确来源引用的事实性回答。

同时，本项目也可作为独立的 **AI Skill** 被其他 Agent 系统集成。

## ✨ 功能特点

- **MCP 协议支持**：作为标准 MCP 服务器运行，可集成到 Claude Desktop 等客户端。
- **Google Grounding**：利用 Google 搜索索引提供实时、准确的信息，并附带来源链接。
- **抗限流机制**：内置重试逻辑 (Retry) 和随机延迟 (Jitter)，提高在 API 不稳定或限流情况下的成功率。
- **多模式运行**：既可以作为 MCP 服务运行，也可以作为 CLI 工具或 AI Skill 使用。
- **灵活配置**：支持自定义模型、API 端点（兼容 NewAPI）、重试次数和延迟时间。

## 📦 安装指南

本项目使用 `uv` 进行依赖管理。

1.  **安装 uv** (如果尚未安装):
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **克隆项目并同步依赖**:
    ```bash
    git clone https://github.com/your-repo/gemini-grounding.git
    cd gemini-grounding
    uv sync
    ```

## 🚀 使用方法 1：MCP 服务 (Claude Desktop)

在 Claude Desktop 的配置文件中添加以下内容。

*   **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
*   **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "gemini-grounding": {
      "command": "uv",
      "args": [
        "run",
        "/absolute/path/to/gemini-grounding/src/gemini_grounding/mcp_server.py"
      ],
      "env": {
        "GEMINI_API_KEY": "your-google-api-key",
        "GEMINI_MODEL": "gemini-2.5-flash",
        "GEMINI_RETRY_COUNT": "3",
        "GEMINI_RETRY_DELAY": "5",
        "GEMINI_PROXY_URL": "https://rp.0x01111110.com" 
      }
    }
  }
}
```

> **注意**: 
> 1. 请将 `/absolute/path/to/...` 替换为实际的绝对路径。
> 2. `GEMINI_PROXY_URL` 为可选配置，若不设置则直接解析重定向链接。

## 🚀 使用方法 2：AI Skill 集成 (CLI)

如果您希望在自定义 Agent (如 OpenCode, AutoGen, LangChain) 中集成此工具作为 Skill，请使用以下命令模式：

**命令模板**:
```bash
export GEMINI_API_KEY="your-key" && uv run src/gemini_grounding/search.py --query "您的搜索关键词"
```

**Skill 定义参考**:
请参考项目中的 `SKILL.md` 文件，其中包含了详细的 Skill 描述、触发词和使用示例，可直接复制到您的 Agent 系统配置中。

**示例调用**:
```bash
# 搜索 Python 最新版本
uv run src/gemini_grounding/search.py --query "Python 最新版本发布时间"
```

## 🚀 使用方法 3：通用 MCP 客户端

对于支持 MCP 协议的其他客户端（如 Cursor, Zed 或自定义客户端），配置通常只需指定服务器的启动命令。

**启动命令**:
```bash
uv run src/gemini_grounding/mcp_server.py
```

**环境变量**:
确保运行环境已设置以下变量（或通��客户端配置传递）：
- `GEMINI_API_KEY`

## ⚙️ 配置说明

您可以通过环境变量或 `.env` 文件来配置服务行为：

| 环境变量 | 说明 | 默认值 | 示例 |
| :--- | :--- | :--- | :--- |
| `GEMINI_API_KEY` | **(必填)** Google Gemini API 密钥 | - | `AIzaSy...` |
| `GEMINI_MODEL` | 使用的模型名称 | `gemini-2.5-flash` | `gemini-2.0-flash-exp` |
| `GEMINI_BASE_URL` | API 基础 URL (支持 NewAPI) | Google 官方 API | `https://api.newapi.com/v1/gemini` |
| `GEMINI_RETRY_COUNT` | 失败重试次数 | `3` | `5` |
| `GEMINI_RETRY_DELAY` | 重试等待时间 (秒) | `5` | `10` |
| `GEMINI_SEARCH_DELAY_MIN` | 搜索前最小随机延迟 (秒) | `0.0` | `1.0` |
| `GEMINI_SEARCH_DELAY_MAX` | 搜索前最大随机延迟 (秒) | `0.0` | `3.0` |
| `GEMINI_CACHE_TTL` | 搜索结果缓存过期时间 (秒) | `3600` | `600` |
| `GEMINI_CACHE_MAXSIZE` | 搜索结果缓存最大条目数 | `100` | `500` |
| `GEMINI_PROXY_URL` | 解析 Grounding 链接的代理服务 URL | - | `https://rp.0x01111110.com` |

### 关于重试与延迟

为了应对 API 限流（Rate Limiting），您可以配置：
- **Retry**: 请求失败时自动重试。
- **Jitter (Delay)**: 在发起请求前增加随机等待时间，避免并发请求瞬间打满 QPS 限制。

## 🛠️ 开发与测试

**运行单元测试 (Mock)**:
```bash
uv run src/gemini_grounding/tests/test_mcp_mock.py
```

**运行真实调用测试 (Real)**:
```bash
# 需要先配置 .env 文件
export $(grep -v '^#' .env | xargs) && uv run src/gemini_grounding/tests/test_mcp_real.py
```

## 📂 项目结构

```
gemini-grounding/
├── src/
│   └── gemini_grounding/
│       ├── search.py        # 核心搜索逻辑 & CLI / Skill 入口
│       ├── mcp_server.py    # MCP 服务器入口
│       ├── tests/           # 测试代码
│       │   ├── test_mcp_mock.py
│       │   ├── test_mcp_real.py
│       │   └── test_proxy.py
│       └── __init__.py
├── worker/                  # Cloudflare Worker 代理 (用于解析重定向链接)
├── SKILL.md                 # AI Agent 技能定义文档
├── pyproject.toml           # 依赖管理配置
└── README.md                # 说明文档
```
