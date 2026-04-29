# EasyPaper

EasyPaper 是一个多智能体学术论文生成系统。它将一小部分元数据（标题、想法、方法、数据、实验、参考文献）转化为结构化的 LaTeX 论文，并可选择通过排版智能体将其编译成 PDF。

## 功能

- **Python SDK** — `pip install easypaper`，然后在您自己的项目中 `import easypaper`
- **流式生成** — 异步生成器在每个阶段生成实时进度事件
- 多智能体管道：规划、撰写、评审、排版以及可选的 VLM 评审
- 可选的 FastAPI 服务器模式，带有健康检查和智能体发现端点
- LaTeX 输出，包含引用验证、图/表插入和评审循环

## 要求

- Python 3.11+
- LaTeX 工具链 (`pdflatex` + `bibtex`) 用于 PDF 编译
- [Poppler](https://poppler.freedesktop.org/) — `pdf2image` 进行 PDF 到图像转换所需
  - macOS: `brew install poppler`
  - Ubuntu/Debian: `apt install poppler-utils`
- 模型 API 密钥在 YAML 中配置（参见[配置](#config)）

## SDK 用法

从 PyPI 安装：

```bash
pip install easypaper
```

当您的配置包含 `typesetter` 模型时，SDK 模式下的 PDF 编译是自包含的。SDK 现在优先进行进程内排版程序（Typesetter）执行，必要时会回退到 HTTP 端点（`AGENTSYS_SELF_URL`）。

### 一次性生成

**内联元数据：**

```python
import asyncio
from easypaper import EasyPaper, PaperMetaData

async def main():
    ep = EasyPaper(config_path="config.yaml")

    metadata = PaperMetaData(
        title="My Paper Title",
        idea_hypothesis="...",
        method="...",
        data="...",
        experiments="...",
    )

    result = await ep.generate(metadata)
    print(result.status, result.total_word_count)
    for sec in result.sections:
        print(f"  {sec.section_type}: {sec.word_count} words")

asyncio.run(main())
```

**从 JSON 文件加载元数据（推荐）：**
准备一个 `metadata.json`（完整 schema 请参见 [`examples/meta.json`](examples/meta.json)）。干净的 SDK 流程是解析为 `PaperGenerationRequest`，然后拆分为内容元数据和运行时选项：

```python
import asyncio
from easypaper import EasyPaper, PaperGenerationRequest

async def main():
    ep = EasyPaper(config_path="config.yaml")

    request = PaperGenerationRequest.model_validate_json_file("metadata.json")
    metadata = request.to_metadata()
    options = request.to_generate_options()

    result = await ep.generate(metadata, **options)
    print(result.status, result.total_word_count)

asyncio.run(main())
```

对于最小的仅元数据 JSON（无运行时选项），您可以使用：

```python
metadata = PaperMetaData.model_validate_json_file("metadata.json")
result = await ep.generate(metadata)
```

### 查找最终 PDF

当评审循环启用时，EasyPaper 可能会编译多个 PDF（每次迭代一个）。使用以下优先级来定位最终成品：

1.  首先使用 `result.pdf_path`（权威的最终路径）。
2.  如果缺失，则按此顺序在 `result.output_path` 下搜索：
    -   `iteration_*_final/**/*.pdf`（最高优先级）
    -   最新的 `iteration_*` 目录中的 PDF
    -   根目录下的 `paper.pdf` 作为最后备选

```python
from pathlib import Path
import re

def resolve_final_pdf(result) -> str | None:
    if getattr(result, "pdf_path", None):
        p = Path(result.pdf_path)
        if p.exists():
            return str(p.resolve())

    out = getattr(result, "output_path", None)
    if not out:
        return None
    base = Path(out)
    if not base.exists():
        return None

    finals = sorted(base.glob("iteration_*_final/**/*.pdf"))
    if finals:
        return str(finals[-1].resolve())

    iter_dirs = []
    for d in base.glob("iteration_*"):
        m = re.match(r"iteration_(\\d+)$", d.name)
        if m and d.is_dir():
            iter_dirs.append((int(m.group(1)), d))
    if iter_dirs:
        iter_dirs.sort(key=lambda x: x[0])
        pdfs = sorted(iter_dirs[-1][1].glob("**/*.pdf"))
        if pdfs:
            return str(pdfs[-1].resolve())

    root_pdf = base / "paper.pdf"
    if root_pdf.exists():
        return str(root_pdf.resolve())
    return None
```

### 流式生成

使用 `generate_stream()` 通过异步生成器接收实时进度事件。元数据可以内联构建，也可以从 JSON 文件加载（例如 `PaperMetaData.model_validate_json_file("metadata.json")`）。

```python
import asyncio
from easypaper import EasyPaper, PaperMetaData, EventType

async def main():
    ep = EasyPaper(config_path="config.yaml")
    metadata = PaperMetaData.model_validate_json_file("metadata.json")  # 或内联构建

    async for event in ep.generate_stream(metadata):
        if event.event_type == EventType.PHASE_START:
            print(f"▶ [{event.phase}] {event.message}")
        elif event.event_type == EventType.SECTION_COMPLETE:
            print(f"  ✎ {event.phase} done")
        elif event.event_type == EventType.COMPLETE:
            result = event.data["result"]
            print(f"Done! {result['total_word_count']} words")

asyncio.run(main())
```

`GenerationEvent` 字段：

| 字段 | 类型 | 描述 |
|---|---|---|
| `event_type` | `EventType` | `PHASE_START`、`PHASE_COMPLETE`、`SECTION_COMPLETE`、`PROGRESS`、`WARNING`、`ERROR`、`COMPLETE` |
| `phase` | `str` | 逻辑阶段名称（例如 `"planning"`、`"introduction"`、`"body"`） |
| `message` | `str` | 人类可读的描述 |
| `data` | `dict \| None` | 结构化负载（章节内容、最终结果等） |
| `timestamp` | `datetime` | 事件创建时间 |

完整的示例可在 [`user_case/`](user_case/) 中找到。

## 服务器模式

要将 EasyPaper 作为 FastAPI 服务运行（需要 `server` 额外依赖）：

```bash
pip install "easypaper[server]"
```

1.  复制示例配置并填写您的 API 密钥：

```bash
cp configs/example.yaml configs/dev.yaml
```

2.  启动服务器：

```bash
uvicorn easypaper.main:app --reload --port 8000
```

3.  通过 API 生成：

```bash
curl -X POST http://localhost:8000/metadata/generate \
  -H "Content-Type: application/json" \
  -d @economist_example/metadata.json
```

## 技能

EasyPaper 包含一个可插拔的**技能**系统，该系统将写作约束、特定场地格式规则和审稿人检查器注入生成管道。内置技能作为静态资源捆绑在 `easypaper` 包中：

| 类别 | 技能 | 描述 |
|---|---|---|
| **写作** | `anti-ai-style`、`academic-polish`、`latex-conventions` | 应用于所有章节的风格约束 — 消除 AI 风格措辞，强制学术语气，确保 LaTeX 最佳实践 |
| **场地** | `neurips`、`icml`、`iclr`、`acl`、`aaai`、`colm`、`nature` | 会议/期刊配置文件，包含页数限制、格式规则和特定场地风格要求 |
| **评审** | `logic-check`、`style-check` | 审稿人检查器提示 — 检测逻辑矛盾、术语不一致和风格违规 |

### 启用技能

默认加载内置技能。您只需要在配置中设置 `enabled`/`active_skills`：

```yaml
skills:
  enabled: true
  active_skills:
    - "*"                   # "*" = 加载所有技能；或列出特定名称
```

如果您添加 `skills.skills_dir`，EasyPaper 会将捆绑技能与该目录**合并**：相同的技能 `name` → 您的文件优先。如果路径缺失，捆绑技能仍会加载并记录警告。

### 场地配置文件

要应用特定场地约束（例如页数限制、格式），请在 `PaperMetaData` 中将 `style_guide` 设置为与场地配置文件名称匹配：

```python
metadata = PaperMetaData(
    title="...",
    idea_hypothesis="...",
    method="...",
    data="...",
    experiments="...",
    style_guide="neurips",   # 激活 neurips 场地配置文件
)
```

### 自定义技能

每个技能都是一个具有以下结构的 YAML 文件：

```yaml
name: my-custom-skill
description: "What this skill does" # 此技能的功能
type: writing_constraint   # writing_constraint | reviewer_checker | venue_profile
target_sections: ["*"]     # ["*"] = 所有章节，或特定章节
priority: 10               # 较低 = 较高优先级

system_prompt_append: |
  ## My Custom Rules
  - Rule 1: ... # 规则 1：...
  - Rule 2: ... # 规则 2：...

anti_patterns:
  - "word to avoid" # 要避免的词语
```

将文件放入您的 `skills_dir`，它将在下次运行时自动加载。有关完整示例，请参见 `easypaper/assets/skills/` 中的内置技能。

## 配置

应用程序从 `AGENT_CONFIG_PATH` 加载配置（默认为 `./configs/dev.yaml`）。您也可以在项目根目录的 `.env` 文件中设置此变量。

有关完整注释的配置模板，请参见 `configs/example.yaml`。每个智能体条目定义其模型和可选的智能体特定设置。

每个智能体的关键字段：
- `model_name` — LLM 模型标识符
- `api_key` — 模型提供商的 API 密钥
- `base_url` — API 端点 URL

其他顶级部分：
- `skills` — 技能系统开关和活动技能列表（参见[技能](#skills)）
- `tools` — ReAct 工具配置（引用验证、论文搜索等）
- `vlm_service` — 用于视觉评审的共享 VLM 提供商（支持 OpenAI 兼容和 Claude）

## 仓库布局

- `easypaper/` — SDK 核心、智能体实现、事件系统、共享工具
- `configs/` — 智能体和模型的 YAML 配置
- `skills/` — 由 Python 服务加载的后端 YAML 技能
- `plugins/easypaper/` — Claude/OpenCode 插件根目录（命令 + SKILL.md 提示）
- `.claude-plugin/marketplace.json` — 市场目录
- `.opencode/opencode.json` — OpenCode/OpenClaw 运行时配置
- `AGENTS.md` — 仓库级智能体编码说明
- `scripts/` — CLI 工具和演示
- `user_case/` — 独立使用示例（独立环境）
- `economist_example/` — 示例元数据输入

## Claude Code 插件市场

此仓库提供了一个 Claude Code 市场，其中包含一个位于 `plugins/easypaper` 的可安装插件。

### 安装

添加市场：

```bash
/plugin marketplace add https://github.com/your-username/easypaper
```

从该市场安装插件：

```bash
/plugin install easypaper@easypaper
```

### 可用插件

| 插件 | 来源 | 描述 |
|--------|--------|-------------|
| easypaper | `./plugins/easypaper` | 通过元数据交互式地生成 AI 驱动的学术论文 |

### 用法

安装后：

```bash
/easypaper
```

相关命令：

```bash
/paper-from-metadata
/paper-section
```

### 插件先决条件

- Python 3.11+
- 已安装 `easypaper` 包（`pip install easypaper`）
- LaTeX 工具链（pdflatex + bibtex）用于 PDF 编译
- LLM 提供商的 API 密钥（通过配置文件配置）

## OpenCode / OpenClaw 用法

此仓库在 `.opencode/opencode.json` 中包含原生的 OpenCode/OpenClaw 配置。

### 在此仓库中直接运行

```bash
opencode
```

运行时加载：

- 插件路径：`./plugins/easypaper`
- 技能：`plugins/easypaper/skills/*/SKILL.md`
- 命令：`easypaper`、`paper-from-metadata`、`paper-section`

对于 Claude Code 和 OpenCode/OpenClaw 工作流，请在生成之前启动 EasyPaper API 服务：

```bash
uv run uvicorn easypaper.main:app --reload --port 8000
```
