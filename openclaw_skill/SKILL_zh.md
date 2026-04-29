---
name: easypaper
description: "使用 EasyPaper Python SDK 根据元数据生成学术论文。当用户需要以编程方式创建结构化 LaTeX 论文时使用。参考 EasyPaper 仓库，特别是 plugins/easypaper/ 目录获取详细工作流、命令和技能。"
---

# EasyPaper 技能

通过 Python SDK 使用 EasyPaper 多智能体系统，根据元数据生成结构化学术论文。

## 仓库

**源码**：https://github.com/PinkGranite/EasyPaper

**主要参考目录**：[`plugins/easypaper/`](https://github.com/PinkGranite/EasyPaper/tree/master/plugins/easypaper)

该目录包含 OpenClaw 智能体的完整指南：
- **命令**：工作流执行合约位于 `plugins/easypaper/commands/`
- **技能**：领域特定技能位于 `plugins/easypaper/skills/`
- **插件文档**：安装与使用说明在 `plugins/easypaper/.claude-plugin/README.md`

## 安装

### Python 包

**重要**：在隔离环境中安装 EasyPaper（推荐用于依赖管理）。

**使用 venv**：
```bash
python -m venv easypaper-env
source easypaper-env/bin/activate  # Windows系统: easypaper-env\Scripts\activate
pip install easypaper
```

**使用 conda**：
```bash
conda create -n easypaper python=3.11
conda activate easypaper
pip install easypaper
```

**直接安装**（不推荐）：
```bash
pip install easypaper
```

### LaTeX 工具链

EasyPaper 需要 LaTeX 工具链（`pdflatex` + `bibtex`）进行 PDF 编译。根据系统安装：

**macOS**：
```bash
# 使用 Homebrew（推荐）
brew install --cask mactex

# 或最小化安装
brew install basictex
sudo tlmgr update --self
sudo tlmgr install collection-basic collection-latex collection-bibtexextra
```

**Linux (Ubuntu/Debian)**：
```bash
sudo apt-get update
sudo apt-get install texlive-latex-base texlive-bibtex-extra texlive-latex-extra
```

**Linux (Fedora/RHEL)**：
```bash
sudo dnf install texlive-scheme-basic texlive-bibtex texlive-latex
```

**Windows**：
- 下载安装 [MiKTeX](https://miktex.org/download)（推荐完整版）
- 或使用 [TeX Live](https://www.tug.org/texlive/windows.html)
- 确保 `pdflatex` 和 `bibtex` 在 PATH 环境变量中

### Poppler（用于 PDF 转图像）

**macOS**：
```bash
brew install poppler
```

**Linux (Ubuntu/Debian)**：
```bash
sudo apt-get install poppler-utils
```

**Linux (Fedora/RHEL)**：
```bash
sudo dnf install poppler-utils
```

**Windows**：
- 从 [Poppler for Windows](http://blog.alivate.com.au/poppler-windows/) 下载
- 解压并将 `bin` 目录加入 PATH
- 或使用 [conda](https://anaconda.org/conda-forge/poppler)：`conda install -c conda-forge poppler`

## 快速开始

**推荐工作流**：准备 `metadata.json`（参考 [`examples/meta.json`](https://github.com/PinkGranite/EasyPaper/blob/master/examples/meta.json)），解析为 `PaperGenerationRequest`，然后通过 `to_metadata()` + `to_generate_options()` 运行。

**排版器行为（SDK + 服务端）**：优先使用进程内排版器（SDK 自包含）。若无本地节点可用，则回退至 HTTP 排版器端点（`AGENTSYS_SELF_URL`）。

### 从文件加载并生成

```python
import asyncio
from pathlib import Path
from easypaper import EasyPaper, PaperGenerationRequest

async def main():
    ep = EasyPaper(config_path=str(Path("configs/dev.yaml").resolve()))
    
    request = PaperGenerationRequest.model_validate_json_file("metadata.json")
    metadata = request.to_metadata()
    options = request.to_generate_options()

    result = await ep.generate(metadata, **options)
    print(f"状态: {result.status}, 字数: {result.total_word_count}")

asyncio.run(main())
```

### 内联元数据

```python
import asyncio
from easypaper import EasyPaper, PaperMetaData

async def main():
    ep = EasyPaper(config_path="configs/dev.yaml")
    
    metadata = PaperMetaData(
        title="我的论文标题",
        idea_hypothesis="...",
        method="...",
        data="...",
        experiments="...",
        references=["@article{...}"],
    )
    
    result = await ep.generate(metadata)
    print(f"状态: {result.status}, 字数: {result.total_word_count}")

asyncio.run(main())
```

## 关键参考文件

使用 EasyPaper 时参考仓库中的以下文件：

### 命令（工作流执行）

- [`plugins/easypaper/commands/easypaper.md`](https://github.com/PinkGranite/EasyPaper/blob/master/plugins/easypaper/commands/easypaper.md) - 端到端元数据工作流合约
- [`plugins/easypaper/commands/paper-from-metadata.md`](https://github.com/PinkGranite/EasyPaper/blob/master/plugins/easypaper/commands/paper-from-metadata.md) - 直接元数据到论文生成
- [`plugins/easypaper/commands/paper-section.md`](https://github.com/PinkGranite/EasyPaper/blob/master/plugins/easypaper/commands/paper-section.md) - 单章节生成

### 技能（领域指南）

- [`plugins/easypaper/skills/setup-environment/SKILL.md`](https://github.com/PinkGranite/EasyPaper/blob/master/plugins/easypaper/skills/setup-environment/SKILL.md) - 自动环境配置（Python, LaTeX）
- [`plugins/easypaper/skills/paper-from-metadata/SKILL.md`](https://github.com/PinkGranite/EasyPaper/blob/master/plugins/easypaper/skills/paper-from-metadata/SKILL.md) - 完整论文生成工作流
- [`plugins/easypaper/skills/venue-selection/SKILL.md`](https://github.com/PinkGranite/EasyPaper/blob/master/plugins/easypaper/skills/venue-selection/SKILL.md) - 会议特定格式（NeurIPS, ICML, ICLR, ACL, AAAI, COLM, Nature）
- [`plugins/easypaper/skills/academic-writing-rules/SKILL.md`](https://github.com/PinkGranite/EasyPaper/blob/master/plugins/easypaper/skills/academic-writing-rules/SKILL.md) - 学术写作与 LaTeX 规范

### 配置与示例

- [`configs/example.yaml`](https://github.com/PinkGranite/EasyPaper/blob/master/configs/example.yaml) - 完整配置模板
- [`economist_example/metadata.json`](https://github.com/PinkGranite/EasyPaper/blob/master/economist_example/metadata.json) - 包含全部字段的元数据示例
- [`user_case/`](https://github.com/PinkGranite/EasyPaper/tree/master/user_case) - 独立使用示例
- [`README.md`](https://github.com/PinkGranite/EasyPaper/blob/master/README.md) - 主文档
- [`AGENTS.md`](https://github.com/PinkGranite/EasyPaper/blob/master/AGENTS.md) - 仓库级智能体指令

## PaperMetaData 字段

**必填项**：
- `title`, `idea_hypothesis`, `method`, `data`, `experiments`, `references`

**可选项**：
- `style_guide`（会议名称）, `target_pages`, `template_path`, `figures`, `tables`, `code_repository`, `export_prompt_traces`

完整示例参考 [`examples/meta.json`](https://github.com/PinkGranite/EasyPaper/blob/master/examples/meta.json) 和 [`economist_example/metadata.json`](https://github.com/PinkGranite/EasyPaper/blob/master/economist_example/metadata.json)。将 `examples/meta.json` 视为完整 `PaperGenerationRequest` 样本：使用 `request = PaperGenerationRequest.model_validate_json_file(...)`，然后通过 `request.to_metadata()` 和 `request.to_generate_options()` 进行 SDK 生成。

## 最终 PDF 选择

启用审阅循环时可能生成多次迭代 PDF。按以下优先级报告最终成果物：

1. `result.pdf_path`（权威最终输出）
2. `result.output_path` 下的 `iteration_*_final/**/*.pdf`
3. `result.output_path` 下最新的 `iteration_*` 目录中的 PDF
4. `result.output_path/paper.pdf`（最后回退方案）

若未找到 PDF，报告最终 PDF 不可用并包含最近编译错误。

## 流式生成

```python
from easypaper import EasyPaper, PaperMetaData, EventType

async for event in ep.generate_stream(metadata):
    if event.event_type == EventType.PHASE_START:
        print(f"▶ [{event.phase}] {event.message}")
    elif event.event_type == EventType.COMPLETE:
        result = event.data["result"]
        print(f"完成! {result['total_word_count']} 词")
```

## 使用场景

在以下情况使用本技能：
- 用户需要以编程方式生成学术论文
- 用户需要理解 EasyPaper SDK 用法
- 用户咨询论文生成工作流
- 用户需要会议特定格式指南

**详细工作流和执行合约请参考 `plugins/easypaper/` 目录中的文件。**
