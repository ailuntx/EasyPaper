# 代理

面向在 EasyPaper 上工作的编码代理的仓库级说明。

## 项目范围

- EasyPaper 是一个元数据到论文生成系统，带 Python SDK（主要用法）和可选的 FastAPI API。
- 此仓库还发布了一个 Claude 代码插件市场。
- 可安装插件的根目录是 `plugins/easypaper/`。

## 插件/市场布局

- 市场清单文件：`.claude-plugin/marketplace.json`
- 插件清单文件：`plugins/easypaper/.claude-plugin/plugin.json`
- 插件命令：`plugins/easypaper/commands/`
- 插件技能：`plugins/easypaper/skills/`
- OpenCode/OpenClaw 配置：`.opencode/opencode.json`

## EasyPaper 工作流程

对于端到端论文生成，直接使用 Python SDK：

1.  **使用 `paper-from-metadata` 技能**（在 `plugins/easypaper/skills/paper-from-metadata/SKILL.md` 中）：
    - 检查用户是否拥有完整的元数据（文件或 JSON）
    - 如果缺失，则交互式收集（标题、想法假设、方法、数据、实验、参考文献）
    - 使用 EasyPaper SDK 生成论文：
      ```python
      from easypaper import EasyPaper, PaperMetaData
      ep = EasyPaper(config_path="configs/openrouter.yaml")
      result = await ep.generate(metadata, **options)
      ```

2.  **对于 Claude Code 插件用法**：
    - 使用 `/easypaper` 命令，该命令会自动处理环境设置和元数据收集
    - 不需要 API 服务器 - 直接使用 Python SDK

3.  **可选的 FastAPI 服务器**（用于外部集成）：
    - `uv run uvicorn easypaper.main:app --reload --port 8000`
    - 端点：`POST /metadata/generate`，`POST /metadata/generate/section`

## 必需元数据字段

- `title`
- `idea_hypothesis`
- `method`
- `data`
- `experiments`
- `references`

可选字段包括 `style_guide`、`target_pages`、`template_path`、`compile_pdf` 和审阅选项。

## 技能真理之源

- 后端 YAML 技能仍保留在 `skills/` 下，并由 Python 服务配置加载。
- Claude/OpenCode 技能提示存在于 `plugins/easypaper/skills/*/SKILL.md` 下。
- 主要技能：
    - `paper-from-metadata`：用于元数据收集和论文生成的统一技能
    - `setup-environment`：自动环境设置（Python，LaTeX）
    - `venue-selection`：特定会议格式设置
    - `academic-writing-rules`：学术写作规范

## 验证清单

- 保持市场 `source` 指向 `./plugins/easypaper`。
- 保持插件版本在 `plugins/easypaper/.claude-plugin/plugin.json` 中。
- 保持 README 安装步骤与实际市场命令语法一致。
