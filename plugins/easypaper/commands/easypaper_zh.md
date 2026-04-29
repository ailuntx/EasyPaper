运行 EasyPaper 端到端论文生成工作流，并提供引导式设置和元数据收集。

## 执行契约

### 阶段 1：环境设置（仅首次）

1. **检查环境是否已设置**：
   - 检查 `.easypaper-env` 目录是否存在
   - 检查 `easypaper` 包是否可导入：`python -c "import easypaper"`
   - 检查 `pdflatex` 命令是否可用

2. **如果环境尚未准备就绪**：
   - 使用 `setup-environment` 技能自动执行以下操作：
     - 创建独立的虚拟环境（优先使用 `uv`，回退到 `venv`）
     - 安装 easypaper 包
     - 检查并指导 LaTeX 安装
     - 验证所有组件是否正常工作

### 阶段 2：论文生成

3. **使用 `paper-from-metadata` 技能，该技能处理**：
   - **检查现有元数据**：询问用户是否拥有完整的元数据文件/JSON
   - **按需收集元数据**：如果缺少或不完整，则以交互方式收集所有必填字段：
     - 必填：`title`、`idea_hypothesis`、`method`、`data`、`experiments`、`references`
     - 可选：`style_guide`、`target_pages`、`template_path`（绝对路径）、`compile_pdf`、`enable_review`、`max_review_iterations`
     - 高级：`figures`（带绝对 `file_path`）、`tables`、`code_repository`（如果为 `local_dir` 则带绝对路径）、`output_dir`（绝对路径）
   - **路径处理**：确保所有路径都是绝对路径 - 使用 `pathlib.Path.resolve()` 将相对路径转换为绝对路径
   - **审阅并确认**：显示摘要，允许编辑，保存到文件，获取确认
   - **生成论文**：直接使用 EasyPaper Python SDK。优先将元数据文件解析为 `PaperGenerationRequest`，然后使用 `to_metadata()` + `to_generate_options()` 进行转换。
     ```python
     from easypaper import EasyPaper, PaperGenerationRequest
     from pathlib import Path
     
     # 配置文件路径应该是绝对路径
     config_path = Path("configs/openrouter.yaml").resolve()
     ep = EasyPaper(config_path=str(config_path))
     # 如果用户有 metadata.json (例如 examples/meta.json)
     request = PaperGenerationRequest.model_validate_json_file("metadata.json")
     metadata = request.to_metadata()
     options = request.to_generate_options()
     result = await ep.generate(metadata, **options)
     ```
   - **报告结果**：显示状态、输出文件、绝对路径、摘要。
   - **最终 PDF 选择规则**：
     1. 首先使用 `result.pdf_path`。
     2. 如果缺失，按顺序搜索 `result.output_path`：
        - `iteration_*_final/**/*.pdf`
        - 最新 `iteration_*` 目录中的 PDF
        - 根目录下的 `paper.pdf`
     3. 如果仍然缺失，报告最终 PDF 不可用，并包含编译错误摘要。

## 路径要求

**重要提示**：元数据中的所有路径都必须是绝对路径：
- `template_path`：LaTeX 模板文件/目录的绝对路径
- `figures[].file_path`：图片文件的绝对路径
- `code_repository.path`：绝对路径（如果类型是 `local_dir`）
- `output_dir`：输出目录的绝对路径
- `config_path`：EasyPaper 配置文件的绝对路径

该技能将自动把相对路径转换为绝对路径，但应鼓励用户提供绝对路径。

## 用户体验指南

- **首次用户**：无需询问即可自动触发环境设置
- **清晰的进度**：显示当前正在进行的步骤（例如，“步骤 1/2：正在设置环境...”）
- **错误处理**：如果任何步骤失败，请清晰解释并提供后续步骤
- **灵活性**：允许用户提供完整的元数据或以交互方式收集
- **路径转换**：自动将相对路径转换为绝对路径并通知用户
- **参考**：当用户询问结构时，将 `examples/meta.json` 作为模板引用（注意：路径应为绝对路径）
- **直接导入**：将 EasyPaper 用作 Python SDK - 生成不需要 API 服务器。
- **排版器路径**：PDF 编译优先使用进程内 Typesetter（自包含 SDK），当对等方不可用时回退到 HTTP Typesetter 端点。
