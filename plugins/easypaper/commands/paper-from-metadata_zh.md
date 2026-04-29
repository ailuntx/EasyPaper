使用 EasyPaper Python SDK 直接从元数据生成论文。

## 执行合约

1. **推荐**：让用户准备 `metadata.json`（模式参考 `examples/meta.json`），然后加载运行：
   - 解析请求：`request = PaperGenerationRequest.model_validate_json_file("metadata.json")`
   - 构建 SDK 输入：`metadata = request.to_metadata()` 和 `options = request.to_generate_options()`
   - 生成：`result = await ep.generate(metadata, **options)`
   - 确保文件中所有路径均为绝对路径（或用 `Path(...).resolve()` 转换）。

2. **使用 `paper-from-metadata` 技能**处理完整工作流：
   - 检查用户是否拥有完整元数据（文件或 JSON 对象）
   - 若缺失或不完整，则交互式收集
   - 根据 `examples/meta.json` 结构验证元数据
   - **重要**：确保元数据中所有路径为绝对路径（template_path、figures[].file_path、code_repository.path、output_dir）
   - 直接通过 EasyPaper SDK 生成论文

3. **该技能将**：
   - 导入 EasyPaper：`from easypaper import EasyPaper, PaperMetaData, PaperGenerationRequest`
   - 通过配置初始化：`ep = EasyPaper(config_path="...")`（配置路径需为绝对路径）
   - 优先通过 `PaperGenerationRequest.model_validate_json_file(path)` 从文件加载，再使用 `to_metadata()` + `to_generate_options()`；否则构建或转换为 `PaperMetaData`（所有路径需绝对）
   - 生成：`result = await ep.generate(metadata, **options)`
   - PDF 编译路径：优先使用进程内排版器；若对等节点不可用则回退至 HTTP 排版器端点
   - 用绝对文件路径和摘要报告结果
   - 按以下优先级选择最终 PDF：
     1. `result.pdf_path`
     2. `result.output_path/iteration_*_final/**/*.pdf`
     3. 最新的 `result.output_path/iteration_*` 目录 PDF
     4. `result.output_path/paper.pdf`
   - 若未找到 PDF，报告最终 PDF 不可用并附编译错误摘要

## 路径要求

**所有路径必须是绝对路径**：
- `template_path`：LaTeX 模板的绝对路径
- `figures[].file_path`：图形文件的绝对路径
- `code_repository.path`：绝对路径（若类型为 local_dir）
- `output_dir`：输出目录的绝对路径
- `config_path`：配置文件绝对路径

若用户提供相对路径，请在使用前通过 `pathlib.Path.resolve()` 转换为绝对路径。

## 回退行为

- **缺失必填字段**：技能自动交互式收集缺失信息
- **元数据格式无效**：显示验证错误并引导用户更正格式（参考 `examples/meta.json`）
- **检测到相对路径**：自动转换为绝对路径并通知用户
- **未安装包**：首先使用 `setup-environment` 技能
- **配置缺失**：向用户询问配置路径或使用默认值（确保为绝对路径）

$ARGUMENTS
