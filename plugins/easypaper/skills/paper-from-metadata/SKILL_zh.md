---
description: 使用 EasyPaper Python SDK 根据元数据生成完整学术论文。若未提供元数据，则通过交互方式收集，随后直接生成论文。
---

当用户需要根据元数据生成学术论文时使用此技能。该技能在统一工作流中同时处理元数据收集和论文生成。

**推荐方案：** 让用户准备 `metadata.json` 文件（完整模式参见 `examples/meta.json`），将其解析为 `PaperGenerationRequest` 并通过 `to_metadata()` + `to_generate_options()` 运行。完全支持类似 `examples/meta.json` 的文件。

## 工作流程

### 阶段一：检查现有元数据

1. **询问用户是否已有完整元数据**：
   - "您是否已有完整的元数据文件（JSON）？或需要我通过交互方式收集？"
   - 若用户提供元数据（文件路径或 JSON 对象），根据 `examples/meta.json` 结构进行验证
   - **重要**：验证前将元数据中所有相对路径转为绝对路径
   - 若元数据不完整，进入收集阶段补充缺失字段
   - 若未提供元数据，进入收集阶段

### 阶段二：收集元数据（如需）

若元数据缺失或不完整，逐项交互收集必填字段：

#### 必填字段（逐项收集并附示例）：

1. **标题**
   - 提示："请提供论文标题"
   - 示例："人工智能工具扩大科学家影响力但缩小科学关注范围"
   - 验证：非空字符串，10-200字符

2. **核心观点/假设**
   - 提示："论文的核心研究问题或假设是什么？描述您要探索的主要观点"
   - 示例："本研究假设AI在科学领域的应用具有双重效应：虽然AI工具提升科学家个体的生产力、引用量和职业发展，但会同时缩小科学探索的集体范围..."
   - 验证：非空，至少50字符

3. **研究方法**
   - 提示："描述您的研究方法。包括实验设计、数据收集、分析工具及所用框架"
   - 示例："本研究分析1980-2025年间生物学、医学、化学、物理学、材料科学和地质学领域的41,298,433篇论文，主要数据源自OpenAlex..."
   - 验证：非空，需描述研究方案

4. **数据**
   - 提示："使用了哪些数据源、数据集或材料？描述数据收集过程及预处理步骤"
   - 示例："主要文献计量数据来自OpenAlex，涵盖1980-2025年六个自然科学学科的41,298,433篇论文..."
   - 验证：非空，需描述数据来源

5. **实验/结果**
   - 提示："描述实验结果、发现或主要成果。包括关键指标、对比分析和解读"
   - 示例："主要发现显示个体与集体层面存在明显分歧。个体层面结果：生产力：使用AI的科学家论文发表量提升3.02倍..."
   - 验证：非空，需描述结果

6. **参考文献**
   - 提示："请以BibTeX格式或结构化引文列表提供参考文献。可逐条提供或批量粘贴"
   - 示例：
     ```json
     [
       "Wang, H. et al. Scientific discovery in the age of artificial intelligence. Nature 620, 47-60 (2023).",
       "LeCun, Y., Bengio, Y. & Hinton, G. Deep learning. Nature 521, 436-444 (2015)."
     ]
     ```
   - 验证：非空数组，建议至少3篇参考文献

#### 可选字段（含智能默认值）：

7. **样式指南（会议）**："目标会议或样式指南？选项：NeurIPS, ICML, ICLR, ACL, AAAI, COLM, Nature 或自定义"（默认："Nature"）

8. **目标页数**："目标页数是多少？"（默认：20）

9. **模板路径**：
   - 提示："是否有自定义LaTeX模板？若有，请提供模板文件/目录的**绝对路径**"
   - **重要**：必须为绝对路径。若用户提供相对路径，使用 `os.path.abspath()` 或 `pathlib.Path.resolve()` 转换
   - 示例：`/Users/username/papers/templates/nature.zip` 或 `/home/user/templates/custom.tex`
   - 验证：路径必须存在且为绝对路径
   - 默认：null

10. **编译PDF**："是否编译生成PDF？(是/否)"（默认：true）

11. **启用评审**："启用基于VLM的评审与迭代优化？(是/否)"（默认：true）

12. **最大评审迭代次数**："最大评审迭代次数（若启用评审）："（默认：3）

#### 高级选项（可选）：

13. **图表**：
   - 提示："需要插入图表吗？请为每张图提供：ID、**绝对文件路径**、标题和描述"
   - 格式：包含 `id`, `file_path`（必须绝对）, `caption`, `description` 的对象数组
   - **重要**：所有 `file_path` 必须为绝对路径。使用 `os.path.abspath()` 或 `pathlib.Path.resolve()` 转换相对路径
   - 示例：
     ```json
     [
       {
         "id": "fig:architecture",
         "file_path": "/Users/username/papers/figures/architecture.png",
         "caption": "系统架构图",
         "description": "展示整体系统设计"
       }
     ]
     ```
   - 验证：每个 `file_path` 必须为绝对路径且文件存在
   - 默认：空数组

14. **表格**：表格对象数组（默认：空数组）

15. **代码仓库**：
   - 提示："需要从仓库引入代码吗？提供类型（local_dir/git）及**绝对路径**或URL"
   - `local_dir` 类型：必须提供**绝对路径**
   - `git_repo` 类型：必须提供URL
   - **重要**：若为 `local_dir` 且用户提供相对路径，使用 `os.path.abspath()` 或 `pathlib.Path.resolve()` 转换
   - local_dir示例：
     ```json
     {
       "type": "local_dir",
       "path": "/Users/username/projects/my_code",
       "on_error": "fallback"
     }
     ```
   - git_repo示例：
     ```json
     {
       "type": "git_repo",
       "url": "https://github.com/user/repo.git",
       "ref": "main"
     }
     ```
   - 验证：local_dir路径必须为绝对路径且目录存在
   - 默认：null

16. **输出目录**：
   - 提示："论文生成后保存位置？请提供输出目录的**绝对路径**"
   - **重要**：必须为绝对路径。若用户提供相对路径，使用 `os.path.abspath()` 或 `pathlib.Path.resolve()` 转换
   - 示例：`/Users/username/papers/output/my_paper` 或 `/home/user/output/output_20250120`
   - 验证：路径必须为绝对路径（可自动创建目录）
   - 默认：`{当前工作目录}/output_{时间戳}`（转为绝对路径）

### 阶段三：审核确认

生成前：
1. 展示所有收集的元数据摘要
2. **验证所有路径为绝对路径**：检查 `template_path`, `figures[].file_path`, `code_repository.path`（若为local_dir）和 `output_dir` 均为绝对路径。自动转换发现的相对路径
3. 询问："需要修改任何字段吗？(是/否)"
4. 可选：将元数据保存至 `metadata.json`（含所有绝对路径）
5. 最终确认："准备生成论文？(是/否)"

### 阶段四：生成论文

1. **环境检查**：
   - 确保已安装easypaper包（如需使用 `setup-environment` 技能）
   - 检查配置文件是否存在或询问用户配置路径
   - **重要**：配置路径应为绝对路径。若为相对路径则转换

2. **导入并初始化EasyPaper**：
   ```python
   from easypaper import EasyPaper, PaperMetaData, PaperGenerationRequest
   from pathlib import Path
   
   # 如需则将配置路径转为绝对路径
   config_path = Path("configs/openrouter.yaml").resolve()  # 或用户提供路径
   
   # 通过配置初始化
   ep = EasyPaper(config_path=str(config_path))
   ```

3. **获取SDK输入（优先从文件加载）**：
   - **当用户有元数据文件（推荐）**：通过 `request = PaperGenerationRequest.model_validate_json_file(path)` 解析，然后：
     - `paper_metadata = request.to_metadata()`
     - `options = request.to_generate_options()`
   - 将 `template_path`, `figures[].file_path`, `code_repository.path`（若为local_dir）和 `output_dir` 转为绝对路径
   - **当元数据来自交互收集或字典**：根据收集内容构建 `PaperMetaData`，并为 `generate()` 构建运行时 `options`
   - **完全支持examples/meta.json**：将其视为完整 `PaperGenerationRequest` 示例
   - 验证必填字段完整性

4. **生成论文**：
   ```python
   from pathlib import Path
   from easypaper import PaperGenerationRequest

   request = PaperGenerationRequest.model_validate_json_file("metadata.json")
   paper_metadata = request.to_metadata()
   options = request.to_generate_options()

   # 若存在则将output_dir转为绝对路径
   if options.get("output_dir"):
       options["output_dir"] = str(Path(options["output_dir"]).resolve())

   result = await ep.generate(metadata=paper_metadata, **options)
   ```

   或使用流式处理获取进度：
   ```python
   async for event in ep.generate_stream(metadata, **options):
       print(f"{event.phase}: {event.message}")
   ```

5. **结果报告**：
   - 显示生成状态
   - 列出输出文件：`paper.tex`, `references.bib`, `paper.pdf`（若编译）
   - 提供文件绝对路径
   - 显示统计：字数、生成章节等
   - **最终PDF选择优先级**：
     1. `result.pdf_path`（权威路径）
     2. `result.output_path/iteration_*_final/**/*.pdf`
     3. 最新 `result.output_path/iteration_*` 目录中的PDF
     4. 回退到 `result.output_path/paper.pdf`
   - 若无PDF，明确报告最终PDF不可用并附编译错误

**排版器执行模式：**
- 当对等代理可用时，优先使用进程内排版器（SDK自包含）
- 若对等代理不可用，回退到HTTP排版器端点（`AGENTSYS_SELF_URL`）

## 路径处理规则

**关键要求**：元数据中所有路径必须为绝对路径。遵循以下规则：

1. **用户输入路径时**：
   - 明确要求提供绝对路径
   - 若用户提供相对路径，立即转换：
     ```python
     from pathlib import Path
     absolute_path = Path(relative_path).resolve()
     ```

2. **从文件读取元数据时**：
   - 加载JSON后，扫描所有路径字段并将相对路径转为绝对路径
   - 需检查字段：`template_path`, `figures[].file_path
