通过 EasyPaper 生成或修改单个论文章节。

## 必填输入

- `section_type`（可选值：`introduction`, `method`, `experiment`, `result`, `related_work`, `discussion`, `abstract`, `conclusion`）
- 包含论文核心字段的 `metadata` 对象

## 可选输入

- 用于综合章节（`abstract`, `conclusion`）的 `prior_sections`
- `style_guide`

## 操作

向 `POST /metadata/generate/section` 发送请求并返回：

- 章节状态
- 生成的 LaTeX 内容
- 字数统计及引用说明（若可用）

$ARGUMENTS
