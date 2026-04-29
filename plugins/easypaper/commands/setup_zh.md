设置EasyPaper环境，包括Python依赖项和LaTeX工具链。

## 执行合同

1. 使用`setup-environment`技能：
   - 创建隔离的虚拟环境（优先使用`uv`，备选`venv`）
   - 安装easypaper包
   - 检查LaTeX安装情况，缺失时提供安装说明
   - 验证所有组件是否正常工作

2. 设置完成后提供明确说明：
   - 如何激活环境
   - 如何将EasyPaper作为Python SDK使用：`from easypaper import EasyPaper, PaperMetaData`
   - 使用插件的后续步骤（需准备包含API密钥的配置文件）

$ARGUMENTS
