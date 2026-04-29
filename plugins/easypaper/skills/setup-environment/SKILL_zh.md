---
description: 自动在隔离环境中设置 EasyPaper 环境，包括 Python 依赖项和 LaTeX 工具链。
---

当插件首次安装或需要环境设置时使用此功能。

## 设置流程

### 1. 检查 Python 环境

- 检测是否可用 `uv`（首选）或 `python`/`python3`
- 若 `uv` 可用，则使用它创建隔离的虚拟环境
- 若不可用，则检查 `venv` 或 `virtualenv` 并创建虚拟环境
- 环境应创建在项目本地目录（例如 `.venv` 或 `.easypaper-env`）

### 2. 安装 EasyPaper 包

- 若 `uv` 可用：
  ```bash
  uv venv .easypaper-env
  source .easypaper-env/bin/activate  # Windows 系统使用 .easypaper-env/Scripts/activate
  uv pip install -e .  # 若在仓库根目录
  # 或
  uv pip install easypaper[server]  # 若从 PyPI 安装
  ```
- 若使用标准 Python：
  ```bash
  python -m venv .easypaper-env
  source .easypaper-env/bin/activate
  pip install -e .  # 或 pip install easypaper[server]
  ```

### 3. 检查 LaTeX 安装

- 检查 LaTeX 发行版：
  - macOS：检查 `pdflatex` 命令，建议 `brew install --cask mactex` 或 `brew install basictex`
  - Linux：检查 `pdflatex`，建议 `sudo apt-get install texlive-full` 或 `texlive-base`
  - Windows：检查 `pdflatex`，建议安装 MiKTeX 或 TeX Live
- 通过运行验证安装：`pdflatex --version`
- 若未安装，提供针对用户操作系统的清晰安装指南

### 4. 验证设置

- 测试 EasyPaper 导入：`python -c "import easypaper; from easypaper import EasyPaper, PaperMetaData; print('EasyPaper 导入成功')"`
- 测试 LaTeX：`pdflatex --version`

### 5. 创建环境激活脚本

创建辅助脚本（`.easypaper-activate.sh` 或 `.easypaper-activate.bat`）用于：
- 激活虚拟环境
- 按需设置 PATH
- 提供使用 EasyPaper 作为 Python SDK 的说明

## 错误处理

- 若 Python 版本 < 3.11，通知用户并建议升级
- 若包安装失败，显示错误并建议手动安装
- 若 LaTeX 缺失，提供操作系统专属安装命令
- 每步执行前均需验证

## 安装后说明

成功设置后告知用户：
1. 环境已就绪于 `.easypaper-env`
2. 激活方式：`source .easypaper-env/bin/activate`（Windows 使用等效命令）
3. 现可直接将 EasyPaper 作为 Python SDK 使用：`from easypaper import EasyPaper, PaperMetaData`
4. 该环境为隔离环境，不影响系统 Python
5. 确保拥有含 API 密钥的配置文件（YAML） - 参考 `configs/example.yaml`
