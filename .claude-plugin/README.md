# easypaper

AI-powered academic paper generation plugin for Claude Code.

## Description

Generate LaTeX academic papers from metadata interactively. Users can input paper details through an interactive conversation, and easypaper will generate a complete paper with proper structure, formatting, and references.

## Installation

```bash
# Install easypaper Python package
pip install easypaper

# Or install from source
pip install -e .
```

## Prerequisites

- Python 3.11+
- `easypaper` package installed
- LaTeX toolchain (pdflatex + bibtex) for PDF compilation
- API key for LLM provider (configured via config file)

## Usage

After installation, invoke the command:

```
/easypaper
```

This will guide you through an interactive workflow to generate your academic paper.

## Configuration

Create a YAML config file (see `configs/example.yaml` in the project) with your API keys:

```yaml
agents:
  - name: metadata
    model:
      model_name: claude-sonnet-4-20250514
      api_key: YOUR_API_KEY
      base_url: https://api.anthropic.com/v1
  # ... other agents
```

Then provide the config path when prompted.

## Supported Venues

- NeurIPS
- ICML
- ICLR
- ACL
- AAAI
- COLM
- Nature
