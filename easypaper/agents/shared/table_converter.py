"""
Table Converter - Convert any readable format to LaTeX tables
- **Description**:
    - Converts CSV, Markdown, plain text tables to LaTeX
    - Uses LLM for intelligent format detection and conversion
    - Handles special characters and academic table formatting
"""
import logging
import os
from typing import Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..metadata_agent.models import TableSpec

logger = logging.getLogger("uvicorn.error")


# Prompt for table conversion
TABLE_CONVERSION_PROMPT = """You are an expert LaTeX typesetter. Convert the following table data into a properly formatted LaTeX table.

## Table Information
- **Label**: {label}
- **Caption**: {caption}

## Table Data (in any format)
```
{content}
```

## Requirements
1. Generate a complete LaTeX table environment with \\begin{{table}}...\\end{{table}}
2. Use \\centering for the table
3. Use booktabs style (\\toprule, \\midrule, \\bottomrule)
4. Include the caption and label
5. Use appropriate column alignment (l for text, c for short items, r for numbers)
6. If numbers represent best results, make them bold with \\textbf{{}}
7. Handle any special characters that need escaping in LaTeX
8. Use [t] or [h] for table placement

## Output
Output ONLY the LaTeX code, no explanations or markdown code blocks.
"""


async def convert_table_to_latex(
    table: "TableSpec",
    llm_client: Any,
    model_name: str,
    base_path: Optional[str] = None,
) -> Optional[str]:
    """
    Convert a TableSpec to LaTeX format using LLM.
    
    - **Args**:
        - `table` (TableSpec): Table specification
        - `llm_client`: OpenAI-compatible async client
        - `model_name` (str): Model to use for conversion
        - `base_path` (str, optional): Base path for resolving file_path
        
    - **Returns**:
        - `str`: Complete LaTeX table code, or None if conversion fails
    """
    # Check for auto_generate (not yet implemented)
    if table.auto_generate:
        logger.warning(
            "table_converter.auto_generate_not_implemented id=%s",
            table.id
        )
        return None
    
    # Get content from file or inline
    content = None
    
    if table.file_path:
        # Resolve file path
        if base_path and not os.path.isabs(table.file_path):
            file_path = os.path.join(base_path, table.file_path)
        else:
            file_path = table.file_path
        
        # Read file content
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                logger.info("table_converter.read_file path=%s", file_path)
            else:
                logger.warning(
                    "table_converter.file_not_found path=%s",
                    file_path
                )
        except Exception as e:
            logger.error(
                "table_converter.file_read_error path=%s error=%s",
                file_path, str(e)
            )
    else:
        content = table.content
    
    if not content:
        logger.warning("table_converter.no_content id=%s", table.id)
        return None
    
    # Build prompt
    prompt = TABLE_CONVERSION_PROMPT.format(
        label=table.id,
        caption=table.caption,
        content=content.strip(),
    )
    
    try:
        # Call LLM for conversion
        response = await llm_client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert LaTeX typesetter specializing in academic tables."
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,  # Lower temperature for consistent formatting
            max_tokens=2000,
        )
        
        latex_content = response.choices[0].message.content.strip()
        
        # Clean up any markdown code block markers
        if latex_content.startswith("```"):
            lines = latex_content.split('\n')
            # Remove first and last line if they are code block markers
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            latex_content = '\n'.join(lines)
        
        logger.info(
            "table_converter.success id=%s length=%d",
            table.id, len(latex_content)
        )
        
        return latex_content
        
    except Exception as e:
        logger.error(
            "table_converter.llm_error id=%s error=%s",
            table.id, str(e)
        )
        return None


async def convert_tables(
    tables: list,
    llm_client: Any,
    model_name: str,
    base_path: Optional[str] = None,
) -> dict:
    """
    Convert multiple tables to LaTeX.
    
    - **Args**:
        - `tables` (List[TableSpec]): List of table specifications
        - `llm_client`: OpenAI-compatible async client
        - `model_name` (str): Model to use
        - `base_path` (str, optional): Base path for file resolution
        
    - **Returns**:
        - `dict`: Mapping of table_id to LaTeX code
    """
    converted = {}
    
    for table in tables:
        latex = await convert_table_to_latex(
            table=table,
            llm_client=llm_client,
            model_name=model_name,
            base_path=base_path,
        )
        if latex:
            converted[table.id] = latex
    
    logger.info(
        "table_converter.batch_complete total=%d converted=%d",
        len(tables), len(converted)
    )
    
    return converted
