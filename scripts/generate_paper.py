#!/usr/bin/env python3
"""
CLI Tool for MetaData-based Paper Generation

This script calls the MetaData Agent API to generate papers.
Requires the easypaper server to be running on port 8000.

Usage:
    # Start the server first:
    uv run uvicorn easypaper.main:app --reload --port 8000
    
    # Then run the CLI:
    python scripts/generate_paper.py --input examples/transkg_metadata.json
    
    # Or with inline arguments:
    python scripts/generate_paper.py \
        --title "My Paper Title" \
        --idea "Research hypothesis..." \
        --method "Method description..." \
        --data "Data description..." \
        --experiments "Experiment results..."
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

try:
    import httpx
except ImportError:
    print("Error: httpx is required. Install with: pip install httpx")
    sys.exit(1)


DEFAULT_API_URL = "http://localhost:8000"


def load_metadata_from_file(filepath: str) -> dict:
    """Load PaperMetaData from JSON file."""
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)


def create_metadata_from_args(args) -> dict:
    """Create PaperMetaData dict from command line arguments"""
    references = []
    if args.refs_file:
        with open(args.refs_file, encoding="utf-8") as f:
            content = f.read()
            if content.strip().startswith("["):
                references = json.loads(content)
            else:
                # Split by @article, @inproceedings, etc.
                import re
                entries = re.split(r'\n(?=@)', content)
                references = [e.strip() for e in entries if e.strip()]
    
    return {
        "title": args.title or "Untitled Paper",
        "idea_hypothesis": args.idea or "",
        "method": args.method or "",
        "data": args.data or "",
        "experiments": args.experiments or "",
        "references": references,
        "template_path": args.template,
        "style_guide": args.style_guide,
    }


async def check_server_health(api_url: str) -> bool:
    """Check if the agentsys server is running"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{api_url}/healthz", timeout=5.0)
            return response.status_code == 200
    except Exception:
        return False


async def generate_paper(api_url: str, metadata: dict, output_dir: str = None) -> dict:
    """Call the MetaData Agent API to generate a paper"""
    request_data = {
        **metadata,
        "save_output": True,
    }
    if output_dir:
        request_data["output_dir"] = output_dir
    
    # Extended timeout for paper generation with review loop (10 minutes)
    async with httpx.AsyncClient(timeout=600.0) as client:
        response = await client.post(
            f"{api_url}/metadata/generate",
            json=request_data,
        )
        response.raise_for_status()
        return response.json()


async def main():
    parser = argparse.ArgumentParser(
        description="Generate academic paper from MetaData using AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # First, start the easypaper server:
  uv run uvicorn easypaper.main:app --reload --port 8000
  
  # Then generate from JSON file:
  python scripts/generate_paper.py --input examples/transkg_metadata.json
  
  # Or from command line arguments:
  python scripts/generate_paper.py \\
    --title "TransKG: Knowledge Graph Completion" \\
    --idea "Transformers can better capture semantic relationships..." \\
    --method "We propose TransKG combining BERT with attention..." \\
    --data "Evaluated on FB15k-237, WN18RR datasets..." \\
    --experiments "Achieved 0.391 MRR, outperforming baselines..."
""",
    )
    
    # Server options
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help=f"API base URL (default: {DEFAULT_API_URL})",
    )
    
    # Input options
    input_group = parser.add_argument_group("Input Options")
    input_group.add_argument(
        "--input", "-i",
        help="Path to JSON file containing PaperMetaData",
    )
    input_group.add_argument(
        "--title", "-t",
        help="Paper title",
    )
    input_group.add_argument(
        "--idea",
        help="Research idea or hypothesis",
    )
    input_group.add_argument(
        "--method",
        help="Method description",
    )
    input_group.add_argument(
        "--data",
        help="Data or validation method description",
    )
    input_group.add_argument(
        "--experiments",
        help="Experiments, results, and findings",
    )
    input_group.add_argument(
        "--refs-file",
        help="Path to file containing BibTeX references",
    )
    
    # Template and compilation options
    template_group = parser.add_argument_group("Template and Compilation")
    template_group.add_argument(
        "--template",
        help="Path to .zip template file for PDF compilation",
    )
    template_group.add_argument(
        "--style-guide",
        help="Writing style guide (e.g., 'ICML', 'NeurIPS')",
    )
    template_group.add_argument(
        "--compile-pdf",
        action="store_true",
        default=True,
        help="Compile to PDF (requires --template)",
    )
    template_group.add_argument(
        "--no-pdf",
        action="store_true",
        help="Skip PDF compilation, only generate LaTeX",
    )
    template_group.add_argument(
        "--figures-dir",
        help="Directory containing figure files",
    )
    template_group.add_argument(
        "--target-pages",
        type=int,
        help="Target page count (default: uses venue standard, e.g., 8 for ICML)",
    )
    
    # Planning options
    planning_group = parser.add_argument_group("Planning Options")
    planning_group.add_argument(
        "--enable-planning",
        action="store_true",
        default=True,
        help="Enable planning phase for structure and word budgets (default: True)",
    )
    planning_group.add_argument(
        "--no-planning",
        action="store_true",
        help="Disable planning phase, use default structure",
    )
    
    # Review options
    review_group = parser.add_argument_group("Review Options")
    review_group.add_argument(
        "--enable-review",
        action="store_true",
        default=True,
        help="Enable review loop for word count optimization (default: True)",
    )
    review_group.add_argument(
        "--no-review",
        action="store_true",
        help="Disable review loop",
    )
    review_group.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Maximum review iterations (default: 3)",
    )
    
    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--output", "-o",
        help="Output directory path",
    )
    
    args = parser.parse_args()
    
    # Get API URL
    api_url = args.api_url
    
    # Validate input
    if not args.input and not (args.idea and args.method):
        parser.error("Either --input file or --idea and --method are required")
    
    # Check server health
    print("Checking server connection...")
    if not await check_server_health(api_url):
        print(f"\nError: Cannot connect to easypaper server at {api_url}")
        print("\nPlease start the server first:")
        print("  uv run uvicorn easypaper.main:app --reload --port 8000")
        return 1
    print("Server connected.\n")
    
    # Load or create metadata
    if args.input:
        print(f"Loading metadata from: {args.input}")
        metadata = load_metadata_from_file(args.input)
    else:
        metadata = create_metadata_from_args(args)
    
    # Override template and compile options from CLI if provided
    if args.template:
        metadata["template_path"] = args.template
    if args.style_guide:
        metadata["style_guide"] = args.style_guide
    if args.figures_dir:
        metadata["figures_source_dir"] = args.figures_dir
    if args.target_pages:
        metadata["target_pages"] = args.target_pages
    
    # Set compile_pdf: CLI --no-pdf flag overrides, otherwise use JSON value
    if args.no_pdf:
        metadata["compile_pdf"] = False
    else:
        metadata.setdefault("compile_pdf", True)
    compile_pdf = metadata["compile_pdf"]
    
    # Set planning: CLI --no-planning flag overrides, otherwise use JSON value
    if args.no_planning:
        metadata["enable_planning"] = False
    else:
        metadata.setdefault("enable_planning", True)
    enable_planning = metadata["enable_planning"]
    
    # Set review options: CLI --no-review flag overrides, otherwise use JSON value
    if args.no_review:
        metadata["enable_review"] = False
    else:
        metadata.setdefault("enable_review", True)
    enable_review = metadata["enable_review"]
    
    # max_review_iterations: JSON value takes priority, CLI is fallback
    metadata.setdefault("max_review_iterations", args.max_iterations)
    max_iterations = metadata["max_review_iterations"]
    
    print(f"{'='*60}")
    print(f"Paper: {metadata.get('title', 'Untitled')}")
    print(f"{'='*60}")
    print(f"Idea/Hypothesis: {metadata.get('idea_hypothesis', '')[:100]}...")
    print(f"Method: {metadata.get('method', '')[:100]}...")
    print(f"Data: {metadata.get('data', '')[:100]}...")
    print(f"Experiments: {metadata.get('experiments', '')[:100]}...")
    print(f"References: {len(metadata.get('references', []))} entries")
    if metadata.get("code_repository"):
        repo_type = metadata.get("code_repository", {}).get("type", "unknown")
        print(f"Code Repository: enabled ({repo_type})")
    if metadata.get('template_path'):
        print(f"Template: {metadata.get('template_path')}")
    if metadata.get('style_guide'):
        print(f"Style Guide: {metadata.get('style_guide')}")
    if metadata.get('target_pages'):
        print(f"Target Pages: {metadata.get('target_pages')}")
    print(f"Compile PDF: {compile_pdf}")
    print(f"Planning: {enable_planning}")
    print(f"Review Loop: {enable_review} (max {max_iterations} iterations)")
    print(f"{'='*60}\n")
    
    # Generate paper
    print("Starting paper generation...")
    if enable_planning:
        print("  Phase 0: Planning (Structure, Word Budgets)")
    print("  Phase 1: Introduction (Leader)")
    print("  Phase 2: Body Sections (Method, Experiment, Results, Related Work)")
    print("  Phase 3: Synthesis (Abstract, Conclusion)")
    if enable_review:
        print("  Phase 3.5: Review Loop (Word Count Optimization)")
    if compile_pdf and metadata.get('template_path'):
        print("  Phase 4: PDF Compilation (via Typesetter Agent)")
    print()
    
    try:
        result = await generate_paper(api_url, metadata, args.output)
    except httpx.HTTPStatusError as e:
        print(f"\nError: API request failed with status {e.response.status_code}")
        print(f"Response: {e.response.text}")
        return 1
    except httpx.ReadTimeout:
        print(f"\nError: Request timed out. Paper generation with review loop can take several minutes.")
        print("Try running with --no-review to disable the review loop, or check server logs.")
        return 1
    except httpx.ConnectError:
        print(f"\nError: Could not connect to server at {api_url}")
        print("Please make sure the agentsys server is running.")
        return 1
    except Exception as e:
        error_msg = str(e) if str(e) else type(e).__name__
        print(f"\nError: {error_msg}")
        return 1
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Generation Complete!")
    print(f"{'='*60}")
    print(f"Status: {result.get('status', 'unknown')}")
    
    total_words = result.get('total_word_count', 0)
    target_words = result.get('target_word_count')
    review_iterations = result.get('review_iterations', 0)
    
    if target_words:
        print(f"Word Count: {total_words} / {target_words} target")
    else:
        print(f"Total Word Count: {total_words}")
    
    if review_iterations > 0:
        print(f"Review Iterations: {review_iterations}")
    
    print(f"\nSections Generated:")
    for section in result.get('sections', []):
        status_icon = "✓" if section.get('status') == "ok" else "✗"
        print(f"  {status_icon} {section.get('section_type', 'unknown')}: {section.get('word_count', 0)} words")
    
    if result.get('errors'):
        print(f"\nErrors:")
        for error in result['errors']:
            print(f"  - {error}")
    
    if result.get('output_path'):
        print(f"\nOutput saved to: {result['output_path']}")
        print(f"  - main.tex")
        print(f"  - references.bib")
        print(f"  - metadata.json")
        if result.get('pdf_path'):
            print(f"  - {Path(result['pdf_path']).name} (PDF)")
    
    if result.get('pdf_path'):
        print(f"\nPDF generated: {result['pdf_path']}")
    
    return 0 if result.get('status') != "error" else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
