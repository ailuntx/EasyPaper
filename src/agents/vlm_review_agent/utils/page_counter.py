"""
Page Counter Utility
- **Description**:
    - System-level page counting using pypdf/pymupdf
    - Fast page count without full rendering
"""
import logging
import subprocess
from pathlib import Path
from typing import Optional, Tuple, Union

logger = logging.getLogger("uvicorn.error")


class PageCounter:
    """
    PDF Page Counter
    
    - **Description**:
        - Fast page counting using multiple backends
        - Provides basic PDF metadata extraction
    """
    
    def __init__(self):
        """Initialize page counter with available backend"""
        self._backend = self._detect_backend()
    
    def _detect_backend(self) -> str:
        """Detect available backend for page counting"""
        # Try pymupdf first
        try:
            import fitz
            return "pymupdf"
        except ImportError:
            pass
        
        # Try pypdf
        try:
            from pypdf import PdfReader
            return "pypdf"
        except ImportError:
            pass
        
        # Try pdfinfo command
        try:
            result = subprocess.run(
                ["pdfinfo", "--version"],
                capture_output=True,
                timeout=5
            )
            if result.returncode == 0:
                return "pdfinfo"
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
        
        raise ImportError(
            "No PDF backend available. "
            "Install 'pymupdf' or 'pypdf'."
        )
    
    def count_pages(self, pdf_path: Union[str, Path]) -> int:
        """
        Count total pages in PDF
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Number of pages
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        if self._backend == "pymupdf":
            return self._count_pymupdf(pdf_path)
        elif self._backend == "pypdf":
            return self._count_pypdf(pdf_path)
        else:
            return self._count_pdfinfo(pdf_path)
    
    def _count_pymupdf(self, pdf_path: Path) -> int:
        """Count pages using PyMuPDF"""
        import fitz
        doc = fitz.open(str(pdf_path))
        count = len(doc)
        doc.close()
        return count
    
    def _count_pypdf(self, pdf_path: Path) -> int:
        """Count pages using pypdf"""
        from pypdf import PdfReader
        reader = PdfReader(str(pdf_path))
        return len(reader.pages)
    
    def _count_pdfinfo(self, pdf_path: Path) -> int:
        """Count pages using pdfinfo command"""
        result = subprocess.run(
            ["pdfinfo", str(pdf_path)],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"pdfinfo failed: {result.stderr}")
        
        for line in result.stdout.split("\n"):
            if line.startswith("Pages:"):
                return int(line.split(":")[1].strip())
        
        raise RuntimeError("Could not parse page count from pdfinfo")
    
    def get_pdf_info(
        self, 
        pdf_path: Union[str, Path]
    ) -> dict:
        """
        Get basic PDF information
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dict with pdf info (pages, width, height, title, etc.)
        """
        pdf_path = Path(pdf_path)
        
        info = {
            "path": str(pdf_path),
            "pages": 0,
            "width": 0.0,
            "height": 0.0,
            "title": "",
            "creator": "",
        }
        
        if self._backend == "pymupdf":
            import fitz
            doc = fitz.open(str(pdf_path))
            try:
                info["pages"] = len(doc)
                if len(doc) > 0:
                    page = doc[0]
                    info["width"] = page.rect.width
                    info["height"] = page.rect.height
                metadata = doc.metadata
                info["title"] = metadata.get("title", "")
                info["creator"] = metadata.get("creator", "")
            finally:
                doc.close()
        
        elif self._backend == "pypdf":
            from pypdf import PdfReader
            reader = PdfReader(str(pdf_path))
            info["pages"] = len(reader.pages)
            if len(reader.pages) > 0:
                box = reader.pages[0].mediabox
                info["width"] = float(box.width)
                info["height"] = float(box.height)
            if reader.metadata:
                info["title"] = reader.metadata.get("/Title", "")
                info["creator"] = reader.metadata.get("/Creator", "")
        
        return info
    
    def estimate_content_pages(
        self,
        pdf_path: Union[str, Path],
        references_start_marker: str = "References",
    ) -> Tuple[int, int]:
        """
        Estimate content pages vs reference pages
        
        This is a heuristic that tries to detect where the references
        section starts by looking at text content.
        
        Args:
            pdf_path: Path to PDF file
            references_start_marker: Text marker for references section
            
        Returns:
            Tuple of (content_pages, total_pages)
        """
        pdf_path = Path(pdf_path)
        total_pages = self.count_pages(pdf_path)
        
        # For now, assume references take ~1 page for every 20 citations
        # This is a rough heuristic
        # TODO: Use VLM to detect actual references page
        
        # Simple heuristic: last 1-2 pages are usually references
        if total_pages <= 4:
            content_pages = total_pages
        elif total_pages <= 8:
            content_pages = total_pages - 1
        else:
            content_pages = total_pages - 2
        
        return content_pages, total_pages
    
    def check_overflow(
        self,
        pdf_path: Union[str, Path],
        page_limit: int,
    ) -> Tuple[bool, int]:
        """
        Quick check for page overflow
        
        Args:
            pdf_path: Path to PDF file
            page_limit: Maximum allowed pages
            
        Returns:
            Tuple of (is_overflow, overflow_count)
        """
        total_pages = self.count_pages(pdf_path)
        is_overflow = total_pages > page_limit
        overflow_count = max(0, total_pages - page_limit)
        
        return is_overflow, overflow_count
