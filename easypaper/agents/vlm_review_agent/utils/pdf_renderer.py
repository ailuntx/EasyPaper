"""
PDF Renderer Utility
- **Description**:
    - Converts PDF pages to images for VLM analysis
    - Supports multiple rendering backends (pdf2image, pymupdf)
"""
import io
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

logger = logging.getLogger("uvicorn.error")


class PDFRenderer:
    """
    PDF to Image renderer
    
    - **Description**:
        - Renders PDF pages to PNG images
        - Supports configurable DPI and page range
    """
    
    def __init__(
        self,
        dpi: int = 150,
        fmt: str = "PNG",
    ):
        """
        Initialize PDF renderer
        
        Args:
            dpi: Rendering resolution (default 150 for good quality/speed balance)
            fmt: Output format (PNG recommended for VLM)
        """
        self.dpi = dpi
        self.fmt = fmt
        self._backend = self._detect_backend()
    
    def _detect_backend(self) -> str:
        """Detect available PDF rendering backend"""
        # Try pymupdf first (faster, no external dependencies)
        try:
            import fitz  # pymupdf
            return "pymupdf"
        except ImportError:
            pass
        
        # Try pdf2image (requires poppler)
        try:
            from pdf2image import convert_from_path
            return "pdf2image"
        except ImportError:
            pass
        
        raise ImportError(
            "No PDF rendering backend available. "
            "Install either 'pymupdf' or 'pdf2image' (with poppler)."
        )
    
    def render_pages(
        self,
        pdf_path: Union[str, Path],
        pages: Optional[List[int]] = None,
        first_page: Optional[int] = None,
        last_page: Optional[int] = None,
    ) -> List[bytes]:
        """
        Render PDF pages to images
        
        Args:
            pdf_path: Path to PDF file
            pages: Specific pages to render (1-indexed). If None, render all.
            first_page: First page to render (1-indexed)
            last_page: Last page to render (1-indexed)
            
        Returns:
            List of image bytes (PNG format)
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        if self._backend == "pymupdf":
            return self._render_pymupdf(pdf_path, pages, first_page, last_page)
        else:
            return self._render_pdf2image(pdf_path, pages, first_page, last_page)
    
    def _render_pymupdf(
        self,
        pdf_path: Path,
        pages: Optional[List[int]],
        first_page: Optional[int],
        last_page: Optional[int],
    ) -> List[bytes]:
        """Render using PyMuPDF (fitz)"""
        import fitz
        
        images = []
        doc = fitz.open(str(pdf_path))
        
        try:
            total_pages = len(doc)
            
            # Determine page range
            if pages is not None:
                page_indices = [p - 1 for p in pages if 0 < p <= total_pages]
            else:
                start = (first_page - 1) if first_page else 0
                end = last_page if last_page else total_pages
                page_indices = list(range(start, min(end, total_pages)))
            
            # Render each page
            zoom = self.dpi / 72  # PDF default is 72 DPI
            mat = fitz.Matrix(zoom, zoom)
            
            for page_idx in page_indices:
                page = doc[page_idx]
                pix = page.get_pixmap(matrix=mat)
                
                # Convert to PNG bytes
                img_bytes = pix.tobytes("png")
                images.append(img_bytes)
                
                logger.debug(f"Rendered page {page_idx + 1}/{total_pages}")
        
        finally:
            doc.close()
        
        return images
    
    def _render_pdf2image(
        self,
        pdf_path: Path,
        pages: Optional[List[int]],
        first_page: Optional[int],
        last_page: Optional[int],
    ) -> List[bytes]:
        """Render using pdf2image (poppler)"""
        from pdf2image import convert_from_path
        
        # Build kwargs
        kwargs = {
            "dpi": self.dpi,
            "fmt": self.fmt.lower(),
        }
        
        if first_page:
            kwargs["first_page"] = first_page
        if last_page:
            kwargs["last_page"] = last_page
        
        # Convert to PIL Images
        pil_images = convert_from_path(str(pdf_path), **kwargs)
        
        # Filter specific pages if requested
        if pages is not None:
            start_idx = (first_page - 1) if first_page else 0
            pil_images = [
                pil_images[p - 1 - start_idx] 
                for p in pages 
                if 0 <= (p - 1 - start_idx) < len(pil_images)
            ]
        
        # Convert to bytes
        images = []
        for img in pil_images:
            buffer = io.BytesIO()
            img.save(buffer, format=self.fmt)
            images.append(buffer.getvalue())
        
        return images
    
    def render_page(
        self,
        pdf_path: Union[str, Path],
        page_number: int,
    ) -> bytes:
        """
        Render a single page
        
        Args:
            pdf_path: Path to PDF file
            page_number: Page number (1-indexed)
            
        Returns:
            Image bytes (PNG format)
        """
        images = self.render_pages(pdf_path, pages=[page_number])
        if not images:
            raise ValueError(f"Failed to render page {page_number}")
        return images[0]
    
    def get_page_count(self, pdf_path: Union[str, Path]) -> int:
        """
        Get total page count of PDF
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Number of pages
        """
        pdf_path = Path(pdf_path)
        
        if self._backend == "pymupdf":
            import fitz
            doc = fitz.open(str(pdf_path))
            count = len(doc)
            doc.close()
            return count
        else:
            from pdf2image import pdfinfo_from_path
            info = pdfinfo_from_path(str(pdf_path))
            return info.get("Pages", 0)
    
    def get_page_dimensions(
        self, 
        pdf_path: Union[str, Path],
        page_number: int = 1
    ) -> Tuple[float, float]:
        """
        Get page dimensions in points (1/72 inch)
        
        Args:
            pdf_path: Path to PDF file
            page_number: Page number (1-indexed)
            
        Returns:
            Tuple of (width, height) in points
        """
        pdf_path = Path(pdf_path)
        
        if self._backend == "pymupdf":
            import fitz
            doc = fitz.open(str(pdf_path))
            try:
                page = doc[page_number - 1]
                rect = page.rect
                return (rect.width, rect.height)
            finally:
                doc.close()
        else:
            # pdf2image doesn't provide easy dimension access
            # Use pypdf as fallback
            try:
                from pypdf import PdfReader
                reader = PdfReader(str(pdf_path))
                page = reader.pages[page_number - 1]
                box = page.mediabox
                return (float(box.width), float(box.height))
            except ImportError:
                return (612.0, 792.0)  # Default letter size
