"""
VLM Provider Base Classes
- **Description**:
    - Abstract base class for VLM providers
    - Factory for creating VLM instances based on configuration
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import base64
import logging

from ..models import VLMResponse


logger = logging.getLogger("uvicorn.error")


class VLMProvider(ABC):
    """
    Abstract base class for Vision Language Model providers
    
    - **Description**:
        - Defines interface for VLM providers (OpenAI, Claude, Qwen, etc.)
        - Supports single page and batch page analysis
    """
    
    def __init__(
        self,
        api_key: str,
        model: str,
        **kwargs
    ):
        """
        Initialize VLM provider
        
        Args:
            api_key: API key for the provider
            model: Model name to use
            **kwargs: Additional provider-specific options
        """
        self.api_key = api_key
        self.model = model
        self.options = kwargs
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name (e.g., 'openai', 'claude', 'qwen')"""
        pass
    
    @abstractmethod
    async def analyze_page(
        self,
        image_data: bytes,
        prompt: str,
        **kwargs
    ) -> VLMResponse:
        """
        Analyze a single page image
        
        Args:
            image_data: Image bytes (PNG/JPEG)
            prompt: Analysis prompt
            **kwargs: Additional options
            
        Returns:
            VLMResponse with analysis result
        """
        pass
    
    async def analyze_pages(
        self,
        images: List[bytes],
        prompts: List[str],
        **kwargs
    ) -> List[VLMResponse]:
        """
        Analyze multiple pages (default: sequential processing)
        
        Args:
            images: List of image bytes
            prompts: List of prompts (one per image, or single prompt for all)
            **kwargs: Additional options
            
        Returns:
            List of VLMResponse objects
        """
        responses = []
        
        # If single prompt provided, use it for all images
        if len(prompts) == 1:
            prompts = prompts * len(images)
        
        for i, (image, prompt) in enumerate(zip(images, prompts)):
            try:
                response = await self.analyze_page(image, prompt, **kwargs)
                responses.append(response)
            except Exception as e:
                logger.error(f"VLM analysis failed for page {i+1}: {e}")
                responses.append(VLMResponse(
                    success=False,
                    error=str(e),
                ))
        
        return responses
    
    @staticmethod
    def encode_image_base64(image_data: bytes) -> str:
        """Encode image bytes to base64 string"""
        return base64.b64encode(image_data).decode('utf-8')
    
    def get_image_media_type(self, image_data: bytes) -> str:
        """Detect image media type from bytes"""
        # Check magic bytes
        if image_data[:8] == b'\x89PNG\r\n\x1a\n':
            return "image/png"
        elif image_data[:2] == b'\xff\xd8':
            return "image/jpeg"
        elif image_data[:4] == b'GIF8':
            return "image/gif"
        elif image_data[:4] == b'RIFF' and image_data[8:12] == b'WEBP':
            return "image/webp"
        else:
            # Default to PNG
            return "image/png"


class VLMFactory:
    """
    Factory for creating VLM provider instances
    
    - **Description**:
        - Creates appropriate VLM provider based on configuration
        - Supports lazy loading of provider implementations
    """
    
    _providers: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str, provider_class: type):
        """
        Register a VLM provider class
        
        Args:
            name: Provider name (e.g., 'openai', 'claude')
            provider_class: VLMProvider subclass
        """
        cls._providers[name.lower()] = provider_class
    
    @classmethod
    def create(
        cls,
        provider: str,
        api_key: str,
        model: str,
        **kwargs
    ) -> VLMProvider:
        """
        Create a VLM provider instance
        
        Args:
            provider: Provider name (openai, claude, qwen)
            api_key: API key
            model: Model name
            **kwargs: Additional options
            
        Returns:
            VLMProvider instance
            
        Raises:
            ValueError: If provider not found
        """
        provider_lower = provider.lower()
        
        # Lazy load providers if not registered
        if provider_lower not in cls._providers:
            cls._load_provider(provider_lower)
        
        if provider_lower not in cls._providers:
            available = list(cls._providers.keys())
            raise ValueError(
                f"Unknown VLM provider: {provider}. "
                f"Available: {available}"
            )
        
        provider_class = cls._providers[provider_lower]
        return provider_class(api_key=api_key, model=model, **kwargs)
    
    @classmethod
    def _load_provider(cls, provider: str):
        """Lazy load provider implementation"""
        try:
            if provider == "openai":
                from .openai_vlm import OpenAIVLM
                cls.register("openai", OpenAIVLM)
            elif provider == "claude":
                from .claude_vlm import ClaudeVLM
                cls.register("claude", ClaudeVLM)
            elif provider == "qwen":
                from .qwen_vlm import QwenVLM
                cls.register("qwen", QwenVLM)
        except ImportError as e:
            logger.warning(f"Failed to load VLM provider '{provider}': {e}")
    
    @classmethod
    def available_providers(cls) -> List[str]:
        """List available provider names"""
        # Try to load all known providers
        for p in ["openai", "claude", "qwen"]:
            try:
                cls._load_provider(p)
            except Exception:
                pass
        return list(cls._providers.keys())
