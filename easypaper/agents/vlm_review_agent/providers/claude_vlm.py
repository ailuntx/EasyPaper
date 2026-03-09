"""
Claude VLM Provider
- **Description**:
    - Implementation for Claude 3.5 Sonnet Vision
    - Supports image analysis for PDF page review
"""
import json
import logging
from typing import Optional

from .base import VLMProvider
from ..models import VLMResponse


logger = logging.getLogger("uvicorn.error")


class ClaudeVLM(VLMProvider):
    """
    Anthropic Claude Vision Language Model Provider
    
    - **Description**:
        - Uses Claude 3.5 Sonnet for image analysis
        - Strong reasoning capabilities for layout analysis
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-5-sonnet-20241022",
        max_tokens: int = 2000,
        temperature: float = 0.1,
        **kwargs
    ):
        """
        Initialize Claude VLM provider
        
        Args:
            api_key: Anthropic API key
            model: Model name
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            **kwargs: Additional options
        """
        super().__init__(api_key=api_key, model=model, **kwargs)
        
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Lazy import anthropic
        try:
            import anthropic
            self.client = anthropic.AsyncAnthropic(api_key=api_key)
        except ImportError:
            raise ImportError("anthropic package required. Install with: pip install anthropic")
    
    @property
    def provider_name(self) -> str:
        return "claude"
    
    async def analyze_page(
        self,
        image_data: bytes,
        prompt: str,
        **kwargs
    ) -> VLMResponse:
        """
        Analyze a page image using Claude Vision
        
        Args:
            image_data: PNG/JPEG image bytes
            prompt: Analysis prompt
            **kwargs: Additional options
            
        Returns:
            VLMResponse with analysis result
        """
        try:
            # Encode image to base64
            image_b64 = self.encode_image_base64(image_data)
            media_type = self.get_image_media_type(image_data)
            
            # Build message with image
            message = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_b64,
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt,
                            }
                        ]
                    }
                ],
            )
            
            # Extract response
            content = message.content[0].text if message.content else ""
            tokens_used = message.usage.input_tokens + message.usage.output_tokens
            
            # Try to parse JSON
            parsed_content = self._try_parse_json(content)
            
            return VLMResponse(
                success=True,
                content=parsed_content if parsed_content else content,
                raw_response=content,
                tokens_used=tokens_used,
            )
        
        except Exception as e:
            logger.error(f"Claude VLM error: {e}")
            return VLMResponse(
                success=False,
                error=str(e),
            )
    
    def _try_parse_json(self, content: str) -> Optional[str]:
        """Try to extract and parse JSON from response"""
        if not content:
            return None
        
        try:
            json.loads(content)
            return content
        except json.JSONDecodeError:
            pass
        
        # Extract from code blocks
        if "```json" in content:
            start = content.find("```json") + 7
            end = content.find("```", start)
            if end > start:
                json_str = content[start:end].strip()
                try:
                    json.loads(json_str)
                    return json_str
                except json.JSONDecodeError:
                    pass
        
        return None


# Register with factory
from .base import VLMFactory
VLMFactory.register("claude", ClaudeVLM)
