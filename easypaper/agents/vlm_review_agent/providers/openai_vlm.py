"""
OpenAI VLM Provider
- **Description**:
    - Implementation for GPT-4V / GPT-4o
    - Supports image analysis for PDF page review
"""
import json
import logging
from typing import Optional

from ...shared.llm_client import LLMClient

from .base import VLMProvider
from ..models import VLMResponse


logger = logging.getLogger("uvicorn.error")


class OpenAIVLM(VLMProvider):
    """
    OpenAI Vision Language Model Provider
    
    - **Description**:
        - Uses GPT-4V or GPT-4o for image analysis
        - Supports high-detail image analysis for PDF pages
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        base_url: Optional[str] = None,
        max_tokens: int = 2000,
        temperature: float = 0.1,
        **kwargs
    ):
        """
        Initialize OpenAI VLM provider
        
        Args:
            api_key: OpenAI API key
            model: Model name (gpt-4o, gpt-4-vision-preview, etc.)
            base_url: Optional custom base URL (for compatible APIs)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (low for consistent analysis)
            **kwargs: Additional options
        """
        super().__init__(api_key=api_key, model=model, **kwargs)
        
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Initialize client
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        
        self.client = LLMClient(**client_kwargs)
    
    @property
    def provider_name(self) -> str:
        return "openai"
    
    async def analyze_page(
        self,
        image_data: bytes,
        prompt: str,
        detail: str = "high",
        **kwargs
    ) -> VLMResponse:
        """
        Analyze a page image using GPT-4V/4o
        
        Args:
            image_data: PNG/JPEG image bytes
            prompt: Analysis prompt
            detail: Image detail level ('low', 'high', 'auto')
            **kwargs: Additional options
            
        Returns:
            VLMResponse with analysis result
        """
        try:
            # Encode image to base64
            image_b64 = self.encode_image_base64(image_data)
            media_type = self.get_image_media_type(image_data)
            
            # Build message with image
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{image_b64}",
                                "detail": detail,
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        }
                    ]
                }
            ]
            
            # Call API
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            
            # Extract response
            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if response.usage else 0
            
            # Try to parse JSON if response contains it
            parsed_content = self._try_parse_json(content)
            
            return VLMResponse(
                success=True,
                content=parsed_content if parsed_content else content,
                raw_response=content,
                tokens_used=tokens_used,
            )
        
        except Exception as e:
            logger.error(f"OpenAI VLM error: {e}")
            return VLMResponse(
                success=False,
                error=str(e),
            )
    
    def _try_parse_json(self, content: str) -> Optional[str]:
        """Try to extract and parse JSON from response"""
        if not content:
            return None
        
        # Try direct parse
        try:
            json.loads(content)
            return content
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from markdown code block
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
        
        # Try to extract JSON from plain code block
        if "```" in content:
            start = content.find("```") + 3
            # Skip language identifier if present
            newline = content.find("\n", start)
            if newline > start:
                start = newline + 1
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
VLMFactory.register("openai", OpenAIVLM)
