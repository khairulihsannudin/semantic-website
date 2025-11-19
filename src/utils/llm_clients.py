"""
LLM Clients Module
Provides unified interface for different LLM providers
"""

from typing import Dict, Optional
import os

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


class LLMClientFactory:
    """Factory for creating LLM clients."""
    
    @staticmethod
    def create_client(provider: str, api_key: Optional[str] = None):
        """
        Create an LLM client for the specified provider.
        
        Args:
            provider: LLM provider name ('openai' or 'anthropic')
            api_key: API key (if None, reads from environment)
            
        Returns:
            LLM client instance
        """
        if provider.lower() == 'openai':
            if not OPENAI_AVAILABLE:
                raise ValueError("OpenAI library not installed. Install with: pip install openai")
            key = api_key or os.getenv('OPENAI_API_KEY')
            if not key:
                raise ValueError("OpenAI API key not provided")
            return OpenAI(api_key=key)
        
        elif provider.lower() == 'anthropic':
            if not ANTHROPIC_AVAILABLE:
                raise ValueError("Anthropic library not installed. Install with: pip install anthropic")
            key = api_key or os.getenv('ANTHROPIC_API_KEY')
            if not key:
                raise ValueError("Anthropic API key not provided")
            return Anthropic(api_key=key)
        
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    @staticmethod
    def get_available_models(provider: str) -> list:
        """
        Get list of available models for a provider.
        
        Args:
            provider: LLM provider name
            
        Returns:
            List of model names
        """
        models = {
            'openai': [
                'gpt-4',
                'gpt-4-turbo-preview',
                'gpt-3.5-turbo',
                'gpt-3.5-turbo-16k'
            ],
            'anthropic': [
                'claude-3-opus-20240229',
                'claude-3-sonnet-20240229',
                'claude-3-haiku-20240307',
                'claude-2.1'
            ]
        }
        return models.get(provider.lower(), [])
