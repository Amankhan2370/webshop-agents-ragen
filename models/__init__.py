"""
Models module for WebShop-WebArena RAGEN
Contains neural network architectures
"""

from .transformer import TransformerLM, WebNavigationModel

__all__ = [
    'TransformerLM',
    'WebNavigationModel'
]