#!/usr/bin/env python3
"""
Models package for poetry generation with Transformer architecture.
Provides convenient imports for all model components.
"""

from .config import config, Config
from .dataset import (
    PoetryDataset,
    AcrosticDataset, 
    MixedDataset,
    create_dataloaders
)
from .model import (
    SimplifiedPoetryTransformer,
    RhythmicPositionalEncoding,
    MultiHeadSelfAttention,
    FeedForwardNetwork,
    TransformerBlock,
    create_poetry_transformer
)

__all__ = [
    # Configuration
    'config',
    'Config',
    
    # Dataset classes
    'PoetryDataset',
    'AcrosticDataset',
    'MixedDataset',
    'create_dataloaders',
    
    # Model classes
    'SimplifiedPoetryTransformer',
    'RhythmicPositionalEncoding',
    'MultiHeadSelfAttention',
    'FeedForwardNetwork',
    'TransformerBlock',
    'create_poetry_transformer'
] 