"""
FHE-AES Modules Package

This package contains modular components for the paper-aligned FHE-AES implementation.
"""

from .coefficient_computation import CoefficientComputer
from .lut_evaluation import LUTEvaluator
from .fhe_operations import FHEOperations
from .aes_operations import AESOperations
from .homomorphic_aes import HomomorphicAES

__all__ = [
    'CoefficientComputer',
    'LUTEvaluator',
    'FHEOperations',
    'AESOperations',
    'HomomorphicAES'
]