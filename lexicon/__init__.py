"""词典模块入口；采样代码与内置 rules/ 词典统一放在本包。"""

from .config import parse_quotas
from .sampler import LexiconSampler

__all__ = ["LexiconSampler", "parse_quotas"]
