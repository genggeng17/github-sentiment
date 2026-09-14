"""独立词典集入口；命令行只需依赖此处的公开接口。"""

from .config import parse_quotas
from .sampler import LexiconSampler

__all__ = ["LexiconSampler", "parse_quotas"]
