"""特征提取模块

该模块提供了点云特征提取和融合的功能，包括：

1. 局部特征提取
2. 特征融合
3. 特征重要性分析
"""

from .local import LocalFeatures
from .fusion import FeatureFusion

__all__ = [
    'LocalFeatures',
    'FeatureFusion'
] 