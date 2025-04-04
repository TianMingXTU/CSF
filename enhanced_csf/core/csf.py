import numpy as np
from scipy.spatial import cKDTree
from typing import Tuple, Optional, List, Dict, Any
from numba import jit, prange
import logging
from tqdm import tqdm
from functools import lru_cache

logger = logging.getLogger(__name__)

@jit(nopython=True)
def _compute_height_diff_numba(points: np.ndarray, cloth_points: np.ndarray, 
                             indices: np.ndarray, time_step: float, 
                             rigidness: int) -> np.ndarray:
    """使用Numba加速的高度差计算"""
    n_cloth = len(cloth_points)
    updated_cloth = np.zeros_like(cloth_points)
    
    for i in prange(n_cloth):
        # 找到对应的点云点
        idx = indices[i]
        
        # 计算高度差
        z_diff = points[idx, 2] - cloth_points[i, 2]
        
        # 更新布料点高度
        update = z_diff * time_step / rigidness
        updated_cloth[i] = cloth_points[i]
        updated_cloth[i, 2] = cloth_points[i, 2] + update
        
        # 确保布料点不会低于点云点
        min_height = points[idx, 2] - 0.1
        max_height = points[idx, 2] + 0.1
        if updated_cloth[i, 2] < min_height:
            updated_cloth[i, 2] = min_height
        elif updated_cloth[i, 2] > max_height:
            updated_cloth[i, 2] = max_height
    
    return updated_cloth

class CSF:
    """基础CSF算法实现
    
    该算法使用布料模拟方法对点云进行地面点分类，主要特点包括：
    
    1. 布料模拟：使用物理模拟方法模拟布料在点云表面的运动
    2. 自适应更新：根据点云高度动态调整布料位置
    3. 参数可调：提供多个参数用于调整算法性能
    """
    
    def __init__(self, 
                 cloth_resolution: float = 0.5,
                 time_step: float = 0.65,
                 class_threshold: float = 0.5,
                 iterations: int = 500,
                 rigidness: int = 1):
        """
        初始化CSF算法参数
        
        Parameters
        ----------
        cloth_resolution : float
            布料分辨率
        time_step : float
            模拟时间步长
        class_threshold : float
            分类阈值
        iterations : int
            最大迭代次数
        rigidness : int
            布料刚性参数
        """
        self.cloth_resolution = cloth_resolution
        self.time_step = time_step
        self.class_threshold = class_threshold
        self.iterations = iterations
        self.rigidness = rigidness
        
        # 验证参数
        self._validate_parameters()
    
    def _validate_parameters(self) -> None:
        """验证参数的有效性"""
        if self.cloth_resolution <= 0:
            raise ValueError("cloth_resolution must be positive")
        if self.time_step <= 0:
            raise ValueError("time_step must be positive")
        if self.class_threshold < 0:
            raise ValueError("class_threshold must be non-negative")
        if self.iterations <= 0:
            raise ValueError("iterations must be positive")
        if self.rigidness <= 0:
            raise ValueError("rigidness must be positive")
    
    def _create_cloth(self, points: np.ndarray) -> np.ndarray:
        """创建初始布料网格"""
        x_min, y_min = points[:, :2].min(axis=0)
        x_max, y_max = points[:, :2].max(axis=0)
        
        x = np.arange(x_min, x_max + self.cloth_resolution, self.cloth_resolution)
        y = np.arange(y_min, y_max + self.cloth_resolution, self.cloth_resolution)
        
        xx, yy = np.meshgrid(x, y)
        cloth_points = np.vstack((xx.flatten(), yy.flatten())).T
        
        # 初始化z坐标为最高点
        z = np.full(len(cloth_points), points[:, 2].max())
        cloth_points = np.column_stack((cloth_points, z))
        
        return cloth_points
    
    def _update_cloth(self, cloth_points: np.ndarray, points: np.ndarray, 
                     tree_indices: np.ndarray) -> np.ndarray:
        """更新布料位置"""
        return _compute_height_diff_numba(points, cloth_points, tree_indices, 
                                       self.time_step, self.rigidness)
    
    def classify(self, points: np.ndarray) -> np.ndarray:
        """
        对点云进行分类
        
        Parameters
        ----------
        points : np.ndarray
            输入点云数据，shape为(n_points, 3)
            
        Returns
        -------
        labels : np.ndarray
            分类标签，1表示地面点，0表示非地面点
        """
        # 创建KD树用于快速近邻搜索
        tree = cKDTree(points[:, :2])
        
        # 创建初始布料
        cloth_points = self._create_cloth(points)
        
        # 迭代更新布料位置
        logger.info("Updating cloth position...")
        for i in tqdm(range(self.iterations), desc="Updating cloth"):
            # 找到每个布料点对应的最近点云点
            distances, indices = tree.query(cloth_points[:, :2])
            
            # 更新布料位置
            cloth_points = self._update_cloth(cloth_points, points, indices)
        
        # 计算每个点与对应布料点的高度差
        cloth_tree = cKDTree(cloth_points[:, :2])
        distances, indices = cloth_tree.query(points[:, :2])
        height_diff = points[:, 2] - cloth_points[indices, 2]
        
        # 根据高度差进行分类
        labels = (height_diff <= self.class_threshold).astype(int)
        
        return labels
        
    def classify_with_progress(self, points: np.ndarray, progress_callback=None) -> np.ndarray:
        """
        对点云进行分类，支持进度回调
        
        Parameters
        ----------
        points : np.ndarray
            输入点云数据，shape为(n_points, 3)
        progress_callback : callable, optional
            进度回调函数，接收一个0-1之间的浮点数表示进度
            
        Returns
        -------
        labels : np.ndarray
            分类标签，1表示地面点，0表示非地面点
        """
        # 创建KD树用于快速近邻搜索
        tree = cKDTree(points[:, :2])
        
        # 创建初始布料
        cloth_points = self._create_cloth(points)
        
        # 迭代更新布料位置
        logger.info("Updating cloth position...")
        for i in range(self.iterations):
            # 找到每个布料点对应的最近点云点
            distances, indices = tree.query(cloth_points[:, :2])
            
            # 更新布料位置
            cloth_points = self._update_cloth(cloth_points, points, indices)
            
            # 更新进度
            if progress_callback is not None:
                progress = (i + 1) / self.iterations * 0.9  # 布料更新占总进度的90%
                progress_callback(progress)
        
        # 计算每个点与对应布料点的高度差
        cloth_tree = cKDTree(cloth_points[:, :2])
        distances, indices = cloth_tree.query(points[:, :2])
        height_diff = points[:, 2] - cloth_points[indices, 2]
        
        # 根据高度差进行分类
        labels = (height_diff <= self.class_threshold).astype(int)
        
        # 更新最终进度
        if progress_callback is not None:
            progress_callback(1.0)
        
        return labels 