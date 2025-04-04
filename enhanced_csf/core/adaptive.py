import numpy as np
from scipy.spatial import cKDTree
from typing import Tuple, Optional, List, Dict, Any
from numba import jit, prange
import logging
from tqdm import tqdm
from functools import lru_cache
from .csf import CSF

logger = logging.getLogger(__name__)

@jit(nopython=True)
def _compute_adaptive_height_diff_numba(points: np.ndarray, cloth_points: np.ndarray, 
                                      indices: np.ndarray, time_step: float, 
                                      rigidness: int, local_slope: np.ndarray) -> np.ndarray:
    """使用Numba加速的自适应高度差计算"""
    n_cloth = len(cloth_points)
    updated_cloth = np.zeros_like(cloth_points)
    
    for i in prange(n_cloth):
        # 找到对应的点云点
        idx = indices[i]
        
        # 计算高度差
        z_diff = points[idx, 2] - cloth_points[i, 2]
        
        # 根据局部坡度调整更新步长
        slope_factor = 1.0 / (1.0 + local_slope[idx])
        
        # 更新布料点高度
        update = z_diff * time_step * slope_factor / rigidness
        updated_cloth[i] = cloth_points[i]
        updated_cloth[i, 2] = cloth_points[i, 2] + update
        
        # 确保布料点不会低于点云点
        min_height = points[idx, 2] - 0.1
        max_height = points[idx, 2] + 0.1
        updated_cloth[i, 2] = np.clip(updated_cloth[i, 2], min_height, max_height)
    
    return updated_cloth

@jit(nopython=True)
def _compute_local_slope_numba(points: np.ndarray, neighbors: np.ndarray, 
                             k: int) -> np.ndarray:
    """使用Numba加速的局部坡度计算"""
    n_points = len(points)
    local_slope = np.zeros(n_points)
    
    for i in prange(n_points):
        # 计算局部平面
        center = np.mean(neighbors[i], axis=0)
        centered = neighbors[i] - center
        
        # 使用SVD计算法向量
        U, S, Vh = np.linalg.svd(centered)
        normal = Vh[2]
        
        # 计算坡度（法向量与z轴的夹角）
        local_slope[i] = np.abs(normal[2])
    
    return local_slope

class AdaptiveCSF(CSF):
    """具有自适应参数优化的CSF算法"""
    
    def __init__(self, 
                 cloth_resolution=2,
                 time_step=0.65,
                 class_threshold=0.5,
                 iterations=500,
                 rigidness=3,
                 slope_threshold=30):
        """
        初始化自适应CSF算法参数
        
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
        slope_threshold : float
            坡度阈值（度）
        """
        super().__init__(cloth_resolution, time_step, class_threshold, 
                        iterations, rigidness)
        self.slope_threshold = slope_threshold
        
    def _compute_local_slope(self, points, k=10):
        """计算局部坡度"""
        tree = cKDTree(points[:, :2])
        slopes = np.zeros(len(points))
        
        for i in range(len(points)):
            # 找到k个最近邻点
            distances, indices = tree.query(points[i, :2], k=k)
            
            if len(indices) < 3:  # 需要至少3个点拟合平面
                continue
                
            # 使用这些点拟合平面
            neighbors = points[indices]
            center = np.mean(neighbors, axis=0)
            
            # 计算协方差矩阵
            cov = np.cov(neighbors.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            
            # 最小特征值对应的特征向量即为法向量
            normal = eigenvectors[:, 0]
            
            # 计算坡度（与垂直方向的夹角）
            slope = np.arccos(np.abs(normal[2])) * 180 / np.pi
            slopes[i] = slope
            
        return slopes
    
    def _adjust_parameters(self, points):
        """根据局部地形特征调整参数"""
        slopes = self._compute_local_slope(points)
        
        # 根据坡度调整参数
        mean_slope = np.mean(slopes)
        
        if mean_slope > self.slope_threshold:
            # 陡坡区域：增加刚性，减小时间步长
            self.rigidness = min(5, self.rigidness + 1)
            self.time_step = max(0.3, self.time_step * 0.8)
        else:
            # 平缓区域：减小刚性，增加时间步长
            self.rigidness = max(1, self.rigidness - 1)
            self.time_step = min(1.0, self.time_step * 1.2)
            
        # 根据点云密度调整分辨率
        tree = cKDTree(points[:, :2])
        distances, _ = tree.query(points[:, :2], k=2)
        mean_distance = np.mean(distances[:, 1])
        self.cloth_resolution = max(mean_distance, self.cloth_resolution * 0.8)
        
    def classify(self, points):
        """
        对点云进行分类（带自适应参数优化）
        
        Parameters
        ----------
        points : np.ndarray
            输入点云数据，shape为(n_points, 3)
            
        Returns
        -------
        labels : np.ndarray
            分类标签，1表示地面点，0表示非地面点
        """
        # 调整参数
        self._adjust_parameters(points)
        
        # 调用父类的分类方法
        return super().classify(points)
        
    def classify_with_progress(self, points, progress_callback=None):
        """
        对点云进行分类（带自适应参数优化），支持进度回调
        
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
        # 调整参数
        if progress_callback is not None:
            progress_callback(0.1)  # 参数调整占总进度的10%
            
        self._adjust_parameters(points)
        
        # 调用父类的分类方法
        if progress_callback is not None:
            # 创建一个新的回调函数，将进度映射到0.1-1.0的范围
            def adjusted_callback(progress):
                adjusted_progress = 0.1 + progress * 0.9
                progress_callback(adjusted_progress)
        else:
            adjusted_callback = None
            
        return super().classify_with_progress(points, adjusted_callback) 