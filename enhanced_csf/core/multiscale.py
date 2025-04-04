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
def _compute_multiscale_height_diff_numba(points: np.ndarray, cloth_points: np.ndarray, 
                                        indices: np.ndarray, time_step: float, 
                                        rigidness: int, scale_weights: np.ndarray) -> np.ndarray:
    """使用Numba加速的多尺度高度差计算"""
    n_cloth = len(cloth_points)
    updated_cloth = np.zeros_like(cloth_points)
    
    for i in prange(n_cloth):
        # 找到对应的点云点
        idx = indices[i]
        
        # 计算高度差
        z_diff = points[idx, 2] - cloth_points[i, 2]
        
        # 根据尺度权重调整更新步长
        scale_factor = np.sum(scale_weights[idx])
        
        # 更新布料点高度
        update = z_diff * time_step * scale_factor / rigidness
        updated_cloth[i] = cloth_points[i]
        updated_cloth[i, 2] = cloth_points[i, 2] + update
        
        # 确保布料点不会低于点云点
        min_height = points[idx, 2] - 0.1
        max_height = points[idx, 2] + 0.1
        updated_cloth[i, 2] = np.clip(updated_cloth[i, 2], min_height, max_height)
    
    return updated_cloth

@jit(nopython=True)
def _compute_scale_weights_numba(points: np.ndarray, neighbors: np.ndarray, 
                               k: int, scales: np.ndarray) -> np.ndarray:
    """使用Numba加速的尺度权重计算"""
    n_points = len(points)
    n_scales = len(scales)
    scale_weights = np.zeros((n_points, n_scales))
    
    for i in prange(n_points):
        # 计算局部平面
        center = np.mean(neighbors[i], axis=0)
        centered = neighbors[i] - center
        
        # 使用SVD计算法向量
        U, S, Vh = np.linalg.svd(centered)
        normal = Vh[2]
        
        # 计算局部曲率
        curvature = S[2] / (S[0] + 1e-6)
        
        # 根据曲率和尺度计算权重
        for j in range(n_scales):
            scale = scales[j]
            # 使用高斯核计算权重
            scale_weights[i, j] = np.exp(-curvature * scale)
    
    # 归一化权重
    row_sums = scale_weights.sum(axis=1, keepdims=True)
    scale_weights = scale_weights / (row_sums + 1e-6)
    
    return scale_weights

class MultiscaleCSF(CSF):
    """多尺度CSF算法实现"""
    
    def __init__(self,
                 base_resolution=2,
                 time_step=0.65,
                 class_threshold=0.5,
                 iterations=500,
                 rigidness=3,
                 n_scales=3):
        """
        初始化多尺度CSF算法参数
        
        Parameters
        ----------
        base_resolution : float
            基础布料分辨率
        time_step : float
            模拟时间步长
        class_threshold : float
            分类阈值
        iterations : int
            最大迭代次数
        rigidness : int
            布料刚性参数
        n_scales : int
            尺度数量
        """
        super().__init__(base_resolution, time_step, class_threshold,
                        iterations, rigidness)
        self.n_scales = n_scales
        
    def _create_pyramid(self, points):
        """创建点云金字塔"""
        pyramid = [points]
        current_points = points.copy()
        
        for i in range(self.n_scales - 1):
            # 降采样
            indices = np.random.choice(
                len(current_points),
                size=len(current_points)//2,
                replace=False
            )
            current_points = current_points[indices]
            pyramid.append(current_points)
            
        return pyramid
    
    def _classify_scale(self, points, scale):
        """在单个尺度上进行分类"""
        # 调整当前尺度的参数
        resolution = self.cloth_resolution * (2 ** scale)
        time_step = self.time_step * (0.8 ** scale)
        rigidness = min(5, self.rigidness + scale)
        
        # 创建临时CSF实例
        csf = CSF(
            cloth_resolution=resolution,
            time_step=time_step,
            class_threshold=self.class_threshold,
            iterations=self.iterations,
            rigidness=rigidness
        )
        
        return csf.classify(points)
    
    def _upsample_labels(self, labels, original_points, scale_points):
        """将标签上采样到原始点云分辨率"""
        tree = cKDTree(scale_points[:, :2])
        distances, indices = tree.query(original_points[:, :2])
        return labels[indices]
    
    def classify(self, points):
        """
        使用多尺度方法对点云进行分类
        
        Parameters
        ----------
        points : np.ndarray
            输入点云数据，shape为(n_points, 3)
            
        Returns
        -------
        labels : np.ndarray
            分类标签，1表示地面点，0表示非地面点
        """
        # 创建点云金字塔
        pyramid = self._create_pyramid(points)
        
        # 在每个尺度上进行分类
        scale_labels = []
        for i, scale_points in enumerate(pyramid):
            labels = self._classify_scale(scale_points, i)
            if i > 0:
                # 上采样到原始分辨率
                labels = self._upsample_labels(labels, points, scale_points)
            scale_labels.append(labels)
        
        # 融合多尺度结果
        final_labels = np.zeros(len(points))
        weights = np.array([1.0, 0.7, 0.4])[:self.n_scales]  # 权重随尺度减小
        weights = weights / weights.sum()
        
        for i, labels in enumerate(scale_labels):
            final_labels += weights[i] * labels
            
        # 根据加权结果进行最终分类
        return (final_labels > 0.5).astype(int)
        
    def classify_with_progress(self, points, progress_callback=None):
        """
        使用多尺度方法对点云进行分类，支持进度回调
        
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
        # 创建点云金字塔
        if progress_callback is not None:
            progress_callback(0.1)  # 金字塔创建占总进度的10%
            
        pyramid = self._create_pyramid(points)
        
        # 在每个尺度上进行分类
        scale_labels = []
        n_scales = len(pyramid)
        
        for i, scale_points in enumerate(pyramid):
            # 调用父类的classify_with_progress方法
            if progress_callback is not None:
                # 创建一个新的回调函数，将进度映射到当前尺度的范围
                def scale_callback(progress):
                    scale_start = 0.1 + i * 0.8 / n_scales
                    scale_end = 0.1 + (i + 1) * 0.8 / n_scales
                    adjusted_progress = scale_start + progress * (scale_end - scale_start)
                    progress_callback(adjusted_progress)
            else:
                scale_callback = None
                
            labels = self._classify_scale(scale_points, i)
            if i > 0:
                # 上采样到原始分辨率
                labels = self._upsample_labels(labels, points, scale_points)
            scale_labels.append(labels)
        
        # 融合多尺度结果
        if progress_callback is not None:
            progress_callback(0.9)  # 开始融合结果
            
        final_labels = np.zeros(len(points))
        weights = np.array([1.0, 0.7, 0.4])[:self.n_scales]  # 权重随尺度减小
        weights = weights / weights.sum()
        
        for i, labels in enumerate(scale_labels):
            final_labels += weights[i] * labels
            
        # 根据加权结果进行最终分类
        if progress_callback is not None:
            progress_callback(1.0)  # 完成
            
        return (final_labels > 0.5).astype(int) 