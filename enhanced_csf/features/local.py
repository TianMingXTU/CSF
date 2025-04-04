import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from scipy.spatial import cKDTree
from scipy.linalg import svd
from sklearn.decomposition import PCA
import logging
from tqdm import tqdm
from numba import jit, prange
from functools import lru_cache

class LocalFeatures:
    """
    局部特征计算类，提供多种点云局部特征提取方法
    
    支持的特征包括：
    - 平面度
    - 曲率
    - 法向量一致性
    - 局部密度
    - 局部高度变化
    - 局部方向性
    """
    
    def __init__(self, 
                 k_neighbors: int = 10,
                 use_numba: bool = True,
                 cache_size: int = 128):
        """
        初始化局部特征计算器
        
        Parameters
        ----------
        k_neighbors : int, optional
            用于计算局部特征的近邻点数量
        use_numba : bool, optional
            是否使用Numba加速
        cache_size : int, optional
            LRU缓存大小
        """
        self.k_neighbors = k_neighbors
        self.use_numba = use_numba
        self.cache_size = cache_size
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
    @staticmethod
    @jit(nopython=True, parallel=True)
    def _compute_planarity_numba(points: np.ndarray) -> float:
        """
        使用Numba加速计算局部平面度
        
        Parameters
        ----------
        points : np.ndarray
            局部点云数据
            
        Returns
        -------
        planarity : float
            平面度值
        """
        # 计算协方差矩阵
        centered = points - np.mean(points, axis=0)
        cov = np.dot(centered.T, centered) / (len(points) - 1)
        
        # 计算特征值
        eigenvalues = np.linalg.eigvalsh(cov)
        
        # 平面度 = (λ2 - λ1) / λ3
        return (eigenvalues[1] - eigenvalues[0]) / eigenvalues[2]
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def _compute_curvature_numba(points: np.ndarray) -> float:
        """
        使用Numba加速计算局部曲率
        
        Parameters
        ----------
        points : np.ndarray
            局部点云数据
            
        Returns
        -------
        curvature : float
            曲率值
        """
        # 计算协方差矩阵
        centered = points - np.mean(points, axis=0)
        cov = np.dot(centered.T, centered) / (len(points) - 1)
        
        # 计算特征值
        eigenvalues = np.linalg.eigvalsh(cov)
        
        # 曲率 = λ1 / (λ1 + λ2 + λ3)
        return eigenvalues[0] / np.sum(eigenvalues)
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def _compute_normal_consistency_numba(points: np.ndarray) -> float:
        """
        使用Numba加速计算法向量一致性
        
        Parameters
        ----------
        points : np.ndarray
            局部点云数据
            
        Returns
        -------
        consistency : float
            法向量一致性值
        """
        # 计算协方差矩阵
        centered = points - np.mean(points, axis=0)
        cov = np.dot(centered.T, centered) / (len(points) - 1)
        
        # 计算特征值和特征向量
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # 获取法向量（最小特征值对应的特征向量）
        normal = eigenvectors[:, 0]
        
        # 计算法向量与垂直方向的一致性
        return np.abs(normal[2])
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def _compute_local_density_numba(points: np.ndarray, 
                                   query_point: np.ndarray, 
                                   radius: float) -> float:
        """
        使用Numba加速计算局部密度
        
        Parameters
        ----------
        points : np.ndarray
            局部点云数据
        query_point : np.ndarray
            查询点
        radius : float
            搜索半径
            
        Returns
        -------
        density : float
            局部密度值
        """
        # 计算点到查询点的距离
        distances = np.sqrt(np.sum((points - query_point) ** 2, axis=1))
        
        # 计算半径内的点数量
        count = np.sum(distances <= radius)
        
        # 计算密度（点数量 / 球体积）
        volume = (4/3) * np.pi * radius ** 3
        return count / volume
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def _compute_height_variation_numba(points: np.ndarray) -> float:
        """
        使用Numba加速计算局部高度变化
        
        Parameters
        ----------
        points : np.ndarray
            局部点云数据
            
        Returns
        -------
        variation : float
            高度变化值
        """
        # 提取z坐标
        z_coords = points[:, 2]
        
        # 计算高度变化（标准差）
        return np.std(z_coords)
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def _compute_directionality_numba(points: np.ndarray) -> float:
        """
        使用Numba加速计算局部方向性
        
        Parameters
        ----------
        points : np.ndarray
            局部点云数据
            
        Returns
        -------
        directionality : float
            方向性值
        """
        # 计算协方差矩阵
        centered = points - np.mean(points, axis=0)
        cov = np.dot(centered.T, centered) / (len(points) - 1)
        
        # 计算特征值
        eigenvalues = np.linalg.eigvalsh(cov)
        
        # 方向性 = (λ3 - λ2) / λ3
        return (eigenvalues[2] - eigenvalues[1]) / eigenvalues[2]
    
    def compute_features(self, points: np.ndarray) -> Dict[str, np.ndarray]:
        """
        计算点云的局部特征
        
        Parameters
        ----------
        points : np.ndarray
            点云数据，shape为(n_points, 3)
            
        Returns
        -------
        features : Dict[str, np.ndarray]
            局部特征字典
        """
        self.logger.info("开始计算局部特征...")
        
        n_points = len(points)
        features = {
            'planarity': np.zeros(n_points),
            'curvature': np.zeros(n_points),
            'normal_consistency': np.zeros(n_points),
            'local_density': np.zeros(n_points),
            'height_variation': np.zeros(n_points),
            'directionality': np.zeros(n_points)
        }
        
        # 使用KD树进行近邻搜索
        tree = cKDTree(points)
        
        # 计算每个点的局部特征
        for i in tqdm(range(n_points), desc="计算局部特征"):
            # 找到k个最近邻
            distances, indices = tree.query(points[i], k=self.k_neighbors)
            neighbor_points = points[indices]
            
            # 计算平面度
            features['planarity'][i] = self._compute_planarity_numba(neighbor_points)
            
            # 计算曲率
            features['curvature'][i] = self._compute_curvature_numba(neighbor_points)
            
            # 计算法向量一致性
            features['normal_consistency'][i] = self._compute_normal_consistency_numba(neighbor_points)
            
            # 计算局部密度
            features['local_density'][i] = self._compute_local_density_numba(neighbor_points, points[i], np.mean(distances))
            
            # 计算高度变化
            features['height_variation'][i] = self._compute_height_variation_numba(neighbor_points)
            
            # 计算方向性
            features['directionality'][i] = self._compute_directionality_numba(neighbor_points)
        
        self.logger.info("局部特征计算完成")
        return features
    
    def compute_pca_features(self, points: np.ndarray) -> Dict[str, np.ndarray]:
        """
        使用PCA计算点云特征
        
        Parameters
        ----------
        points : np.ndarray
            点云数据，shape为(n_points, 3)
            
        Returns
        -------
        features : Dict[str, np.ndarray]
            PCA特征字典
        """
        self.logger.info("开始计算PCA特征...")
        
        # 使用PCA进行降维
        pca = PCA(n_components=3)
        pca.fit(points)
        
        # 计算特征
        features = {
            'pca_components': pca.components_,
            'pca_explained_variance': pca.explained_variance_,
            'pca_explained_variance_ratio': pca.explained_variance_ratio_,
            'pca_singular_values': pca.singular_values_
        }
        
        self.logger.info("PCA特征计算完成")
        return features
    
    def compute_all_features(self, points: np.ndarray) -> Dict[str, np.ndarray]:
        """
        计算所有可用的点云特征
        
        Parameters
        ----------
        points : np.ndarray
            输入点云数据
            
        Returns
        -------
        features : Dict[str, np.ndarray]
            所有特征的字典
        """
        # 计算局部特征
        local_features = self.compute_features(points)
        
        # 计算PCA特征
        pca_features = self.compute_pca_features(points)
        
        # 合并所有特征
        all_features = {**local_features}
        for key, value in pca_features.items():
            all_features[f'pca_{key}'] = value
            
        return all_features 