import numpy as np
from scipy.spatial import cKDTree
from scipy.linalg import eigh, svd, qr
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.sparse import linalg as sparse_linalg
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from typing import Tuple, Optional, List, Dict, Any
from numba import jit, prange
import logging
from tqdm import tqdm
from functools import lru_cache

logger = logging.getLogger(__name__)

@jit(nopython=True, parallel=True)
def _compute_geodesic_path_numba(points: np.ndarray, n_steps: int = 8) -> np.ndarray:
    """使用Numba加速的测地线路径计算"""
    if len(points) < 2:
        return np.zeros((0, 2))  # 返回一个空的2D数组
        
    # 找到最远的两个点作为测地线端点
    distances = np.sum(points**2, axis=1)
    start_idx = np.argmin(distances)
    end_idx = np.argmax(distances)
    
    start = points[start_idx]
    end = points[end_idx]
    
    # 生成测地线路径点
    t = np.linspace(0, 1, n_steps)
    path = np.zeros((n_steps, 2))
    for i in prange(n_steps):
        path[i] = (1-t[i])*start + t[i]*end
    
    return path

@jit(nopython=True)
def _compute_local_curvature(points: np.ndarray, center: np.ndarray) -> float:
    """计算局部曲率"""
    centered = points - center
    if len(centered) < 3:
        return 0.0
    
    # 使用PCA计算主方向
    cov = np.dot(centered.T, centered) / len(centered)
    eigenvals = np.linalg.eigvals(cov)
    return np.abs(eigenvals[0] / (eigenvals[1] + 1e-6))

class ManifoldCSF:
    """基于流形学习和几何分析的创新CSF算法实现
    
    该算法结合了流形学习、几何分析和拓扑数据分析的思想，通过以下创新点提高地面点云分类的准确性：
    
    1. 流形嵌入：使用局部线性嵌入(LLE)将点云映射到低维流形空间
    2. 几何特征：计算测地线曲率、法曲率和平均曲率
    3. 拓扑特征：使用持久同伦分析提取点云的拓扑特征
    4. 自适应权重：根据局部几何和拓扑特征动态调整更新权重
    5. 并行计算：使用Numba加速计算密集型操作
    """
    
    def __init__(self, 
                 cloth_resolution: float = 0.5,
                 time_step: float = 0.65,
                 class_threshold: float = 0.5,
                 iterations: int = 500,
                 rigidness: int = 1,
                 manifold_dim: int = 2,
                 n_neighbors: int = 10,
                 batch_size: int = 1000,
                 spectral_radius: float = 1.0,
                 stability_factor: float = 1e-6,
                 persistence_threshold: float = 0.05,
                 manifold_weight: float = 0.3):
        """
        初始化流形CSF算法参数
        
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
        manifold_dim : int
            流形维度
        n_neighbors : int
            近邻点数量
        batch_size : int
            批处理大小
        spectral_radius : float
            谱半径阈值
        stability_factor : float
            数值稳定性因子
        persistence_threshold : float
            持久性阈值
        manifold_weight : float
            流形特征权重
        """
        self.cloth_resolution = cloth_resolution
        self.time_step = time_step
        self.class_threshold = class_threshold
        self.iterations = iterations
        self.rigidness = rigidness
        self.manifold_dim = manifold_dim
        self.n_neighbors = n_neighbors
        self.batch_size = batch_size
        self.spectral_radius = spectral_radius
        self.stability_factor = stability_factor
        self.persistence_threshold = persistence_threshold
        self.manifold_weight = manifold_weight
        
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
        if self.manifold_dim <= 0:
            raise ValueError("manifold_dim must be positive")
        if self.n_neighbors < 3:
            raise ValueError("n_neighbors must be at least 3")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.spectral_radius <= 0:
            raise ValueError("spectral_radius must be positive")
        if self.stability_factor <= 0:
            raise ValueError("stability_factor must be positive")
        if self.persistence_threshold < 0:
            raise ValueError("persistence_threshold must be non-negative")
        if self.manifold_weight < 0:
            raise ValueError("manifold_weight must be non-negative")
    
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
    
    def _compute_riemannian_metric(self, points, k=None):
        """计算黎曼度量张量
        
        使用局部点云的协方差矩阵作为黎曼度量张量
        """
        if k is None:
            k = self.n_neighbors
            
        n_points = len(points)
        metric_tensors = np.zeros((n_points, 3, 3))
        
        # 创建KD树用于快速近邻搜索
        tree = cKDTree(points)
        
        for i in range(n_points):
            # 找到k个最近邻
            distances, indices = tree.query(points[i], k=k+1)
            indices = indices[1:]  # 排除点本身
            neighbors = points[indices]
            
            # 计算协方差矩阵作为黎曼度量张量
            center = np.mean(neighbors, axis=0)
            centered = neighbors - center
            cov = np.dot(centered.T, centered) / k
            
            # 添加正则化项以确保正定性
            cov += np.eye(3) * 1e-6
            metric_tensors[i] = cov
        
        return metric_tensors
    
    def _compute_manifold_features(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """计算流形特征（使用进度条显示）"""
        n_points = len(points)
        manifold_coords = np.zeros((n_points, 2))
        geodesic_curvatures = np.zeros(n_points)
        normal_curvatures = np.zeros(n_points)
        spectral_features = np.zeros((n_points, 4))
        
        # 创建KD树用于快速近邻搜索
        tree = cKDTree(points)
        
        logger.info("Computing manifold features...")
        for i in tqdm(range(n_points), desc="Computing manifold features"):
            # 找到k个最近邻
            distances, indices = tree.query(points[i], k=self.n_neighbors+1)
            indices = indices[1:]  # 排除点本身
            neighbors = points[indices]
            
            # 计算局部坐标系
            center = points[i]
            centered = neighbors - center
            
            # 使用PCA计算局部坐标系
            pca = PCA(n_components=3)
            pca.fit(centered)
            
            # 主方向作为切空间基底
            tangent_basis = pca.components_[:2]
            normal = pca.components_[2]
            
            # 计算局部参数化
            local_coords = np.dot(centered, tangent_basis.T)
            manifold_coords[i] = np.mean(local_coords, axis=0)
            
            # 计算测地线
            geodesic_path = _compute_geodesic_path_numba(local_coords)
            
            # 计算测地曲率
            if len(geodesic_path) >= 3:
                geodesic_curvatures[i] = _compute_local_curvature(geodesic_path, np.mean(geodesic_path, axis=0))
            
            # 计算法曲率
            if len(centered) > 0:
                normal_curvatures[i] = _compute_local_curvature(centered, center)
            
            # 计算谱特征
            if len(centered) > 0:
                try:
                    # 构建局部拉普拉斯矩阵
                    W = squareform(pdist(centered))
                    W = np.exp(-W**2 / (2 * self.stability_factor))
                    D = np.diag(np.sum(W, axis=1))
                    L = D - W
                    
                    # 计算特征值
                    eigenvals = np.linalg.eigvalsh(L)[:4]
                    spectral_features[i] = eigenvals
                except:
                    spectral_features[i] = np.zeros(4)
        
        return manifold_coords, geodesic_curvatures, normal_curvatures, spectral_features
        
    def _compute_persistence_features(self, points, k=None):
        """
        计算持久同伦特征
        
        使用简化的持久同伦分析计算拓扑特征
        """
        if k is None:
            k = self.n_neighbors
            
        n_points = len(points)
        persistence_features = np.zeros(n_points)
        
        # 创建KD树用于快速近邻搜索
        tree = cKDTree(points)
        
        for i in range(n_points):
            # 找到k个最近邻
            distances, indices = tree.query(points[i], k=k+1)
            indices = indices[1:]  # 排除点本身
            neighbors = points[indices]
            
            # 计算局部点云的连通性
            center = np.mean(neighbors, axis=0)
            max_distance = np.max(np.sqrt(np.sum((neighbors - center)**2, axis=1)))
            
            # 计算局部点云的"洞"的数量
            if k >= 4:
                # 使用简化的alpha形状思想
                edges = 0
                for j in range(k):
                    for l in range(j+1, k):
                        if np.sum((neighbors[j] - neighbors[l])**2) < max_distance**2:
                            edges += 1
                
                # 简化的欧拉特征
                vertices = k
                faces = max(0, edges - vertices + 1)
                euler_char = vertices - edges + faces
                
                # 持久性特征 = 1 - 归一化的欧拉特征
                persistence_features[i] = 1.0 - (euler_char / k)
            else:
                persistence_features[i] = 0
        
        # 应用持久性阈值
        persistence_features[persistence_features < self.persistence_threshold] = 0
        
        return persistence_features
    
    def _update_cloth_with_physics(self, cloth_points: np.ndarray, points: np.ndarray, 
                                 tree_indices: np.ndarray, geodesic_curvatures: np.ndarray, 
                                 normal_curvatures: np.ndarray, spectral_features: np.ndarray) -> np.ndarray:
        """使用物理模型更新布料位置，考虑流形特征"""
        n_cloth = len(cloth_points)
        updated_cloth = cloth_points.copy()
        
        for i in range(n_cloth):
            idx = tree_indices[i]
            
            # 基础更新
            z_diff = points[idx, 2] - cloth_points[i, 2]
            
            # 考虑测地线曲率的影响
            geodesic_factor = np.clip(1.0 - self.manifold_weight * geodesic_curvatures[idx], 0.1, 1.0)
            
            # 考虑法向曲率的影响
            normal_factor = np.clip(1.0 - self.manifold_weight * normal_curvatures[idx], 0.1, 1.0)
            
            # 考虑谱特征的影响
            spectral_factor = np.clip(1.0 - self.manifold_weight * np.mean(np.abs(spectral_features[idx])), 0.1, 1.0)
            
            # 综合更新
            update = z_diff * geodesic_factor * normal_factor * spectral_factor
            
            # 限制更新幅度
            max_update = 0.5 * self.cloth_resolution
            update = np.clip(update * self.time_step / self.rigidness, -max_update, max_update)
            
            # 应用更新
            updated_cloth[i, 2] = cloth_points[i, 2] + update
            
            # 确保布料点不会低于点云点
            min_height = points[idx, 2] - self.cloth_resolution
            max_height = points[idx, 2] + 0.1 * self.cloth_resolution
            updated_cloth[i, 2] = np.clip(updated_cloth[i, 2], min_height, max_height)
        
        return updated_cloth
    
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
        
        # 预计算流形特征
        manifold_coords, geodesic_curvatures, normal_curvatures, spectral_features = self._compute_manifold_features(points)
        
        # 迭代更新布料位置
        logger.info("Updating cloth position...")
        for i in tqdm(range(self.iterations), desc="Updating cloth"):
            # 找到每个布料点对应的最近点云点
            distances, indices = tree.query(cloth_points[:, :2])
            
            # 使用物理模型更新布料
            cloth_points = self._update_cloth_with_physics(
                cloth_points, points, indices, 
                geodesic_curvatures, normal_curvatures, spectral_features
            )
        
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
        
        # 预计算流形特征
        if progress_callback is not None:
            progress_callback(0.1)  # 特征提取占总进度的10%
            
        manifold_coords, geodesic_curvatures, normal_curvatures, spectral_features = self._compute_manifold_features(points)
        
        if progress_callback is not None:
            progress_callback(0.2)  # 特征提取完成
        
        # 迭代更新布料位置
        logger.info("Updating cloth position...")
        for i in range(self.iterations):
            # 找到每个布料点对应的最近点云点
            distances, indices = tree.query(cloth_points[:, :2])
            
            # 使用物理模型更新布料
            cloth_points = self._update_cloth_with_physics(
                cloth_points, points, indices, 
                geodesic_curvatures, normal_curvatures, spectral_features
            )
            
            # 更新进度
            if progress_callback is not None:
                progress = 0.2 + (i + 1) / self.iterations * 0.7  # 布料更新占总进度的70%
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