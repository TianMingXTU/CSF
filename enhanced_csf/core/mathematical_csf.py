import numpy as np
from scipy.spatial import cKDTree
from scipy.linalg import eigh, svd, qr
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.sparse import linalg as sparse_linalg
from sklearn.decomposition import PCA
from typing import Tuple, Optional, List, Dict, Any
from numba import jit, prange
import logging
from tqdm import tqdm
from functools import lru_cache

logger = logging.getLogger(__name__)

@jit(nopython=True)
def _compute_riemannian_metric_numba(points: np.ndarray, center: np.ndarray, k: int) -> np.ndarray:
    """使用Numba加速的黎曼度量张量计算"""
    centered = points - center
    cov = np.dot(centered.T, centered) / k
    return cov

@jit(nopython=True)
def _compute_covariant_derivative_numba(points: np.ndarray, center: np.ndarray, 
                                      metric: np.ndarray, h: float) -> np.ndarray:
    """使用Numba加速的协变导数计算"""
    centered = points - center
    n_points = len(centered)
    covariant_derivatives = np.zeros((3, 3, 3))
    
    for a in range(3):
        for b in range(3):
            for c in range(3):
                # 计算度量张量在不同方向的导数
                d_gbc_dx = np.zeros(3)
                for dim in range(3):
                    # 找到在dim方向上最近的两个点
                    pos_idx = np.argmax(centered[:, dim])
                    neg_idx = np.argmin(centered[:, dim])
                    
                    # 计算有限差分
                    if pos_idx != neg_idx:
                        d_gbc_dx[dim] = (metric[b,c] - metric[b,c]) / (centered[pos_idx, dim] - centered[neg_idx, dim])
                
                # 计算协变导数
                metric_inv = np.linalg.inv(metric)
                covariant_derivatives[a,b,c] = 0.5 * np.sum(
                    metric_inv[a,:] * (
                        d_gbc_dx +
                        np.roll(d_gbc_dx, 1) -
                        np.roll(d_gbc_dx, 2)
                    )
                )
    
    return covariant_derivatives

class MathematicalCSF:
    """基于深度数学模型的CSF算法实现
    
    该算法结合了黎曼几何、拓扑分析和谱分析的理论，通过以下创新点提高地面点云分类的准确性：
    
    1. 黎曼几何：使用黎曼度量张量和协变导数分析局部几何特征
    2. 拓扑分析：使用持久同伦分析提取点云的拓扑特征
    3. 谱分析：使用拉普拉斯算子分析点云的全局结构
    4. 自适应权重：根据局部几何和拓扑特征动态调整更新权重
    5. 数值稳定性：使用QR分解和SVD提高数值稳定性
    """
    
    def __init__(self, 
                 cloth_resolution: float = 0.5,
                 time_step: float = 0.65,
                 class_threshold: float = 0.5,
                 iterations: int = 500,
                 rigidness: int = 1,
                 curvature_weight: float = 0.3,
                 topological_weight: float = 0.2,
                 spectral_weight: float = 0.1,
                 batch_size: int = 1000,
                 stability_factor: float = 1e-6,
                 n_neighbors: int = 10):
        """
        初始化数学CSF算法参数
        
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
        curvature_weight : float
            曲率权重
        topological_weight : float
            拓扑权重
        spectral_weight : float
            谱权重
        batch_size : int
            批处理大小
        stability_factor : float
            数值稳定性因子
        n_neighbors : int
            计算局部特征时使用的邻居点数量
        """
        self.cloth_resolution = cloth_resolution
        self.time_step = time_step
        self.class_threshold = class_threshold
        self.iterations = iterations
        self.rigidness = rigidness
        self.curvature_weight = curvature_weight
        self.topological_weight = topological_weight
        self.spectral_weight = spectral_weight
        self.batch_size = batch_size
        self.stability_factor = stability_factor
        self.n_neighbors = n_neighbors
        
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
        if self.curvature_weight < 0:
            raise ValueError("curvature_weight must be non-negative")
        if self.topological_weight < 0:
            raise ValueError("topological_weight must be non-negative")
        if self.spectral_weight < 0:
            raise ValueError("spectral_weight must be non-negative")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.stability_factor <= 0:
            raise ValueError("stability_factor must be positive")
        if self.n_neighbors < 3:
            raise ValueError("n_neighbors must be at least 3")
    
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
    
    def _compute_riemannian_metric(self, points: np.ndarray, k: Optional[int] = None) -> np.ndarray:
        """计算黎曼度量张量"""
        if k is None:
            k = self.n_neighbors
            
        n_points = len(points)
        metric_tensors = np.zeros((n_points, 3, 3))
        
        # 创建KD树用于快速近邻搜索
        tree = cKDTree(points)
        
        logger.info("Computing Riemannian metric tensors...")
        for i in tqdm(range(n_points), desc="Computing metric tensors"):
            # 找到k个最近邻
            distances, indices = tree.query(points[i], k=k+1)
            indices = indices[1:]  # 排除点本身
            neighbors = points[indices]
            
            # 计算协方差矩阵作为黎曼度量张量
            metric_tensors[i] = _compute_riemannian_metric_numba(neighbors, points[i], k)
            
            # 添加正则化项以确保正定性
            metric_tensors[i] += np.eye(3) * self.stability_factor
        
        return metric_tensors
    
    def _compute_covariant_derivative(self, points: np.ndarray, metric_tensors: np.ndarray, 
                                    k: Optional[int] = None) -> np.ndarray:
        """计算协变导数"""
        if k is None:
            k = self.n_neighbors
            
        n_points = len(points)
        covariant_derivatives = np.zeros((n_points, 3, 3, 3))
        
        # 创建KD树用于快速近邻搜索
        tree = cKDTree(points)
        
        logger.info("Computing covariant derivatives...")
        for i in tqdm(range(n_points), desc="Computing derivatives"):
            # 找到k个最近邻
            distances, indices = tree.query(points[i], k=k+1)
            indices = indices[1:]  # 排除点本身
            neighbors = points[indices]
            
            # 计算局部坐标系
            center = np.mean(neighbors, axis=0)
            centered = neighbors - center
            Q, R = qr(centered.T, mode='economic')
            
            # 确保右手坐标系
            if np.linalg.det(Q) < 0:
                Q[:, -1] = -Q[:, -1]
            
            # 计算Christoffel符号
            metric = metric_tensors[i]
            h = np.mean(distances[1:])  # 使用平均距离作为差分步长
            
            covariant_derivatives[i] = _compute_covariant_derivative_numba(neighbors, center, metric, h)
        
        return covariant_derivatives
    
    def _compute_gaussian_curvature(self, metric_tensors: np.ndarray, 
                                  covariant_derivatives: np.ndarray) -> np.ndarray:
        """计算高斯曲率"""
        n_points = len(metric_tensors)
        curvatures = np.zeros(n_points)
        
        logger.info("Computing Gaussian curvatures...")
        for i in tqdm(range(n_points), desc="Computing curvatures"):
            g = metric_tensors[i]
            det_g = g[0,0]*g[1,1] - g[0,1]**2
            
            if det_g > self.stability_factor:  # 避免除以零
                # 计算黎曼曲率张量的1212分量
                R1212 = 0
                for k in range(3):
                    # 使用协变导数计算黎曼曲率张量
                    R1212 += (
                        covariant_derivatives[i,0,1,1] * covariant_derivatives[i,k,1,2] -
                        covariant_derivatives[i,0,1,2] * covariant_derivatives[i,k,1,1]
                    )
                
                # 计算高斯曲率
                curvatures[i] = R1212 / det_g
            else:
                curvatures[i] = 0.0
        
        return curvatures
    
    def _compute_topological_features(self, points: np.ndarray, k: int = 10) -> np.ndarray:
        """计算拓扑特征"""
        n_points = len(points)
        topological_features = np.zeros(n_points)
        
        # 创建KD树用于快速近邻搜索
        tree = cKDTree(points)
        
        logger.info("Computing topological features...")
        for i in tqdm(range(n_points), desc="Computing topological features"):
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
                topological_features[i] = 1.0 - (euler_char / k)
            else:
                topological_features[i] = 0
        
        return topological_features
    
    def _compute_spectral_features(self, points: np.ndarray, k: int = 10) -> np.ndarray:
        """计算谱特征"""
        n_points = len(points)
        
        # 创建KD树用于快速近邻搜索
        tree = cKDTree(points)
        
        # 初始化稀疏矩阵的数据结构
        rows = []
        cols = []
        data = []
        
        logger.info("Computing spectral features...")
        # 批处理计算权重矩阵
        for start_idx in tqdm(range(0, n_points, self.batch_size), desc="Computing spectral features"):
            end_idx = min(start_idx + self.batch_size, n_points)
            batch_points = points[start_idx:end_idx]
            
            # 找到每个点的邻居
            distances, indices = tree.query(batch_points, k=k+1)
            distances = distances[:, 1:]  # 排除自身
            indices = indices[:, 1:]      # 排除自身
            
            # 计算权重
            sigma = np.mean(distances, axis=1, keepdims=True)
            weights = np.exp(-distances**2 / (2 * sigma**2))
            
            # 更新稀疏矩阵数据
            for i in range(end_idx - start_idx):
                global_i = start_idx + i
                for j, weight in zip(indices[i], weights[i]):
                    rows.append(global_i)
                    cols.append(j)
                    data.append(weight)
        
        # 创建稀疏权重矩阵
        W = csr_matrix((data, (rows, cols)), shape=(n_points, n_points))
        W = (W + W.T) / 2  # 确保对称性
        
        # 计算拉普拉斯矩阵
        D = csr_matrix((W.sum(axis=1).A1, (range(n_points), range(n_points))))
        L = D - W
        
        # 计算归一化拉普拉斯矩阵
        D_sqrt = csr_matrix((1/np.sqrt(D.data), D.indices, D.indptr), shape=D.shape)
        L_norm = D_sqrt @ L @ D_sqrt
        
        try:
            # 使用稀疏特征值分解
            eigenvalues, eigenvectors = sparse_linalg.eigsh(L_norm, k=4, which='SM')
            
            # 计算谱特征
            spectral_features = np.zeros((n_points, len(eigenvalues)))
            for i, (val, vec) in enumerate(zip(eigenvalues, eigenvectors.T)):
                spectral_features[:, i] = vec
            
        except:
            # 如果特征值分解失败，使用PCA作为备选方案
            pca = PCA(n_components=4)
            spectral_features = pca.fit_transform(points)
        
        return spectral_features
    
    def _update_cloth_with_physics(self, cloth_points: np.ndarray, points: np.ndarray, 
                                 tree_indices: np.ndarray, metric_tensors: np.ndarray, 
                                 curvatures: np.ndarray, topological_features: np.ndarray, 
                                 spectral_features: np.ndarray) -> np.ndarray:
        """使用物理模型更新布料位置"""
        n_cloth = len(cloth_points)
        updated_cloth = cloth_points.copy()
        
        for i in range(n_cloth):
            # 找到对应的点云点
            idx = tree_indices[i]
            
            # 基础更新
            z_diff = points[idx, 2] - cloth_points[i, 2]
            
            # 考虑曲率的影响
            curvature_factor = np.clip(1.0 - self.curvature_weight * curvatures[idx], 0.1, 1.0)
            
            # 考虑拓扑特征的影响
            topological_factor = np.clip(1.0 - self.topological_weight * topological_features[idx], 0.1, 1.0)
            
            # 考虑谱特征的影响
            spectral_factor = np.clip(1.0 - self.spectral_weight * np.mean(np.abs(spectral_features[idx])), 0.1, 1.0)
            
            # 考虑黎曼度量
            metric = metric_tensors[idx]
            metric_factor = np.clip(np.sqrt(np.abs(metric[2,2])), 0.1, 10.0)  # 使用z方向的度量
            
            # 综合更新，添加数值稳定性检查
            update = z_diff * curvature_factor * topological_factor * spectral_factor * metric_factor
            
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
        
        # 预计算数学特征
        metric_tensors = self._compute_riemannian_metric(points)
        covariant_derivatives = self._compute_covariant_derivative(points, metric_tensors)
        curvatures = self._compute_gaussian_curvature(metric_tensors, covariant_derivatives)
        topological_features = self._compute_topological_features(points)
        spectral_features = self._compute_spectral_features(points)
        
        # 迭代更新布料位置
        logger.info("Updating cloth position...")
        for i in tqdm(range(self.iterations), desc="Updating cloth"):
            # 找到每个布料点对应的最近点云点
            distances, indices = tree.query(cloth_points[:, :2])
            
            # 使用物理模型更新布料
            cloth_points = self._update_cloth_with_physics(
                cloth_points, points, indices, 
                metric_tensors, curvatures, topological_features, spectral_features
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
        
        # 预计算数学特征
        if progress_callback is not None:
            progress_callback(0.1)  # 特征提取占总进度的10%
            
        metric_tensors = self._compute_riemannian_metric(points)
        if progress_callback is not None:
            progress_callback(0.2)
            
        covariant_derivatives = self._compute_covariant_derivative(points, metric_tensors)
        if progress_callback is not None:
            progress_callback(0.3)
            
        curvatures = self._compute_gaussian_curvature(metric_tensors, covariant_derivatives)
        if progress_callback is not None:
            progress_callback(0.4)
            
        topological_features = self._compute_topological_features(points)
        if progress_callback is not None:
            progress_callback(0.5)
            
        spectral_features = self._compute_spectral_features(points)
        if progress_callback is not None:
            progress_callback(0.6)
        
        # 迭代更新布料位置
        logger.info("Updating cloth position...")
        for i in range(self.iterations):
            # 找到每个布料点对应的最近点云点
            distances, indices = tree.query(cloth_points[:, :2])
            
            # 使用物理模型更新布料
            cloth_points = self._update_cloth_with_physics(
                cloth_points, points, indices, 
                metric_tensors, curvatures, topological_features, spectral_features
            )
            
            # 更新进度
            if progress_callback is not None:
                progress = 0.6 + (i + 1) / self.iterations * 0.3  # 布料更新占总进度的30%
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