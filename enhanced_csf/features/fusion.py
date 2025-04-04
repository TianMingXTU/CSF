import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import logging
from tqdm import tqdm
import joblib
from functools import lru_cache

class FeatureFusion:
    """
    特征融合类，提供多种特征融合方法和模型选择功能
    
    支持多种分类器和特征融合策略，包括：
    - 随机森林
    - 梯度提升
    - 支持向量机
    - 特征重要性分析
    - 交叉验证
    """
    
    def __init__(self, 
                 method: str = 'random_forest',
                 n_estimators: int = 100,
                 cross_validate: bool = False,
                 n_folds: int = 5,
                 cache_size: int = 128):
        """
        初始化特征融合器
        
        Parameters
        ----------
        method : str
            融合方法，可选 'random_forest', 'svm', 'gradient_boosting'
        n_estimators : int
            随机森林的树数量
        cross_validate : bool
            是否进行交叉验证
        n_folds : int
            交叉验证的折数
        cache_size : int
            缓存大小
        """
        self.method = method
        self.n_estimators = n_estimators
        self.cross_validate = cross_validate
        self.n_folds = n_folds
        self.cache_size = cache_size
        
        # 初始化模型
        if method == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        elif method == 'svm':
            self.model = SVC(probability=True, random_state=42)
        elif method == 'gradient_boosting':
            self.model = GradientBoostingClassifier(n_estimators=n_estimators, random_state=42)
        else:
            raise ValueError(f"不支持的方法: {method}")
        
        # 初始化特征缩放器
        self.scaler = StandardScaler()
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
    def _prepare_feature_matrix(self, csf_labels: Tuple[int, ...], 
                              local_features: Dict[str, Tuple[float, ...]]) -> np.ndarray:
        """
        准备特征矩阵
        
        Parameters
        ----------
        csf_labels : Tuple[int, ...]
            CSF算法分类标签
        local_features : Dict[str, Tuple[float, ...]]
            局部特征字典
            
        Returns
        -------
        feature_matrix : np.ndarray
            特征矩阵
        """
        # 将标签和特征转换为特征矩阵
        n_samples = len(csf_labels)
        n_features = 1 + len(local_features)  # CSF标签 + 局部特征数量
        
        feature_matrix = np.zeros((n_samples, n_features))
        
        # 添加CSF标签
        feature_matrix[:, 0] = csf_labels
        
        # 添加局部特征
        for i, (feature_name, feature_values) in enumerate(local_features.items(), 1):
            feature_matrix[:, i] = feature_values
        
        return feature_matrix
        
    def fuse_features(self, 
                     csf_labels: np.ndarray, 
                     local_features: Dict[str, np.ndarray],
                     true_labels: Optional[np.ndarray] = None) -> np.ndarray:
        """
        融合CSF分类结果和局部特征
        
        Parameters
        ----------
        csf_labels : np.ndarray
            CSF算法的分类结果
        local_features : Dict[str, np.ndarray]
            局部特征字典
        true_labels : Optional[np.ndarray]
            真实标签（用于交叉验证）
            
        Returns
        -------
        final_labels : np.ndarray
            融合后的分类结果
        """
        self.logger.info(f"开始特征融合，使用分类器: {self.method}")
        
        # 准备特征矩阵
        feature_matrix = self._prepare_feature_matrix(
            tuple(csf_labels), 
            {k: tuple(v) for k, v in local_features.items()}
        )
        
        # 使用CSF结果作为初始标签
        initial_labels = csf_labels
        
        # 交叉验证
        if self.cross_validate and true_labels is not None:
            self.logger.info(f"执行{self.n_folds}折交叉验证")
            scores = cross_val_score(
                self.model, 
                feature_matrix, 
                true_labels, 
                cv=self.n_folds,
                n_jobs=-1
            )
            self.logger.info(f"交叉验证得分: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
        
        # 训练分类器
        self.logger.info("训练分类器")
        self.model.fit(feature_matrix, initial_labels)
        
        # 预测最终标签
        self.logger.info("预测最终标签")
        final_labels = self.model.predict(feature_matrix)
        
        return final_labels
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        获取特征重要性
        
        Returns
        -------
        importance : Dict[str, float]
            特征重要性字典
        """
        if not hasattr(self.model, 'feature_importances_'):
            self.logger.warning("当前分类器不支持特征重要性分析")
            return {}
            
        feature_names = ['csf_label', 'planarity', 'curvature', 'normal_consistency']
        importance = dict(zip(feature_names, self.model.feature_importances_))
        
        # 按重要性排序
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        
        return importance
    
    def save_model(self, filepath: str) -> None:
        """
        保存模型到文件
        
        Parameters
        ----------
        filepath : str
            模型保存路径
        """
        self.logger.info(f"保存模型到: {filepath}")
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'method': self.method
        }, filepath)
    
    def load_model(self, filepath: str) -> None:
        """
        从文件加载模型
        
        Parameters
        ----------
        filepath : str
            模型文件路径
        """
        self.logger.info(f"从{filepath}加载模型")
        model_dict = joblib.load(filepath)
        self.model = model_dict['model']
        self.scaler = model_dict['scaler']
        self.method = model_dict['method'] 