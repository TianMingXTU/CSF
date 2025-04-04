import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class ClassificationMetrics:
    """分类评估指标计算类"""
    
    @staticmethod
    def compute_metrics(true_labels, pred_labels):
        """
        计算分类评估指标
        
        Parameters
        ----------
        true_labels : np.ndarray
            真实标签
        pred_labels : np.ndarray
            预测标签
            
        Returns
        -------
        metrics : dict
            包含各种评估指标的字典
        """
        metrics = {
            'accuracy': accuracy_score(true_labels, pred_labels),
            'precision': precision_score(true_labels, pred_labels),
            'recall': recall_score(true_labels, pred_labels),
            'f1': f1_score(true_labels, pred_labels),
            'confusion_matrix': confusion_matrix(true_labels, pred_labels)
        }
        
        return metrics
    
    @staticmethod
    def compute_efficiency_metrics(execution_time, memory_usage):
        """
        计算效率评估指标
        
        Parameters
        ----------
        execution_time : float
            执行时间（秒）
        memory_usage : float
            内存使用量（MB）
            
        Returns
        -------
        metrics : dict
            包含效率评估指标的字典
        """
        metrics = {
            'execution_time': execution_time,
            'memory_usage': memory_usage,
            'points_per_second': 0  # 将在计算时更新
        }
        
        return metrics
    
    @staticmethod
    def compute_robustness_metrics(results_dict):
        """
        计算鲁棒性评估指标
        
        Parameters
        ----------
        results_dict : dict
            包含不同场景下分类结果的字典
            
        Returns
        -------
        metrics : dict
            包含鲁棒性评估指标的字典
        """
        metrics = {
            'std_accuracy': np.std([r['accuracy'] for r in results_dict.values()]),
            'min_accuracy': min(r['accuracy'] for r in results_dict.values()),
            'max_accuracy': max(r['accuracy'] for r in results_dict.values())
        }
        
        return metrics 