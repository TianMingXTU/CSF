import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Tuple, Optional, Union, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import logging
from tqdm import tqdm
import os
from pathlib import Path
import open3d as o3d

class PointCloudVisualizer:
    """
    点云可视化类，提供多种可视化方法和交互功能
    
    支持的可视化方式：
    - 2D/3D散点图
    - 密度图
    - 等高线图
    - 特征分布图
    - 交互式3D可视化
    - 动画效果
    """
    
    def __init__(self, 
                 backend: str = 'matplotlib',
                 interactive: bool = True,
                 save_dir: Optional[str] = None):
        """
        初始化可视化器
        
        Parameters
        ----------
        backend : str, optional
            可视化后端，可选 'matplotlib', 'plotly', 'open3d'
        interactive : bool, optional
            是否使用交互式可视化
        save_dir : Optional[str], optional
            图像保存目录
        """
        self.backend = backend
        self.interactive = interactive
        self.save_dir = save_dir
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 创建保存目录
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # 设置样式
        if backend == 'matplotlib':
            try:
                plt.style.use('seaborn')
            except:
                plt.style.use('default')
                self.logger.warning("无法使用seaborn样式，使用默认样式")
        elif backend == 'plotly':
            # 设置Plotly主题
            import plotly.io as pio
            pio.templates.default = "plotly_white"
    
    def plot_points(self, 
                   points: np.ndarray, 
                   labels: Optional[np.ndarray] = None,
                   colors: Optional[np.ndarray] = None,
                   title: str = "Point Cloud Visualization",
                   save_path: Optional[str] = None) -> None:
        """
        绘制点云数据，支持多种可视化方式
        
        Parameters
        ----------
        points : np.ndarray
            点云数据，shape为(n_points, 3)
        labels : Optional[np.ndarray], optional
            点云标签，用于着色
        colors : Optional[np.ndarray], optional
            自定义颜色
        title : str, optional
            图表标题
        save_path : Optional[str], optional
            保存路径
        """
        self.logger.info(f"绘制点云，共{len(points)}个点")
        
        if self.backend == 'matplotlib':
            self._plot_points_matplotlib(points, labels, colors, title, save_path)
        elif self.backend == 'plotly':
            self._plot_points_plotly(points, labels, colors, title, save_path)
        elif self.backend == 'open3d':
            self._plot_points_open3d(points, labels, colors, title, save_path)
        else:
            raise ValueError(f"不支持的可视化后端: {self.backend}")
    
    def _plot_points_matplotlib(self, 
                              points: np.ndarray, 
                              labels: Optional[np.ndarray],
                              colors: Optional[np.ndarray],
                              title: str,
                              save_path: Optional[str]) -> None:
        """使用Matplotlib绘制点云"""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        if colors is not None:
            scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                               c=colors, cmap='viridis')
            plt.colorbar(scatter)
        elif labels is not None:
            scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                               c=labels, cmap='viridis')
            plt.colorbar(scatter)
        else:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2])
            
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        if self.interactive:
            plt.show()
        else:
            plt.close()
    
    def _plot_points_plotly(self, 
                          points: np.ndarray, 
                          labels: Optional[np.ndarray],
                          colors: Optional[np.ndarray],
                          title: str,
                          save_path: Optional[str]) -> None:
        """使用Plotly绘制点云"""
        if colors is not None:
            color_data = colors
        elif labels is not None:
            color_data = labels
        else:
            color_data = None
            
        fig = go.Figure(data=[go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=color_data,
                colorscale='Viridis',
                opacity=0.8
            )
        )])
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path)
            
        if self.interactive:
            fig.show()
    
    def _plot_points_open3d(self, 
                          points: np.ndarray, 
                          labels: Optional[np.ndarray],
                          colors: Optional[np.ndarray],
                          title: str,
                          save_path: Optional[str]) -> None:
        """使用Open3D绘制点云"""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        elif labels is not None:
            # 将标签转换为颜色
            colors = plt.cm.viridis(labels / np.max(labels))
            pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
            
        if save_path:
            o3d.io.write_point_cloud(save_path, pcd)
            
        if self.interactive:
            o3d.visualization.draw_geometries([pcd], window_name=title)
    
    def _plot_points_pptk(self, 
                        points: np.ndarray, 
                        labels: Optional[np.ndarray],
                        colors: Optional[np.ndarray],
                        title: str,
                        save_path: Optional[str]) -> None:
        """使用Open3D替代PPTK绘制点云"""
        self.logger.warning("PPTK后端不可用，使用Open3D替代")
        self._plot_points_open3d(points, labels, colors, title, save_path)
    
    def plot_metrics(self, 
                    metrics_dict: Dict[str, float], 
                    title: str = "Classification Metrics",
                    save_path: Optional[str] = None) -> None:
        """
        绘制评估指标
        
        Parameters
        ----------
        metrics_dict : Dict[str, float]
            评估指标字典
        title : str, optional
            图表标题
        save_path : Optional[str], optional
            保存路径
        """
        self.logger.info("绘制评估指标")
        
        if self.backend == 'matplotlib':
            self._plot_metrics_matplotlib(metrics_dict, title, save_path)
        elif self.backend == 'plotly':
            self._plot_metrics_plotly(metrics_dict, title, save_path)
        else:
            raise ValueError(f"不支持的可视化后端: {self.backend}")
    
    def _plot_metrics_matplotlib(self, 
                               metrics_dict: Dict[str, float],
                               title: str,
                               save_path: Optional[str]) -> None:
        """使用Matplotlib绘制评估指标"""
        metrics = list(metrics_dict.keys())
        values = list(metrics_dict.values())
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics, values)
        plt.ylim(0, 1)
        plt.title(title)
        plt.ylabel('Score')
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom')
            
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        if self.interactive:
            plt.show()
        else:
            plt.close()
    
    def _plot_metrics_plotly(self, 
                           metrics_dict: Dict[str, float],
                           title: str,
                           save_path: Optional[str]) -> None:
        """使用Plotly绘制评估指标"""
        fig = go.Figure(data=[
            go.Bar(
                x=list(metrics_dict.keys()),
                y=list(metrics_dict.values()),
                text=[f'{v:.3f}' for v in metrics_dict.values()],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title=title,
            yaxis=dict(range=[0, 1]),
            xaxis_tickangle=-45
        )
        
        if save_path:
            fig.write_html(save_path)
            
        if self.interactive:
            fig.show()
    
    def plot_feature_importance(self, 
                              importance_dict: Dict[str, float], 
                              title: str = "Feature Importance",
                              save_path: Optional[str] = None) -> None:
        """
        绘制特征重要性
        
        Parameters
        ----------
        importance_dict : Dict[str, float]
            特征重要性字典
        title : str, optional
            图表标题
        save_path : Optional[str], optional
            保存路径
        """
        self.logger.info("绘制特征重要性")
        
        if self.backend == 'matplotlib':
            self._plot_feature_importance_matplotlib(importance_dict, title, save_path)
        elif self.backend == 'plotly':
            self._plot_feature_importance_plotly(importance_dict, title, save_path)
        else:
            raise ValueError(f"不支持的可视化后端: {self.backend}")
    
    def _plot_feature_importance_matplotlib(self, 
                                         importance_dict: Dict[str, float],
                                         title: str,
                                         save_path: Optional[str]) -> None:
        """使用Matplotlib绘制特征重要性"""
        features = list(importance_dict.keys())
        importance = list(importance_dict.values())
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(features, importance)
        plt.title(title)
        plt.ylabel('Importance')
        plt.xticks(rotation=45)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom')
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        if self.interactive:
            plt.show()
        else:
            plt.close()
    
    def _plot_feature_importance_plotly(self, 
                                     importance_dict: Dict[str, float],
                                     title: str,
                                     save_path: Optional[str]) -> None:
        """使用Plotly绘制特征重要性"""
        fig = go.Figure(data=[
            go.Bar(
                x=list(importance_dict.keys()),
                y=list(importance_dict.values()),
                text=[f'{v:.3f}' for v in importance_dict.values()],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title=title,
            xaxis_tickangle=-45
        )
        
        if save_path:
            fig.write_html(save_path)
            
        if self.interactive:
            fig.show()
    
    def plot_confusion_matrix(self, 
                            confusion_mat: np.ndarray, 
                            title: str = "Confusion Matrix",
                            save_path: Optional[str] = None) -> None:
        """
        绘制混淆矩阵
        
        Parameters
        ----------
        confusion_mat : np.ndarray
            混淆矩阵
        title : str, optional
            图表标题
        save_path : Optional[str], optional
            保存路径
        """
        self.logger.info("绘制混淆矩阵")
        
        if self.backend == 'matplotlib':
            self._plot_confusion_matrix_matplotlib(confusion_mat, title, save_path)
        elif self.backend == 'plotly':
            self._plot_confusion_matrix_plotly(confusion_mat, title, save_path)
        else:
            raise ValueError(f"不支持的可视化后端: {self.backend}")
    
    def _plot_confusion_matrix_matplotlib(self, 
                                       confusion_mat: np.ndarray,
                                       title: str,
                                       save_path: Optional[str]) -> None:
        """使用Matplotlib绘制混淆矩阵"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues')
        plt.title(title)
        
        # 添加标签
        classes = ['Non-ground', 'Ground']
        plt.xticks([0.5, 1.5], classes)
        plt.yticks([0.5, 1.5], classes)
        
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        if self.interactive:
            plt.show()
        else:
            plt.close()
    
    def _plot_confusion_matrix_plotly(self, 
                                    confusion_mat: np.ndarray,
                                    title: str,
                                    save_path: Optional[str]) -> None:
        """使用Plotly绘制混淆矩阵"""
        fig = go.Figure(data=go.Heatmap(
            z=confusion_mat,
            x=['Non-ground', 'Ground'],
            y=['Non-ground', 'Ground'],
            text=confusion_mat,
            texttemplate='%{text}',
            textfont={"size": 16},
            colorscale='Blues'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Predicted label',
            yaxis_title='True label'
        )
        
        if save_path:
            fig.write_html(save_path)
            
        if self.interactive:
            fig.show()
    
    def create_animation(self, 
                        points_list: List[np.ndarray], 
                        labels_list: Optional[List[np.ndarray]] = None,
                        title: str = "Point Cloud Animation",
                        save_path: Optional[str] = None) -> None:
        """
        创建点云动画
        
        Parameters
        ----------
        points_list : List[np.ndarray]
            点云数据列表
        labels_list : Optional[List[np.ndarray]], optional
            标签数据列表
        title : str, optional
            动画标题
        save_path : Optional[str], optional
            保存路径
        """
        self.logger.info("创建点云动画")
        
        if self.backend == 'plotly':
            self._create_animation_plotly(points_list, labels_list, title, save_path)
        else:
            raise ValueError("目前只支持使用Plotly创建动画")
    
    def _create_animation_plotly(self, 
                               points_list: List[np.ndarray],
                               labels_list: Optional[List[np.ndarray]],
                               title: str,
                               save_path: Optional[str]) -> None:
        """使用Plotly创建点云动画"""
        frames = []
        
        for i, points in enumerate(points_list):
            if labels_list is not None:
                color_data = labels_list[i]
            else:
                color_data = None
                
            frame = go.Frame(
                data=[go.Scatter3d(
                    x=points[:, 0],
                    y=points[:, 1],
                    z=points[:, 2],
                    mode='markers',
                    marker=dict(
                        size=2,
                        color=color_data,
                        colorscale='Viridis',
                        opacity=0.8
                    )
                )],
                name=f'frame{i}'
            )
            frames.append(frame)
            
        fig = go.Figure(
            data=[go.Scatter3d(
                x=points_list[0][:, 0],
                y=points_list[0][:, 1],
                z=points_list[0][:, 2],
                mode='markers',
                marker=dict(
                    size=2,
                    color=labels_list[0] if labels_list is not None else None,
                    colorscale='Viridis',
                    opacity=0.8
                )
            )],
            frames=frames
        )
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {'frame': {'duration': 500, 'redraw': True}, 'fromcurrent': True}]
                    },
                    {
                        'label': 'Pause',
                        'method': 'animate',
                        'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate', 'transition': {'duration': 0}}]
                    }
                ]
            }]
        )
        
        if save_path:
            fig.write_html(save_path)
            
        if self.interactive:
            fig.show()
    
    def plot_feature_distribution(self, 
                                features: Dict[str, np.ndarray],
                                title: str = "Feature Distribution",
                                save_path: Optional[str] = None) -> None:
        """
        绘制特征分布
        
        Parameters
        ----------
        features : Dict[str, np.ndarray]
            特征字典
        title : str, optional
            图表标题
        save_path : Optional[str], optional
            保存路径
        """
        self.logger.info("绘制特征分布")
        
        if self.backend == 'matplotlib':
            self._plot_feature_distribution_matplotlib(features, title, save_path)
        elif self.backend == 'plotly':
            self._plot_feature_distribution_plotly(features, title, save_path)
        else:
            raise ValueError(f"不支持的可视化后端: {self.backend}")
    
    def _plot_feature_distribution_matplotlib(self, 
                                           features: Dict[str, np.ndarray],
                                           title: str,
                                           save_path: Optional[str]) -> None:
        """使用Matplotlib绘制特征分布"""
        n_features = len(features)
        fig, axes = plt.subplots(n_features, 1, figsize=(10, 3*n_features))
        
        if n_features == 1:
            axes = [axes]
            
        for ax, (name, values) in zip(axes, features.items()):
            sns.histplot(values, kde=True, ax=ax)
            ax.set_title(name)
            
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        if self.interactive:
            plt.show()
        else:
            plt.close()
    
    def _plot_feature_distribution_plotly(self, 
                                       features: Dict[str, np.ndarray],
                                       title: str,
                                       save_path: Optional[str]) -> None:
        """使用Plotly绘制特征分布"""
        fig = make_subplots(rows=len(features), cols=1, subplot_titles=list(features.keys()))
        
        for i, (name, values) in enumerate(features.items(), 1):
            fig.add_trace(
                go.Histogram(x=values, name=name, nbinsx=30),
                row=i, col=1
            )
            
        fig.update_layout(
            title_text=title,
            height=300*len(features)
        )
        
        if save_path:
            fig.write_html(save_path)
            
        if self.interactive:
            fig.show() 