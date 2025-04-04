#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced CSF算法测试与对比

本脚本用于同时运行所有CSF算法实现，并进行全面的对比分析。
包括性能对比、分类效果对比、参数敏感性分析和结果可视化。
所有结果将保存到output目录中，包括详细的评估报告、可视化图表和原始数据。
"""

import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
from scipy.io import loadmat
from scipy.spatial import cKDTree
import laspy
import json
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import logging
import open3d as o3d
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from numba import jit, prange, njit, vectorize, float64, int64, boolean

# 导入所有CSF算法实现
from enhanced_csf.core import CSF, MathematicalCSF, DifferentialCSF, ManifoldCSF

# 设置随机种子，确保结果可复现
np.random.seed(42)

# 创建输出目录
output_dir = Path("examples/output")
output_dir.mkdir(parents=True, exist_ok=True)

# 设置全局日志记录器
logger = logging.getLogger("CSF_Test")
logger.setLevel(logging.INFO)

# 添加文件处理器
log_file = output_dir / f"csf_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# 添加控制台处理器
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

class NumpyEncoder(json.JSONEncoder):
    """处理NumPy类型的JSON编码器"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# Numba加速函数 - 生成建筑物点
@jit(nopython=True, parallel=True)
def _generate_building_points_numba(bx, by, bw, bl, bh, base_height, n_points):
    """使用Numba加速生成建筑物点"""
    points = np.zeros((n_points, 3))
    for i in prange(n_points):
        px = bx + np.random.uniform(-bw/2, bw/2)
        py = by + np.random.uniform(-bl/2, bl/2)
        pz = base_height + np.random.uniform(0, bh)
        points[i, 0] = px
        points[i, 1] = py
        points[i, 2] = pz
    return points

# Numba加速函数 - 生成树木点
@jit(nopython=True, parallel=True)
def _generate_tree_points_numba(tx, ty, tr, th, base_height, n_points):
    """使用Numba加速生成树木点"""
    points = np.zeros((n_points, 3))
    for i in prange(n_points):
        r = tr * np.sqrt(np.random.uniform(0, 1))
        theta = np.random.uniform(0, 2*np.pi)
        px = tx + r * np.cos(theta)
        py = ty + r * np.sin(theta)
        pz = base_height + np.random.uniform(0, th * (1 - r/tr))
        points[i, 0] = px
        points[i, 1] = py
        points[i, 2] = pz
    return points

# Numba加速函数 - 计算点云密度
@jit(nopython=True, parallel=True)
def _compute_point_density_numba(points_2d, k=2):
    """使用Numba加速计算点云密度"""
    n_points = len(points_2d)
    distances = np.zeros((n_points, k))
    
    for i in prange(n_points):
        for j in range(n_points):
            if i != j:
                dist = np.sqrt((points_2d[i, 0] - points_2d[j, 0])**2 + 
                              (points_2d[i, 1] - points_2d[j, 1])**2)
                
                # 更新k个最近邻
                for k_idx in range(k):
                    if k_idx < len(distances[i]) and (distances[i, k_idx] == 0 or dist < distances[i, k_idx]):
                        # 移动其他距离
                        for m in range(k-1, k_idx, -1):
                            distances[i, m] = distances[i, m-1]
                        distances[i, k_idx] = dist
                        break
    
    return distances

# Numba加速函数 - 计算分类错误的空间分布
@jit(nopython=True, parallel=True)
def _compute_error_distribution_numba(points, labels, ground_truth):
    """使用Numba加速计算分类错误的空间分布"""
    errors = np.abs(labels - ground_truth)
    error_indices = np.where(errors == 1)[0]
    
    if len(error_indices) == 0:
        return np.array([0.0, 0.0, 0.0])
    
    error_points = np.zeros((len(error_indices), 3))
    for i in prange(len(error_indices)):
        error_points[i] = points[error_indices[i]]
    
    mean_elevation = np.mean(error_points[:, 2])
    elevation_std = np.std(error_points[:, 2])
    error_spatial_density = len(error_indices) / len(points)
    
    return np.array([mean_elevation, elevation_std, error_spatial_density])

def load_sample_data(file_path=None):
    """
    加载示例点云数据
    
    Parameters
    ----------
    file_path : str, optional
        点云数据文件路径，支持.npy, .txt, .mat, .las格式
        
    Returns
    -------
    points : np.ndarray
        点云数据，shape为(n_points, 3)
    ground_truth : np.ndarray, optional
        地面真值标签，如果数据集中包含
    """
    if file_path is None:
        # 生成模拟点云数据
        logger.info("生成模拟点云数据...")
        
        # 生成地面点
        x = np.linspace(-10, 10, 100)
        y = np.linspace(-10, 10, 100)
        xx, yy = np.meshgrid(x, y)
        
        # 添加一些地形变化
        z_ground = 0.1 * np.sin(0.5 * xx) * np.cos(0.5 * yy) + 0.05 * np.sin(2 * xx) * np.cos(2 * yy)
        
        # 生成地面点
        ground_points = np.column_stack((xx.flatten(), yy.flatten(), z_ground.flatten()))
        
        # 生成非地面点（建筑物、树木等）
        n_buildings = 5
        n_trees = 20
        n_points_per_building = 1000
        n_points_per_tree = 200
        
        non_ground_points = []
        
        # 生成建筑物点
        for i in range(n_buildings):
            # 随机建筑物位置
            bx = np.random.uniform(-8, 8)
            by = np.random.uniform(-8, 8)
            bw = np.random.uniform(1, 3)
            bl = np.random.uniform(1, 3)
            bh = np.random.uniform(2, 5)
            
            # 找到建筑物底部的地面高度
            tree = cKDTree(ground_points[:, :2])
            _, idx = tree.query([bx, by])
            base_height = ground_points[idx, 2]
            
            # 使用Numba加速生成建筑物点
            building_points = _generate_building_points_numba(bx, by, bw, bl, bh, base_height, n_points_per_building)
            non_ground_points.append(building_points)
        
        # 生成树木点
        for i in range(n_trees):
            # 随机树木位置
            tx = np.random.uniform(-9, 9)
            ty = np.random.uniform(-9, 9)
            tr = np.random.uniform(0.3, 0.8)
            th = np.random.uniform(1, 3)
            
            # 找到树木底部的地面高度
            tree = cKDTree(ground_points[:, :2])
            _, idx = tree.query([tx, ty])
            base_height = ground_points[idx, 2]
            
            # 使用Numba加速生成树木点
            tree_points = _generate_tree_points_numba(tx, ty, tr, th, base_height, n_points_per_tree)
            non_ground_points.append(tree_points)
        
        # 合并所有点
        non_ground_points = np.vstack(non_ground_points)
        points = np.vstack((ground_points, non_ground_points))
        
        # 生成地面真值标签
        ground_truth = np.zeros(len(points))
        ground_truth[:len(ground_points)] = 1
        
        # 添加一些噪声
        noise = np.random.normal(0, 0.05, size=points.shape)
        points += noise
        
        return points, ground_truth
    
    else:
        # 根据文件扩展名加载数据
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.npy':
            data = np.load(file_path)
            if isinstance(data, np.ndarray) and data.shape[1] >= 3:
                points = data[:, :3]
                if data.shape[1] > 3:
                    ground_truth = data[:, 3]
                    return points, ground_truth
                return points, None
            else:
                raise ValueError("Invalid .npy file format")
                
        elif ext == '.txt':
            data = np.loadtxt(file_path)
            if data.shape[1] >= 3:
                points = data[:, :3]
                if data.shape[1] > 3:
                    ground_truth = data[:, 3]
                    return points, ground_truth
                return points, None
            else:
                raise ValueError("Invalid .txt file format")
                
        elif ext == '.mat':
            data = loadmat(file_path)
            # 尝试找到点云数据
            for key in data.keys():
                if not key.startswith('__') and isinstance(data[key], np.ndarray) and data[key].shape[1] >= 3:
                    points = data[key][:, :3]
                    if data[key].shape[1] > 3:
                        ground_truth = data[key][:, 3]
                        return points, ground_truth
                    return points, None
            raise ValueError("Could not find point cloud data in .mat file")
            
        elif ext == '.las':
            # 加载LAS文件
            las = laspy.read(file_path)
            
            # 提取点云坐标
            points = np.vstack((las.x, las.y, las.z)).transpose()
            
            # 提取分类标签（如果有）
            if hasattr(las, 'classification') and las.classification is not None:
                ground_truth = las.classification
                # 将分类标签转换为二值标签（1表示地面点，0表示非地面点）
                ground_truth = (ground_truth == 2).astype(int)  # LAS标准中，2通常表示地面点
                return points, ground_truth
            
            return points, None
            
        else:
            raise ValueError(f"Unsupported file format: {ext}")

def evaluate_classification(points, labels, ground_truth=None):
    """
    增强版分类评估函数
    
    Parameters
    ----------
    points : np.ndarray
        点云数据，shape为(n_points, 3)
    labels : np.ndarray
        分类标签，1表示地面点，0表示非地面点
    ground_truth : np.ndarray, optional
        地面真值标签，如果提供则计算分类指标
        
    Returns
    -------
    metrics : dict
        评估指标
    """
    metrics = {}
    
    # 基本统计信息
    metrics['total_points'] = len(points)
    metrics['ground_points'] = np.sum(labels == 1)
    metrics['non_ground_points'] = np.sum(labels == 0)
    metrics['ground_ratio'] = metrics['ground_points'] / metrics['total_points']
    
    # 计算点云密度和分布特征
    # 使用Numba加速计算点云密度
    distances = _compute_point_density_numba(points[:, :2], k=2)
    metrics['mean_point_density'] = 1.0 / np.mean(distances[:, 1])
    metrics['point_density_std'] = np.std(1.0 / distances[:, 1])
    
    # 计算高程统计特征
    metrics['mean_elevation'] = np.mean(points[:, 2])
    metrics['elevation_std'] = np.std(points[:, 2])
    metrics['elevation_range'] = np.ptp(points[:, 2])
    
    if ground_truth is not None:
        # 分类性能指标
        cm = confusion_matrix(ground_truth, labels)
        metrics['confusion_matrix'] = cm.tolist()
        
        report = classification_report(ground_truth, labels, output_dict=True)
        metrics['accuracy'] = report['accuracy']
        metrics['precision'] = report['1.0']['precision'] if '1.0' in report else report['1']['precision']
        metrics['recall'] = report['1.0']['recall'] if '1.0' in report else report['1']['recall']
        metrics['f1_score'] = report['1.0']['f1-score'] if '1.0' in report else report['1']['f1-score']
        
        # Kappa系数
        n = np.sum(cm)
        p0 = (cm[0, 0] + cm[1, 1]) / n
        pe = ((cm[0, 0] + cm[0, 1]) * (cm[0, 0] + cm[1, 0]) + 
              (cm[1, 0] + cm[1, 1]) * (cm[0, 1] + cm[1, 1])) / (n * n)
        kappa = (p0 - pe) / (1 - pe)
        metrics['kappa'] = kappa
        
        # ROC曲线和AUC
        fpr, tpr, _ = roc_curve(ground_truth, labels)
        metrics['auc'] = auc(fpr, tpr)
        metrics['roc_curve'] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}
        
        # 使用Numba加速计算分类错误的空间分布
        error_stats = _compute_error_distribution_numba(points, labels, ground_truth)
        metrics['error_mean_elevation'] = error_stats[0]
        metrics['error_elevation_std'] = error_stats[1]
        metrics['error_spatial_density'] = error_stats[2]
    
    return metrics

def save_results(results, output_dir):
    """
    保存测试结果
    
    Parameters
    ----------
    results : dict
        测试结果字典
    output_dir : str
        输出目录路径
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存评估指标
    metrics_file = output_dir / "evaluation_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)
    
    # 保存为CSV格式
    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv(output_dir / "evaluation_metrics.csv", index=False)
    
    # 生成HTML报告
    generate_html_report(results, output_dir)

def generate_html_report(results, output_dir):
    """
    生成HTML格式的测试报告
    
    Parameters
    ----------
    results : dict
        测试结果字典
    output_dir : Path
        输出目录路径
    """
    # 获取基本信息
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_points = next(iter(results.values())).get('total_points', 'N/A')
    
    # 创建HTML内容
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>CSF Algorithm Test Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .section {{ margin-bottom: 20px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .plot {{ margin: 20px 0; }}
        </style>
    </head>
    <body>
        <h1>CSF Algorithm Test Report</h1>
        <div class="section">
            <h2>Test Information</h2>
            <p>Test Date: {date_str}</p>
            <p>Total Points: {total_points}</p>
        </div>
        <div class="section">
            <h2>Evaluation Metrics</h2>
            <table>
                <tr>
                    <th>Algorithm</th>
                    <th>Accuracy</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1 Score</th>
                    <th>Kappa</th>
                </tr>
    """
    
    # 添加每个算法的评估指标
    for algo_name, metrics in results.items():
        html_content += f"""
                <tr>
                    <td>{algo_name}</td>
                    <td>{metrics.get('accuracy', 'N/A'):.4f}</td>
                    <td>{metrics.get('precision', 'N/A'):.4f}</td>
                    <td>{metrics.get('recall', 'N/A'):.4f}</td>
                    <td>{metrics.get('f1_score', 'N/A'):.4f}</td>
                    <td>{metrics.get('kappa', 'N/A'):.4f}</td>
                </tr>
        """
    
    # 添加表格结束标签和其他内容
    html_content += """
            </table>
        </div>
    </body>
    </html>
    """
    
    # 保存HTML报告
    with open(output_dir / "test_report.html", 'w') as f:
        f.write(html_content)

def visualize_results_3d(points, labels, ground_truth=None, title="Classification Results", save_path=None):
    """
    使用Plotly生成交互式3D可视化
    
    Parameters
    ----------
    points : np.ndarray
        点云数据
    labels : np.ndarray
        分类标签
    ground_truth : np.ndarray, optional
        地面真值标签
    title : str
        图表标题
    save_path : str, optional
        保存路径
    """
    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]])
    
    # 添加分类结果
    fig.add_trace(
        go.Scatter3d(
            x=points[labels == 1, 0],
            y=points[labels == 1, 1],
            z=points[labels == 1, 2],
            mode='markers',
            marker=dict(size=2, color='red', opacity=0.8),
            name='Ground'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter3d(
            x=points[labels == 0, 0],
            y=points[labels == 0, 1],
            z=points[labels == 0, 2],
            mode='markers',
            marker=dict(size=2, color='blue', opacity=0.8),
            name='Non-ground'
        ),
        row=1, col=1
    )
    
    if ground_truth is not None:
        # 添加地面真值
        fig.add_trace(
            go.Scatter3d(
                x=points[ground_truth == 1, 0],
                y=points[ground_truth == 1, 1],
                z=points[ground_truth == 1, 2],
                mode='markers',
                marker=dict(size=2, color='green', opacity=0.8),
                name='Ground Truth'
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter3d(
                x=points[ground_truth == 0, 0],
                y=points[ground_truth == 0, 1],
                z=points[ground_truth == 0, 2],
                mode='markers',
                marker=dict(size=2, color='yellow', opacity=0.8),
                name='Non-ground Truth'
            ),
            row=1, col=2
        )
    
    fig.update_layout(
        title_text=title,
        scene=dict(
            aspectmode='data',
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        )
    )
    
    if save_path:
        fig.write_html(save_path)
    else:
        fig.show()

def plot_performance_comparison(results, execution_times, output_dir):
    """
    生成算法性能对比图
    
    Parameters
    ----------
    results : dict
        评估结果
    execution_times : dict
        执行时间
    output_dir : str
        输出目录
    """
    # 创建性能指标DataFrame
    performance_data = []
    for algo_name in results.keys():
        metrics = results[algo_name]
        performance_data.append({
            'Algorithm': algo_name,
            'Accuracy': metrics.get('accuracy', 0),
            'F1 Score': metrics.get('f1_score', 0),
            'Kappa': metrics.get('kappa', 0),
            'Execution Time (s)': execution_times[algo_name]
        })
    
    df = pd.DataFrame(performance_data)
    
    # 创建性能对比图
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=('Accuracy', 'F1 Score', 'Kappa', 'Execution Time'))
    
    fig.add_trace(
        go.Bar(x=df['Algorithm'], y=df['Accuracy'], name='Accuracy'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=df['Algorithm'], y=df['F1 Score'], name='F1 Score'),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Bar(x=df['Algorithm'], y=df['Kappa'], name='Kappa'),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Bar(x=df['Algorithm'], y=df['Execution Time (s)'], name='Execution Time'),
        row=2, col=2
    )
    
    fig.update_layout(height=800, title_text="Algorithm Performance Comparison")
    
    # 保存图表
    fig.write_html(str(Path(output_dir) / "performance_comparison.html"))

# Numba加速函数 - 参数敏感性分析中的算法运行
@jit(nopython=False)
def _run_algorithm_with_param(points, param_name, param_value):
    """使用Numba加速运行算法并返回标签"""
    # 注意：这个函数不能完全用numba加速，因为CSF类不是numba兼容的
    # 但我们可以在这里准备数据，然后在外部运行算法
    return param_name, param_value

def compare_algorithms(points, ground_truth=None, algorithms_to_run=None, output_dir="examples/output"):
    """
    增强版算法对比函数
    
    Parameters
    ----------
    points : np.ndarray
        点云数据
    ground_truth : np.ndarray, optional
        地面真值标签
    algorithms_to_run : list, optional
        要运行的算法列表
    output_dir : str
        输出目录
    """
    if algorithms_to_run is None:
        algorithms_to_run = ['CSF', 'MathematicalCSF', 'DifferentialCSF', 'ManifoldCSF']
    
    results = {}
    execution_times = {}
    
    for algo_name in algorithms_to_run:
        logger.info(f"Running {algo_name}...")
        
        # 选择算法
        if algo_name == 'CSF':
            algo = CSF()
        elif algo_name == 'MathematicalCSF':
            algo = MathematicalCSF()
        elif algo_name == 'DifferentialCSF':
            algo = DifferentialCSF()
        elif algo_name == 'ManifoldCSF':
            algo = ManifoldCSF()
        else:
            logger.warning(f"Unknown algorithm: {algo_name}")
            continue
        
        # 运行算法并计时
        start_time = time.time()
        labels = algo.classify(points)
        execution_time = time.time() - start_time
        execution_times[algo_name] = execution_time
        
        # 评估结果
        metrics = evaluate_classification(points, labels, ground_truth)
        results[algo_name] = metrics
        
        # 保存可视化结果
        vis_dir = Path(output_dir) / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        # 3D可视化
        visualize_results_3d(
            points, labels, ground_truth,
            title=f"{algo_name} Classification Results",
            save_path=str(vis_dir / f"{algo_name}_3d.html")
        )
        
        # 保存点云数据
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(
            np.column_stack((labels, 1-labels, np.zeros_like(labels)))
        )
        o3d.io.write_point_cloud(
            str(vis_dir / f"{algo_name}_result.ply"),
            pcd
        )
    
    # 保存结果
    save_results(results, output_dir)
    
    # 生成性能对比图
    plot_performance_comparison(results, execution_times, output_dir)
    
    return results, execution_times

def parameter_sensitivity_analysis(points, ground_truth, parameters, output_dir):
    """
    参数敏感性分析
    
    Parameters
    ----------
    points : np.ndarray
        点云数据
    ground_truth : np.ndarray
        地面真值标签
    parameters : list
        要分析的参数列表
    output_dir : str
        输出目录
    """
    logger.info("开始参数敏感性分析...")
    
    # 参数范围定义
    param_ranges = {
        'cloth_resolution': np.linspace(0.5, 5, 10),
        'time_step': np.linspace(0.1, 1.0, 10),
        'class_threshold': np.linspace(0.1, 0.9, 9),
        'rigidness': np.linspace(1, 5, 5),
        'iterations': np.array([100, 200, 300, 400, 500])
    }
    
    results = {}
    
    for param in parameters:
        if param not in param_ranges:
            logger.warning(f"Unknown parameter: {param}")
            continue
        
        logger.info(f"分析参数: {param}")
        param_results = []
        
        for value in tqdm(param_ranges[param]):
            # 使用基础CSF算法进行测试
            algo = CSF()
            setattr(algo, param, value)
            
            # 运行算法
            labels = algo.classify(points)
            
            # 评估结果
            metrics = evaluate_classification(points, labels, ground_truth)
            metrics['param_value'] = value
            param_results.append(metrics)
        
        results[param] = param_results
    
    # 生成敏感性分析图
    fig = make_subplots(rows=len(parameters), cols=1,
                        subplot_titles=[f"Parameter: {p}" for p in parameters])
    
    for i, param in enumerate(parameters, 1):
        param_data = pd.DataFrame(results[param])
        
        fig.add_trace(
            go.Scatter(
                x=param_data['param_value'],
                y=param_data['accuracy'],
                name='Accuracy',
                mode='lines+markers'
            ),
            row=i, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=param_data['param_value'],
                y=param_data['f1_score'],
                name='F1 Score',
                mode='lines+markers'
            ),
            row=i, col=1
        )
    
    fig.update_layout(height=300*len(parameters),
                     title_text="Parameter Sensitivity Analysis")
    
    # 保存结果
    sensitivity_dir = Path(output_dir) / "sensitivity_analysis"
    sensitivity_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存图表
    fig.write_html(str(sensitivity_dir / "sensitivity_analysis.html"))
    
    # 保存原始数据
    with open(sensitivity_dir / "sensitivity_results.json", 'w') as f:
        json.dump(results, f, indent=4)
    
    return results

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='CSF Algorithm Test and Comparison')
    parser.add_argument('--data', type=str, help='Path to point cloud data file')
    parser.add_argument('--output-dir', type=str, default='examples/output',
                      help='Output directory for results')
    parser.add_argument('--algorithms', nargs='+', 
                      default=['CSF', 'MathematicalCSF', 'DifferentialCSF', 'ManifoldCSF'],
                      help='Algorithms to run')
    parser.add_argument('--sensitivity', action='store_true',
                      help='Run parameter sensitivity analysis')
    parser.add_argument('--parameters', nargs='+',
                      help='Parameters to analyze in sensitivity analysis')
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_arguments()
    
    # 设置输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    logger.info("Loading point cloud data...")
    points, ground_truth = load_sample_data(args.data)
    
    # 运行算法对比
    logger.info("Running algorithm comparison...")
    results, execution_times = compare_algorithms(
        points,
        ground_truth,
        args.algorithms,
        output_dir
    )
    
    # 参数敏感性分析
    if args.sensitivity:
        logger.info("Running parameter sensitivity analysis...")
        parameter_sensitivity_analysis(
            points,
            ground_truth,
            args.parameters,
            output_dir
        )
    
    logger.info("Test completed successfully!")

if __name__ == "__main__":
    main()