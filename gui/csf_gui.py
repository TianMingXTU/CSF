#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CSF算法GUI界面
提供参数调整、数据集加载/生成、算法选择和结果可视化功能
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QPushButton, QComboBox, QFileDialog, QTabWidget, 
                            QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox, QCheckBox,
                            QProgressBar, QMessageBox, QSplitter, QFrame, QScrollArea)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# 导入CSF算法
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from enhanced_csf.core import CSF, MathematicalCSF, DifferentialCSF, ManifoldCSF
from enhanced_csf.utils.io import PointCloudIO
from enhanced_csf.utils.visualization import PointCloudVisualizer

class AlgorithmThread(QThread):
    """算法执行线程，避免GUI卡顿"""
    progress = pyqtSignal(int)
    finished = pyqtSignal(object, object, float)  # 结果、标签、执行时间
    
    def __init__(self, algorithm, points, ground_truth=None):
        super().__init__()
        self.algorithm = algorithm
        self.points = points
        self.ground_truth = ground_truth
        
    def run(self):
        import time
        start_time = time.time()
        
        # 设置进度回调函数
        def progress_callback(progress):
            self.progress.emit(int(progress * 100))
        
        # 执行算法，传入进度回调函数
        if hasattr(self.algorithm, 'classify_with_progress'):
            labels = self.algorithm.classify_with_progress(self.points, progress_callback)
        else:
            # 如果算法不支持进度回调，则使用默认方法
            labels = self.algorithm.classify(self.points)
            # 模拟进度更新
            for i in range(10):
                self.progress.emit(i * 10)
                time.sleep(0.1)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # 发送结果
        self.finished.emit(self.points, labels, execution_time)


class CSFGUI(QMainWindow):
    """CSF算法GUI主窗口"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CSF算法可视化工具")
        self.setGeometry(100, 100, 1200, 800)
        
        # 初始化变量
        self.points = None
        self.ground_truth = None
        self.labels = None
        self.algorithm = None
        self.algorithm_thread = None
        self.visualizer = PointCloudVisualizer()
        
        # 创建UI
        self.init_ui()
        
    def init_ui(self):
        """初始化UI界面"""
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QHBoxLayout(central_widget)
        
        # 左侧控制面板
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        
        # 数据集控制
        dataset_group = QGroupBox("数据集")
        dataset_layout = QVBoxLayout(dataset_group)
        
        # 数据集类型选择
        dataset_type_layout = QHBoxLayout()
        dataset_type_layout.addWidget(QLabel("数据集类型:"))
        self.dataset_type_combo = QComboBox()
        self.dataset_type_combo.addItems(["生成模拟数据", "加载文件"])
        self.dataset_type_combo.currentIndexChanged.connect(self.on_dataset_type_changed)
        dataset_type_layout.addWidget(self.dataset_type_combo)
        dataset_layout.addLayout(dataset_type_layout)
        
        # 模拟数据参数
        self.sim_params_group = QGroupBox("模拟数据参数")
        sim_params_layout = QFormLayout(self.sim_params_group)
        
        self.sim_points_spin = QSpinBox()
        self.sim_points_spin.setRange(1000, 100000)
        self.sim_points_spin.setValue(20000)
        self.sim_points_spin.setSingleStep(1000)
        sim_params_layout.addRow("点数量:", self.sim_points_spin)
        
        self.sim_noise_spin = QDoubleSpinBox()
        self.sim_noise_spin.setRange(0.0, 1.0)
        self.sim_noise_spin.setValue(0.1)
        self.sim_noise_spin.setSingleStep(0.01)
        sim_params_layout.addRow("噪声水平:", self.sim_noise_spin)
        
        self.sim_terrain_combo = QComboBox()
        self.sim_terrain_combo.addItems(["平面", "正弦波", "丘陵"])
        sim_params_layout.addRow("地形类型:", self.sim_terrain_combo)
        
        dataset_layout.addWidget(self.sim_params_group)
        
        # 文件加载参数
        self.file_params_group = QGroupBox("文件加载参数")
        file_params_layout = QVBoxLayout(self.file_params_group)
        
        file_buttons_layout = QHBoxLayout()
        self.load_file_btn = QPushButton("加载点云文件")
        self.load_file_btn.clicked.connect(self.load_point_cloud)
        file_buttons_layout.addWidget(self.load_file_btn)
        
        self.load_ground_truth_btn = QPushButton("加载地面真值")
        self.load_ground_truth_btn.clicked.connect(self.load_ground_truth)
        self.load_ground_truth_btn.setEnabled(False)
        file_buttons_layout.addWidget(self.load_ground_truth_btn)
        
        file_params_layout.addLayout(file_buttons_layout)
        
        self.file_info_label = QLabel("未加载文件")
        file_params_layout.addWidget(self.file_info_label)
        
        dataset_layout.addWidget(self.file_params_group)
        
        # 生成/加载按钮
        self.generate_btn = QPushButton("生成模拟数据")
        self.generate_btn.clicked.connect(self.generate_simulated_data)
        dataset_layout.addWidget(self.generate_btn)
        
        control_layout.addWidget(dataset_group)
        
        # 算法选择
        algorithm_group = QGroupBox("算法选择")
        algorithm_layout = QVBoxLayout(algorithm_group)
        
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems(["CSF", "MathematicalCSF", "DifferentialCSF", "ManifoldCSF"])
        self.algorithm_combo.currentIndexChanged.connect(self.on_algorithm_changed)
        algorithm_layout.addWidget(self.algorithm_combo)
        
        # 算法参数
        self.algorithm_params_group = QGroupBox("算法参数")
        self.algorithm_params_layout = QFormLayout(self.algorithm_params_group)
        
        # 通用参数
        self.cloth_resolution_spin = QDoubleSpinBox()
        self.cloth_resolution_spin.setRange(0.1, 2.0)
        self.cloth_resolution_spin.setValue(0.5)
        self.cloth_resolution_spin.setSingleStep(0.1)
        self.algorithm_params_layout.addRow("布料分辨率:", self.cloth_resolution_spin)
        
        self.time_step_spin = QDoubleSpinBox()
        self.time_step_spin.setRange(0.1, 1.0)
        self.time_step_spin.setValue(0.65)
        self.time_step_spin.setSingleStep(0.05)
        self.algorithm_params_layout.addRow("时间步长:", self.time_step_spin)
        
        self.class_threshold_spin = QDoubleSpinBox()
        self.class_threshold_spin.setRange(0.1, 1.0)
        self.class_threshold_spin.setValue(0.5)
        self.class_threshold_spin.setSingleStep(0.1)
        self.algorithm_params_layout.addRow("分类阈值:", self.class_threshold_spin)
        
        self.iterations_spin = QSpinBox()
        self.iterations_spin.setRange(100, 1000)
        self.iterations_spin.setValue(500)
        self.iterations_spin.setSingleStep(50)
        self.algorithm_params_layout.addRow("迭代次数:", self.iterations_spin)
        
        self.rigidness_spin = QSpinBox()
        self.rigidness_spin.setRange(1, 10)
        self.rigidness_spin.setValue(1)
        self.rigidness_spin.setSingleStep(1)
        self.algorithm_params_layout.addRow("刚性参数:", self.rigidness_spin)
        
        algorithm_layout.addWidget(self.algorithm_params_group)
        
        # 数学CSF特有参数
        self.math_params_group = QGroupBox("数学CSF参数")
        math_params_layout = QFormLayout(self.math_params_group)
        
        self.curvature_weight_spin = QDoubleSpinBox()
        self.curvature_weight_spin.setRange(0.0, 1.0)
        self.curvature_weight_spin.setValue(0.3)
        self.curvature_weight_spin.setSingleStep(0.1)
        math_params_layout.addRow("曲率权重:", self.curvature_weight_spin)
        
        self.topological_weight_spin = QDoubleSpinBox()
        self.topological_weight_spin.setRange(0.0, 1.0)
        self.topological_weight_spin.setValue(0.2)
        self.topological_weight_spin.setSingleStep(0.1)
        math_params_layout.addRow("拓扑权重:", self.topological_weight_spin)
        
        self.spectral_weight_spin = QDoubleSpinBox()
        self.spectral_weight_spin.setRange(0.0, 1.0)
        self.spectral_weight_spin.setValue(0.1)
        self.spectral_weight_spin.setSingleStep(0.1)
        math_params_layout.addRow("谱权重:", self.spectral_weight_spin)
        
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(100, 10000)
        self.batch_size_spin.setValue(1000)
        self.batch_size_spin.setSingleStep(100)
        math_params_layout.addRow("批处理大小:", self.batch_size_spin)
        
        self.stability_factor_spin = QDoubleSpinBox()
        self.stability_factor_spin.setRange(1e-8, 1e-4)
        self.stability_factor_spin.setValue(1e-6)
        self.stability_factor_spin.setSingleStep(1e-7)
        self.stability_factor_spin.setDecimals(8)
        math_params_layout.addRow("稳定性因子:", self.stability_factor_spin)
        
        self.n_neighbors_spin = QSpinBox()
        self.n_neighbors_spin.setRange(3, 50)
        self.n_neighbors_spin.setValue(10)
        self.n_neighbors_spin.setSingleStep(1)
        math_params_layout.addRow("邻居点数量:", self.n_neighbors_spin)
        
        algorithm_layout.addWidget(self.math_params_group)
        self.math_params_group.setVisible(False)
        
        # 微分CSF特有参数
        self.diff_params_group = QGroupBox("微分CSF参数")
        diff_params_layout = QFormLayout(self.diff_params_group)
        
        self.mean_curvature_weight_spin = QDoubleSpinBox()
        self.mean_curvature_weight_spin.setRange(0.0, 1.0)
        self.mean_curvature_weight_spin.setValue(0.3)
        self.mean_curvature_weight_spin.setSingleStep(0.1)
        diff_params_layout.addRow("平均曲率权重:", self.mean_curvature_weight_spin)
        
        self.principal_curvature_weight_spin = QDoubleSpinBox()
        self.principal_curvature_weight_spin.setRange(0.0, 1.0)
        self.principal_curvature_weight_spin.setValue(0.2)
        self.principal_curvature_weight_spin.setSingleStep(0.1)
        diff_params_layout.addRow("主曲率权重:", self.principal_curvature_weight_spin)
        
        self.gaussian_curvature_weight_spin = QDoubleSpinBox()
        self.gaussian_curvature_weight_spin.setRange(0.0, 1.0)
        self.gaussian_curvature_weight_spin.setValue(0.1)
        self.gaussian_curvature_weight_spin.setSingleStep(0.1)
        diff_params_layout.addRow("高斯曲率权重:", self.gaussian_curvature_weight_spin)
        
        self.diff_batch_size_spin = QSpinBox()
        self.diff_batch_size_spin.setRange(100, 10000)
        self.diff_batch_size_spin.setValue(1000)
        self.diff_batch_size_spin.setSingleStep(100)
        diff_params_layout.addRow("批处理大小:", self.diff_batch_size_spin)
        
        self.diff_stability_factor_spin = QDoubleSpinBox()
        self.diff_stability_factor_spin.setRange(1e-8, 1e-4)
        self.diff_stability_factor_spin.setValue(1e-6)
        self.diff_stability_factor_spin.setSingleStep(1e-7)
        self.diff_stability_factor_spin.setDecimals(8)
        diff_params_layout.addRow("稳定性因子:", self.diff_stability_factor_spin)
        
        algorithm_layout.addWidget(self.diff_params_group)
        self.diff_params_group.setVisible(False)
        
        # 流形CSF特有参数
        self.manifold_params_group = QGroupBox("流形CSF参数")
        manifold_params_layout = QFormLayout(self.manifold_params_group)
        
        self.manifold_dim_spin = QSpinBox()
        self.manifold_dim_spin.setRange(1, 5)
        self.manifold_dim_spin.setValue(2)
        self.manifold_dim_spin.setSingleStep(1)
        manifold_params_layout.addRow("流形维度:", self.manifold_dim_spin)
        
        self.manifold_n_neighbors_spin = QSpinBox()
        self.manifold_n_neighbors_spin.setRange(3, 50)
        self.manifold_n_neighbors_spin.setValue(10)
        self.manifold_n_neighbors_spin.setSingleStep(1)
        manifold_params_layout.addRow("邻居点数量:", self.manifold_n_neighbors_spin)
        
        self.manifold_batch_size_spin = QSpinBox()
        self.manifold_batch_size_spin.setRange(100, 10000)
        self.manifold_batch_size_spin.setValue(1000)
        self.manifold_batch_size_spin.setSingleStep(100)
        manifold_params_layout.addRow("批处理大小:", self.manifold_batch_size_spin)
        
        self.spectral_radius_spin = QDoubleSpinBox()
        self.spectral_radius_spin.setRange(0.1, 10.0)
        self.spectral_radius_spin.setValue(1.0)
        self.spectral_radius_spin.setSingleStep(0.1)
        manifold_params_layout.addRow("谱半径:", self.spectral_radius_spin)
        
        self.manifold_stability_factor_spin = QDoubleSpinBox()
        self.manifold_stability_factor_spin.setRange(1e-8, 1e-4)
        self.manifold_stability_factor_spin.setValue(1e-6)
        self.manifold_stability_factor_spin.setSingleStep(1e-7)
        self.manifold_stability_factor_spin.setDecimals(8)
        manifold_params_layout.addRow("稳定性因子:", self.manifold_stability_factor_spin)
        
        self.persistence_threshold_spin = QDoubleSpinBox()
        self.persistence_threshold_spin.setRange(0.0, 1.0)
        self.persistence_threshold_spin.setValue(0.05)
        self.persistence_threshold_spin.setSingleStep(0.01)
        manifold_params_layout.addRow("持久性阈值:", self.persistence_threshold_spin)
        
        self.manifold_weight_spin = QDoubleSpinBox()
        self.manifold_weight_spin.setRange(0.0, 1.0)
        self.manifold_weight_spin.setValue(0.3)
        self.manifold_weight_spin.setSingleStep(0.1)
        manifold_params_layout.addRow("流形权重:", self.manifold_weight_spin)
        
        algorithm_layout.addWidget(self.manifold_params_group)
        self.manifold_params_group.setVisible(False)
        
        control_layout.addWidget(algorithm_group)
        
        # 执行控制
        execution_group = QGroupBox("执行控制")
        execution_layout = QVBoxLayout(execution_group)
        
        self.run_btn = QPushButton("运行算法")
        self.run_btn.clicked.connect(self.run_algorithm)
        self.run_btn.setEnabled(False)
        execution_layout.addWidget(self.run_btn)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        execution_layout.addWidget(self.progress_bar)
        
        control_layout.addWidget(execution_group)
        
        # 右侧可视化面板
        visualization_panel = QWidget()
        visualization_layout = QVBoxLayout(visualization_panel)
        
        # 可视化控制
        vis_control_layout = QHBoxLayout()
        
        self.vis_type_combo = QComboBox()
        self.vis_type_combo.addItems(["原始点云", "分类结果", "地面点", "非地面点", "误差分析"])
        self.vis_type_combo.currentIndexChanged.connect(self.update_visualization)
        vis_control_layout.addWidget(self.vis_type_combo)
        
        self.vis_ground_truth_check = QCheckBox("显示地面真值")
        self.vis_ground_truth_check.stateChanged.connect(self.update_visualization)
        vis_control_layout.addWidget(self.vis_ground_truth_check)
        
        self.vis_metrics_btn = QPushButton("计算指标")
        self.vis_metrics_btn.clicked.connect(self.calculate_metrics)
        self.vis_metrics_btn.setEnabled(False)
        vis_control_layout.addWidget(self.vis_metrics_btn)
        
        self.save_results_btn = QPushButton("保存结果")
        self.save_results_btn.clicked.connect(self.save_results)
        self.save_results_btn.setEnabled(False)
        vis_control_layout.addWidget(self.save_results_btn)
        
        visualization_layout.addLayout(vis_control_layout)
        
        # 可视化区域
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        visualization_layout.addWidget(self.canvas)
        
        # 指标显示区域
        self.metrics_text = QLabel("指标将在这里显示")
        self.metrics_text.setAlignment(Qt.AlignCenter)
        visualization_layout.addWidget(self.metrics_text)
        
        # 使用分割器组织左右面板
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(control_panel)
        splitter.addWidget(visualization_panel)
        splitter.setSizes([300, 900])  # 设置初始分割比例
        
        main_layout.addWidget(splitter)
        
        # 初始化UI状态
        self.on_dataset_type_changed(0)
        
    def on_dataset_type_changed(self, index):
        """数据集类型改变时的处理"""
        if index == 0:  # 生成模拟数据
            self.sim_params_group.setVisible(True)
            self.file_params_group.setVisible(False)
            self.generate_btn.setText("生成模拟数据")
            self.generate_btn.clicked.disconnect()
            self.generate_btn.clicked.connect(self.generate_simulated_data)
        else:  # 加载文件
            self.sim_params_group.setVisible(False)
            self.file_params_group.setVisible(True)
            self.generate_btn.setText("加载文件")
            self.generate_btn.clicked.disconnect()
            self.generate_btn.clicked.connect(self.load_point_cloud)
            
    def on_algorithm_changed(self, index):
        """算法选择改变时的处理"""
        # 隐藏所有特定算法参数组
        self.math_params_group.setVisible(False)
        self.diff_params_group.setVisible(False)
        self.manifold_params_group.setVisible(False)
        
        # 显示选中算法的参数组
        if index == 1:  # MathematicalCSF
            self.math_params_group.setVisible(True)
        elif index == 2:  # DifferentialCSF
            self.diff_params_group.setVisible(True)
        elif index == 3:  # ManifoldCSF
            self.manifold_params_group.setVisible(True)
            
    def generate_simulated_data(self):
        """生成模拟点云数据"""
        try:
            # 获取参数
            n_points = self.sim_points_spin.value()
            noise_level = self.sim_noise_spin.value()
            terrain_type = self.sim_terrain_combo.currentText()
            
            # 生成地面点
            x = np.linspace(-10, 10, int(np.sqrt(n_points/2)))
            y = np.linspace(-10, 10, int(np.sqrt(n_points/2)))
            xx, yy = np.meshgrid(x, y)
            
            # 根据地形类型生成z坐标
            if terrain_type == "平面":
                zz = np.zeros_like(xx)
            elif terrain_type == "正弦波":
                zz = 2 * np.sin(xx/2) * np.cos(yy/2)
            else:  # 丘陵
                zz = 3 * np.exp(-(xx**2 + yy**2) / 20) + 1.5 * np.exp(-((xx-5)**2 + (yy+5)**2) / 10)
            
            # 展平并组合坐标
            ground_points = np.column_stack((xx.flatten(), yy.flatten(), zz.flatten()))
            
            # 生成非地面点（建筑物和树木）
            n_non_ground = n_points - len(ground_points)
            non_ground_points = np.zeros((n_non_ground, 3))
            
            # 随机生成建筑物
            n_buildings = n_non_ground // 2
            for i in range(n_buildings):
                # 随机建筑物位置和尺寸
                x_pos = np.random.uniform(-9, 9)
                y_pos = np.random.uniform(-9, 9)
                width = np.random.uniform(0.5, 2.0)
                length = np.random.uniform(0.5, 2.0)
                height = np.random.uniform(2.0, 5.0)
                
                # 找到地面高度
                idx_x = np.abs(x - x_pos).argmin()
                idx_y = np.abs(y - y_pos).argmin()
                base_height = zz[idx_y, idx_x]
                
                # 生成建筑物点
                x_building = np.random.uniform(x_pos - width/2, x_pos + width/2, 1)
                y_building = np.random.uniform(y_pos - length/2, y_pos + length/2, 1)
                z_building = np.random.uniform(base_height, base_height + height, 1)
                
                non_ground_points[i] = [x_building, y_building, z_building]
            
            # 随机生成树木
            for i in range(n_buildings, n_non_ground):
                # 随机树木位置和尺寸
                x_pos = np.random.uniform(-9, 9)
                y_pos = np.random.uniform(-9, 9)
                radius = np.random.uniform(0.3, 0.8)
                height = np.random.uniform(3.0, 7.0)
                
                # 找到地面高度
                idx_x = np.abs(x - x_pos).argmin()
                idx_y = np.abs(y - y_pos).argmin()
                base_height = zz[idx_y, idx_x]
                
                # 生成树木点
                angle = np.random.uniform(0, 2*np.pi, 1)
                r = np.random.uniform(0, radius, 1)
                x_tree = x_pos + r * np.cos(angle)
                y_tree = y_pos + r * np.sin(angle)
                z_tree = base_height + np.random.uniform(0, height, 1)
                
                non_ground_points[i] = [x_tree, y_tree, z_tree]
            
            # 合并点云
            self.points = np.vstack((ground_points, non_ground_points))
            
            # 添加噪声
            if noise_level > 0:
                noise = np.random.normal(0, noise_level, self.points.shape)
                self.points += noise
            
            # 生成地面真值标签
            self.ground_truth = np.zeros(len(self.points))
            self.ground_truth[:len(ground_points)] = 1
            
            # 更新UI
            self.file_info_label.setText(f"已生成模拟数据: {len(self.points)}个点")
            self.run_btn.setEnabled(True)
            self.vis_ground_truth_check.setEnabled(True)
            
            # 显示原始点云
            self.update_visualization()
            
            QMessageBox.information(self, "成功", f"已生成{len(self.points)}个模拟点云数据点")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"生成模拟数据时出错: {str(e)}")
            
    def load_point_cloud(self):
        """加载点云文件"""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择点云文件", "", 
                "点云文件 (*.npy *.txt *.mat *.las *.ply);;所有文件 (*.*)"
            )
            
            if not file_path:
                return
                
            # 加载点云
            io = PointCloudIO()
            self.points = io.read_point_cloud(file_path)
            
            # 更新UI
            self.file_info_label.setText(f"已加载文件: {os.path.basename(file_path)}, {len(self.points)}个点")
            self.load_ground_truth_btn.setEnabled(True)
            self.run_btn.setEnabled(True)
            
            # 显示原始点云
            self.update_visualization()
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载点云文件时出错: {str(e)}")
            
    def load_ground_truth(self):
        """加载地面真值标签"""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择地面真值文件", "", 
                "标签文件 (*.npy *.txt);;所有文件 (*.*)"
            )
            
            if not file_path:
                return
                
            # 加载地面真值
            io = PointCloudIO()
            self.ground_truth = io.read_labels(file_path)
            
            # 检查标签数量是否匹配
            if len(self.ground_truth) != len(self.points):
                QMessageBox.warning(self, "警告", "地面真值标签数量与点云数量不匹配")
                return
                
            # 更新UI
            self.vis_ground_truth_check.setEnabled(True)
            self.vis_ground_truth_check.setChecked(True)
            
            # 更新可视化
            self.update_visualization()
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载地面真值文件时出错: {str(e)}")
            
    def create_algorithm(self):
        """根据当前参数创建算法实例"""
        algorithm_type = self.algorithm_combo.currentText()
        
        # 通用参数
        cloth_resolution = self.cloth_resolution_spin.value()
        time_step = self.time_step_spin.value()
        class_threshold = self.class_threshold_spin.value()
        iterations = self.iterations_spin.value()
        rigidness = self.rigidness_spin.value()
        
        # 创建算法实例
        if algorithm_type == "CSF":
            return CSF(
                cloth_resolution=cloth_resolution,
                time_step=time_step,
                class_threshold=class_threshold,
                iterations=iterations,
                rigidness=rigidness
            )
        elif algorithm_type == "MathematicalCSF":
            return MathematicalCSF(
                cloth_resolution=cloth_resolution,
                time_step=time_step,
                class_threshold=class_threshold,
                iterations=iterations,
                rigidness=rigidness,
                curvature_weight=self.curvature_weight_spin.value(),
                topological_weight=self.topological_weight_spin.value(),
                spectral_weight=self.spectral_weight_spin.value(),
                batch_size=self.batch_size_spin.value(),
                stability_factor=self.stability_factor_spin.value(),
                n_neighbors=self.n_neighbors_spin.value()
            )
        elif algorithm_type == "DifferentialCSF":
            return DifferentialCSF(
                cloth_resolution=cloth_resolution,
                time_step=time_step,
                class_threshold=class_threshold,
                iterations=iterations,
                rigidness=rigidness,
                mean_curvature_weight=self.mean_curvature_weight_spin.value(),
                principal_curvature_weight=self.principal_curvature_weight_spin.value(),
                gaussian_curvature_weight=self.gaussian_curvature_weight_spin.value(),
                batch_size=self.diff_batch_size_spin.value(),
                stability_factor=self.diff_stability_factor_spin.value()
            )
        else:  # ManifoldCSF
            return ManifoldCSF(
                cloth_resolution=cloth_resolution,
                time_step=time_step,
                class_threshold=class_threshold,
                iterations=iterations,
                rigidness=rigidness,
                manifold_dim=self.manifold_dim_spin.value(),
                n_neighbors=self.manifold_n_neighbors_spin.value(),
                batch_size=self.manifold_batch_size_spin.value(),
                spectral_radius=self.spectral_radius_spin.value(),
                stability_factor=self.manifold_stability_factor_spin.value(),
                persistence_threshold=self.persistence_threshold_spin.value(),
                manifold_weight=self.manifold_weight_spin.value()
            )
            
    def run_algorithm(self):
        """运行选定的算法"""
        if self.points is None:
            QMessageBox.warning(self, "警告", "请先加载或生成点云数据")
            return
            
        try:
            # 创建算法实例
            self.algorithm = self.create_algorithm()
            
            # 禁用UI
            self.run_btn.setEnabled(False)
            self.progress_bar.setValue(0)
            
            # 创建并启动算法线程
            self.algorithm_thread = AlgorithmThread(self.algorithm, self.points, self.ground_truth)
            self.algorithm_thread.progress.connect(self.update_progress)
            self.algorithm_thread.finished.connect(self.on_algorithm_finished)
            self.algorithm_thread.start()
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"运行算法时出错: {str(e)}")
            self.run_btn.setEnabled(True)
            
    def update_progress(self, value):
        """更新进度条"""
        self.progress_bar.setValue(value)
        
    def on_algorithm_finished(self, points, labels, execution_time):
        """算法执行完成的处理"""
        # 保存结果
        self.points = points
        self.labels = labels
        
        # 更新UI
        self.run_btn.setEnabled(True)
        self.progress_bar.setValue(100)
        self.vis_metrics_btn.setEnabled(True)
        self.save_results_btn.setEnabled(True)
        
        # 显示结果
        self.vis_type_combo.setCurrentIndex(1)  # 切换到分类结果视图
        self.update_visualization()
        
        # 显示执行时间
        QMessageBox.information(self, "完成", f"算法执行完成，耗时: {execution_time:.2f}秒")
        
    def update_visualization(self):
        """更新可视化"""
        if self.points is None:
            return
            
        # 清除图形
        self.figure.clear()
        
        # 获取可视化类型
        vis_type = self.vis_type_combo.currentText()
        
        # 创建子图
        ax = self.figure.add_subplot(111, projection='3d')
        
        # 根据可视化类型显示不同的内容
        if vis_type == "原始点云":
            # 显示原始点云
            ax.scatter(self.points[:, 0], self.points[:, 1], self.points[:, 2], 
                      c='b', marker='.', s=1, alpha=0.5)
            
            # 如果有地面真值且选中显示，则用不同颜色标记
            if self.ground_truth is not None and self.vis_ground_truth_check.isChecked():
                ground_indices = np.where(self.ground_truth == 1)[0]
                non_ground_indices = np.where(self.ground_truth == 0)[0]
                
                ax.scatter(self.points[ground_indices, 0], self.points[ground_indices, 1], self.points[ground_indices, 2],
                          c='g', marker='.', s=1, alpha=0.7, label='地面点')
                ax.scatter(self.points[non_ground_indices, 0], self.points[non_ground_indices, 1], self.points[non_ground_indices, 2],
                          c='r', marker='.', s=1, alpha=0.7, label='非地面点')
                
                ax.legend()
                
        elif vis_type == "分类结果" and self.labels is not None:
            # 显示分类结果
            ground_indices = np.where(self.labels == 1)[0]
            non_ground_indices = np.where(self.labels == 0)[0]
            
            ax.scatter(self.points[ground_indices, 0], self.points[ground_indices, 1], self.points[ground_indices, 2],
                      c='g', marker='.', s=1, alpha=0.7, label='地面点')
            ax.scatter(self.points[non_ground_indices, 0], self.points[non_ground_indices, 1], self.points[non_ground_indices, 2],
                      c='r', marker='.', s=1, alpha=0.7, label='非地面点')
            
            ax.legend()
            
        elif vis_type == "地面点" and self.labels is not None:
            # 只显示地面点
            ground_indices = np.where(self.labels == 1)[0]
            ax.scatter(self.points[ground_indices, 0], self.points[ground_indices, 1], self.points[ground_indices, 2],
                      c='g', marker='.', s=1, alpha=0.7, label='地面点')
            ax.legend()
            
        elif vis_type == "非地面点" and self.labels is not None:
            # 只显示非地面点
            non_ground_indices = np.where(self.labels == 0)[0]
            ax.scatter(self.points[non_ground_indices, 0], self.points[non_ground_indices, 1], self.points[non_ground_indices, 2],
                      c='r', marker='.', s=1, alpha=0.7, label='非地面点')
            ax.legend()
            
        elif vis_type == "误差分析" and self.labels is not None and self.ground_truth is not None:
            # 显示分类误差
            errors = np.abs(self.labels - self.ground_truth)
            error_indices = np.where(errors == 1)[0]
            correct_indices = np.where(errors == 0)[0]
            
            ax.scatter(self.points[correct_indices, 0], self.points[correct_indices, 1], self.points[correct_indices, 2],
                      c='g', marker='.', s=1, alpha=0.7, label='正确分类')
            ax.scatter(self.points[error_indices, 0], self.points[error_indices, 1], self.points[error_indices, 2],
                      c='r', marker='.', s=1, alpha=0.7, label='错误分类')
            
            ax.legend()
            
        # 设置坐标轴标签
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # 更新画布
        self.canvas.draw()

    def calculate_metrics(self):
        """计算分类指标"""
        if self.labels is None or self.ground_truth is None:
            QMessageBox.warning(self, "警告", "需要分类结果和地面真值才能计算指标")
            return
            
        try:
            # 计算混淆矩阵
            tp = np.sum((self.labels == 1) & (self.ground_truth == 1))
            fp = np.sum((self.labels == 1) & (self.ground_truth == 0))
            fn = np.sum((self.labels == 0) & (self.ground_truth == 1))
            tn = np.sum((self.labels == 0) & (self.ground_truth == 0))
            
            # 计算指标
            accuracy = (tp + tn) / len(self.labels)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # 显示指标
            metrics_text = f"""
            混淆矩阵:
            TP: {tp}, FP: {fp}
            FN: {fn}, TN: {tn}
            
            指标:
            准确率 (Accuracy): {accuracy:.4f}
            精确率 (Precision): {precision:.4f}
            召回率 (Recall): {recall:.4f}
            F1分数: {f1:.4f}
            """
            
            self.metrics_text.setText(metrics_text)
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"计算指标时出错: {str(e)}")
            
    def save_results(self):
        """保存分类结果"""
        if self.labels is None:
            QMessageBox.warning(self, "警告", "没有可保存的分类结果")
            return
            
        try:
            # 选择保存路径
            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存分类结果", "", 
                "标签文件 (*.npy *.txt);;所有文件 (*.*)"
            )
            
            if not file_path:
                return
                
            # 保存标签
            io = PointCloudIO()
            io.write_labels(self.labels, file_path)
            
            QMessageBox.information(self, "成功", "分类结果已保存")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存结果时出错: {str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CSFGUI()
    window.show()
    sys.exit(app.exec_()) 