import numpy as np
import open3d as o3d
from typing import Dict, List, Tuple, Optional, Union, Any
import os
import logging
from tqdm import tqdm
import json
import pandas as pd
from pathlib import Path
import laspy

class PointCloudIO:
    """
    点云数据读写类，支持多种文件格式和数据处理功能
    
    支持的文件格式：
    - PLY
    - LAS/LAZ
    - PCD
    - XYZ
    - NPY
    - CSV
    - TXT
    """
    
    def __init__(self, verbose: bool = True):
        """
        初始化点云IO类
        
        Parameters
        ----------
        verbose : bool, optional
            是否显示详细信息
        """
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        
    def read_point_cloud(self, file_path: str) -> np.ndarray:
        """
        读取点云数据，支持多种文件格式
        
        Parameters
        ----------
        file_path : str
            点云文件路径
            
        Returns
        -------
        points : np.ndarray
            点云数据，shape为(n_points, 3)
        """
        self.logger.info(f"读取点云文件: {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_ext == '.ply':
                # 使用Open3D读取PLY文件
                pcd = o3d.io.read_point_cloud(file_path)
                points = np.asarray(pcd.points)
                
            elif file_ext in ['.las', '.laz']:
                # 使用LASpy读取LAS/LAZ文件
                las = laspy.read(file_path)
                points = np.vstack((las.x, las.y, las.z)).transpose()
                
            elif file_ext == '.pcd':
                # 使用Open3D读取PCD文件
                pcd = o3d.io.read_point_cloud(file_path)
                points = np.asarray(pcd.points)
                
            elif file_ext == '.xyz':
                # 使用NumPy读取XYZ文件
                points = np.loadtxt(file_path)
                
            elif file_ext == '.npy':
                # 读取NPY文件
                points = np.load(file_path)
                
            elif file_ext == '.csv':
                # 读取CSV文件
                df = pd.read_csv(file_path)
                points = df[['x', 'y', 'z']].values
                
            elif file_ext == '.txt':
                # 读取TXT文件
                points = np.loadtxt(file_path)
                
            else:
                raise ValueError(f"不支持的文件格式: {file_ext}")
                
            self.logger.info(f"成功读取点云，共{len(points)}个点")
            return points
            
        except Exception as e:
            self.logger.error(f"读取点云文件失败: {str(e)}")
            raise
    
    def write_point_cloud(self, points: np.ndarray, file_path: str) -> None:
        """
        保存点云数据，支持多种文件格式
        
        Parameters
        ----------
        points : np.ndarray
            点云数据，shape为(n_points, 3)
        file_path : str
            保存文件路径
        """
        self.logger.info(f"保存点云到: {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_ext == '.ply':
                # 保存为PLY文件
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                o3d.io.write_point_cloud(file_path, pcd)
                
            elif file_ext in ['.las', '.laz']:
                # 保存为LAS/LAZ文件
                header = laspy.LasHeader(point_format=2, version="1.2")
                las = laspy.LasData(header)
                las.x = points[:, 0]
                las.y = points[:, 1]
                las.z = points[:, 2]
                las.write(file_path)
                
            elif file_ext == '.pcd':
                # 保存为PCD文件
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                o3d.io.write_point_cloud(file_path, pcd)
                
            elif file_ext == '.xyz':
                # 保存为XYZ文件
                np.savetxt(file_path, points)
                
            elif file_ext == '.npy':
                # 保存为NPY文件
                np.save(file_path, points)
                
            elif file_ext == '.csv':
                # 保存为CSV文件
                df = pd.DataFrame(points, columns=['x', 'y', 'z'])
                df.to_csv(file_path, index=False)
                
            elif file_ext == '.txt':
                # 保存为TXT文件
                np.savetxt(file_path, points)
                
            else:
                raise ValueError(f"不支持的文件格式: {file_ext}")
                
            self.logger.info(f"成功保存点云，共{len(points)}个点")
            
        except Exception as e:
            self.logger.error(f"保存点云文件失败: {str(e)}")
            raise
    
    def read_labels(self, file_path: str) -> np.ndarray:
        """
        读取标签数据，支持多种文件格式
        
        Parameters
        ----------
        file_path : str
            标签文件路径
            
        Returns
        -------
        labels : np.ndarray
            标签数据
        """
        self.logger.info(f"读取标签文件: {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_ext == '.txt':
                # 读取TXT文件
                labels = np.loadtxt(file_path, dtype=int)
                
            elif file_ext == '.npy':
                # 读取NPY文件
                labels = np.load(file_path)
                
            elif file_ext == '.csv':
                # 读取CSV文件
                labels = pd.read_csv(file_path).values.flatten()
                
            else:
                raise ValueError(f"不支持的文件格式: {file_ext}")
                
            self.logger.info(f"成功读取标签，共{len(labels)}个标签")
            return labels
            
        except Exception as e:
            self.logger.error(f"读取标签文件失败: {str(e)}")
            raise
    
    def write_labels(self, labels: np.ndarray, file_path: str) -> None:
        """
        保存标签数据，支持多种文件格式
        
        Parameters
        ----------
        labels : np.ndarray
            标签数据
        file_path : str
            保存文件路径
        """
        self.logger.info(f"保存标签到: {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_ext == '.txt':
                # 保存为TXT文件
                np.savetxt(file_path, labels, fmt='%d')
                
            elif file_ext == '.npy':
                # 保存为NPY文件
                np.save(file_path, labels)
                
            elif file_ext == '.csv':
                # 保存为CSV文件
                pd.DataFrame(labels, columns=['label']).to_csv(file_path, index=False)
                
            else:
                raise ValueError(f"不支持的文件格式: {file_ext}")
                
            self.logger.info(f"成功保存标签，共{len(labels)}个标签")
            
        except Exception as e:
            self.logger.error(f"保存标签文件失败: {str(e)}")
            raise
    
    def read_metrics(self, file_path: str) -> Dict[str, Any]:
        """
        读取评估指标数据，支持多种文件格式
        
        Parameters
        ----------
        file_path : str
            指标文件路径
            
        Returns
        -------
        metrics : Dict[str, Any]
            评估指标字典
        """
        self.logger.info(f"读取评估指标文件: {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_ext == '.json':
                # 读取JSON文件
                with open(file_path, 'r') as f:
                    metrics = json.load(f)
                    
            elif file_ext == '.npy':
                # 读取NPY文件
                metrics = np.load(file_path, allow_pickle=True).item()
                
            elif file_ext == '.csv':
                # 读取CSV文件
                metrics = pd.read_csv(file_path).to_dict('records')[0]
                
            else:
                raise ValueError(f"不支持的文件格式: {file_ext}")
                
            self.logger.info("成功读取评估指标")
            return metrics
            
        except Exception as e:
            self.logger.error(f"读取评估指标文件失败: {str(e)}")
            raise
    
    def write_metrics(self, metrics: Dict[str, Any], file_path: str) -> None:
        """
        保存评估指标数据，支持多种文件格式
        
        Parameters
        ----------
        metrics : Dict[str, Any]
            评估指标字典
        file_path : str
            保存文件路径
        """
        self.logger.info(f"保存评估指标到: {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_ext == '.json':
                # 保存为JSON文件
                with open(file_path, 'w') as f:
                    json.dump(metrics, f, indent=4)
                    
            elif file_ext == '.npy':
                # 保存为NPY文件
                np.save(file_path, metrics)
                
            elif file_ext == '.csv':
                # 保存为CSV文件
                pd.DataFrame([metrics]).to_csv(file_path, index=False)
                
            else:
                raise ValueError(f"不支持的文件格式: {file_ext}")
                
            self.logger.info("成功保存评估指标")
            
        except Exception as e:
            self.logger.error(f"保存评估指标文件失败: {str(e)}")
            raise
    
    def batch_process(self, 
                     input_dir: str, 
                     output_dir: str, 
                     process_func: callable,
                     file_ext: str = '.ply') -> None:
        """
        批量处理点云文件
        
        Parameters
        ----------
        input_dir : str
            输入目录
        output_dir : str
            输出目录
        process_func : callable
            处理函数
        file_ext : str, optional
            文件扩展名
        """
        self.logger.info(f"开始批量处理点云文件: {input_dir}")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取所有点云文件
        input_files = list(Path(input_dir).glob(f"*{file_ext}"))
        
        # 使用tqdm显示进度
        for file_path in tqdm(input_files, desc="处理点云文件"):
            try:
                # 读取点云
                points = self.read_point_cloud(str(file_path))
                
                # 处理点云
                result = process_func(points)
                
                # 保存结果
                output_path = os.path.join(output_dir, file_path.name)
                self.write_point_cloud(result, output_path)
                
            except Exception as e:
                self.logger.error(f"处理文件{file_path}失败: {str(e)}")
                continue
                
        self.logger.info("批量处理完成") 