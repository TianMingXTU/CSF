#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
生成示例LAS点云数据

本脚本用于生成不同场景的LAS格式点云数据，包括：
1. 简单地形：平坦地形，有少量建筑物和树木
2. 复杂地形：起伏地形，有多个建筑物和树木
3. 城市地形：城市环境，有高楼、道路和树木
4. 森林地形：森林环境，有大量树木和少量建筑物
"""

import os
import numpy as np
import argparse
from scipy.spatial import cKDTree
import laspy
from tqdm import tqdm

def generate_ground_points(x_range, y_range, z_func, resolution=1.0):
    """
    生成地面点
    
    Parameters
    ----------
    x_range : tuple
        x坐标范围，格式为(min, max)
    y_range : tuple
        y坐标范围，格式为(min, max)
    z_func : function
        计算z坐标的函数，接受x和y作为参数
    resolution : float
        点云分辨率
        
    Returns
    -------
    points : np.ndarray
        地面点坐标，shape为(n_points, 3)
    """
    x_min, x_max = x_range
    y_min, y_max = y_range
    
    x = np.arange(x_min, x_max, resolution)
    y = np.arange(y_min, y_max, resolution)
    xx, yy = np.meshgrid(x, y)
    
    z = z_func(xx, yy)
    
    points = np.column_stack((xx.flatten(), yy.flatten(), z.flatten()))
    
    return points

def generate_building(x, y, width, length, height, base_height, n_points):
    """
    生成建筑物点
    
    Parameters
    ----------
    x, y : float
        建筑物中心坐标
    width, length : float
        建筑物宽度和长度
    height : float
        建筑物高度
    base_height : float
        建筑物底部高度
    n_points : int
        建筑物点数
        
    Returns
    -------
    points : np.ndarray
        建筑物点坐标，shape为(n_points, 3)
    """
    points = []
    
    # 生成建筑物主体
    for _ in range(n_points):
        px = x + np.random.uniform(-width/2, width/2)
        py = y + np.random.uniform(-length/2, length/2)
        pz = base_height + np.random.uniform(0, height)
        points.append([px, py, pz])
    
    return np.array(points)

def generate_tree(x, y, radius, height, base_height, n_points):
    """
    生成树木点
    
    Parameters
    ----------
    x, y : float
        树木中心坐标
    radius : float
        树木半径
    height : float
        树木高度
    base_height : float
        树木底部高度
    n_points : int
        树木点数
        
    Returns
    -------
    points : np.ndarray
        树木点坐标，shape为(n_points, 3)
    """
    points = []
    
    # 生成树木主体（近似为圆锥）
    for _ in range(n_points):
        r = radius * np.sqrt(np.random.uniform(0, 1))
        theta = np.random.uniform(0, 2*np.pi)
        px = x + r * np.cos(theta)
        py = y + r * np.sin(theta)
        pz = base_height + np.random.uniform(0, height * (1 - r/radius))
        points.append([px, py, pz])
    
    return np.array(points)

def generate_road(x_start, y_start, x_end, y_end, width, base_height, n_points):
    """
    生成道路点
    
    Parameters
    ----------
    x_start, y_start : float
        道路起点坐标
    x_end, y_end : float
        道路终点坐标
    width : float
        道路宽度
    base_height : float
        道路高度
    n_points : int
        道路点数
        
    Returns
    -------
    points : np.ndarray
        道路点坐标，shape为(n_points, 3)
    """
    points = []
    
    # 计算道路长度
    length = np.sqrt((x_end - x_start)**2 + (y_end - y_start)**2)
    
    # 计算道路方向
    dx = (x_end - x_start) / length
    dy = (y_end - y_start) / length
    
    # 生成道路点
    for _ in range(n_points):
        # 随机选择道路上的位置
        t = np.random.uniform(0, length)
        x = x_start + t * dx
        y = y_start + t * dy
        
        # 随机选择道路宽度上的位置
        w = np.random.uniform(-width/2, width/2)
        
        # 计算垂直于道路的方向
        px = x + w * dy
        py = y - w * dx
        pz = base_height + np.random.normal(0, 0.05)  # 添加一些噪声
        
        points.append([px, py, pz])
    
    return np.array(points)

def save_to_las(points, labels, output_file):
    """
    将点云数据保存为LAS格式
    
    Parameters
    ----------
    points : np.ndarray
        点云坐标，shape为(n_points, 3)
    labels : np.ndarray
        分类标签，1表示地面点，0表示非地面点
    output_file : str
        输出文件路径
    """
    # 创建LAS文件头
    header = laspy.LasHeader(point_format=2, version="1.2")
    header.point_count = len(points)
    
    # 设置坐标范围
    header.x_min = points[:, 0].min()
    header.x_max = points[:, 0].max()
    header.y_min = points[:, 1].min()
    header.y_max = points[:, 1].max()
    header.z_min = points[:, 2].min()
    header.z_max = points[:, 2].max()
    
    # 创建LAS数据对象
    las = laspy.LasData(header)
    
    # 设置点云坐标
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]
    
    # 设置分类标签 - 确保标签是整数类型
    # 在LAS标准中，2表示地面点，1表示未分类点
    classification = np.zeros(len(points), dtype=np.uint8)
    classification[labels == 1] = 2  # 地面点
    classification[labels == 0] = 1  # 非地面点
    las.classification = classification
    
    # 保存文件
    las.write(output_file)
    
    print(f"已保存LAS文件: {output_file}")

def generate_simple_terrain(output_file, n_points=100000):
    """
    生成简单地形点云数据
    
    Parameters
    ----------
    output_file : str
        输出文件路径
    n_points : int
        总点数
    """
    print("生成简单地形点云数据...")
    
    # 生成地面点
    def z_func(x, y):
        return 0.1 * np.sin(0.5 * x) * np.cos(0.5 * y)
    
    ground_points = generate_ground_points((-10, 10), (-10, 10), z_func, resolution=0.5)
    
    # 生成建筑物点
    n_buildings = 3
    n_points_per_building = n_points // (n_buildings + 10)  # 分配点数
    
    building_points = []
    for i in range(n_buildings):
        # 随机建筑物位置
        bx = np.random.uniform(-8, 8)
        by = np.random.uniform(-8, 8)
        bw = np.random.uniform(1, 2)
        bl = np.random.uniform(1, 2)
        bh = np.random.uniform(2, 4)
        
        # 找到建筑物底部的地面高度
        tree = cKDTree(ground_points[:, :2])
        _, idx = tree.query([bx, by])
        base_height = ground_points[idx, 2]
        
        # 生成建筑物点
        building = generate_building(bx, by, bw, bl, bh, base_height, n_points_per_building)
        building_points.append(building)
    
    # 生成树木点
    n_trees = 10
    n_points_per_tree = n_points // (n_buildings + n_trees + 5)  # 分配点数
    
    tree_points = []
    for i in range(n_trees):
        # 随机树木位置
        tx = np.random.uniform(-9, 9)
        ty = np.random.uniform(-9, 9)
        tr = np.random.uniform(0.3, 0.6)
        th = np.random.uniform(1, 2)
        
        # 找到树木底部的地面高度
        tree = cKDTree(ground_points[:, :2])
        _, idx = tree.query([tx, ty])
        base_height = ground_points[idx, 2]
        
        # 生成树木点
        tree = generate_tree(tx, ty, tr, th, base_height, n_points_per_tree)
        tree_points.append(tree)
    
    # 合并所有点
    all_points = np.vstack([ground_points] + building_points + tree_points)
    
    # 生成标签
    labels = np.zeros(len(all_points))
    labels[:len(ground_points)] = 1  # 地面点标签为1
    
    # 添加一些噪声
    noise = np.random.normal(0, 0.05, size=all_points.shape)
    all_points += noise
    
    # 保存为LAS文件
    save_to_las(all_points, labels, output_file)
    
    return all_points, labels

def generate_complex_terrain(output_file, n_points=200000):
    """
    生成复杂地形点云数据
    
    Parameters
    ----------
    output_file : str
        输出文件路径
    n_points : int
        总点数
    """
    print("生成复杂地形点云数据...")
    
    # 生成地面点
    def z_func(x, y):
        return 0.5 * np.sin(0.3 * x) * np.cos(0.3 * y) + 0.2 * np.sin(1.5 * x) * np.cos(1.5 * y)
    
    ground_points = generate_ground_points((-20, 20), (-20, 20), z_func, resolution=0.5)
    
    # 生成建筑物点
    n_buildings = 8
    n_points_per_building = n_points // (n_buildings + 15)  # 分配点数
    
    building_points = []
    for i in range(n_buildings):
        # 随机建筑物位置
        bx = np.random.uniform(-15, 15)
        by = np.random.uniform(-15, 15)
        bw = np.random.uniform(1, 3)
        bl = np.random.uniform(1, 3)
        bh = np.random.uniform(2, 6)
        
        # 找到建筑物底部的地面高度
        tree = cKDTree(ground_points[:, :2])
        _, idx = tree.query([bx, by])
        base_height = ground_points[idx, 2]
        
        # 生成建筑物点
        building = generate_building(bx, by, bw, bl, bh, base_height, n_points_per_building)
        building_points.append(building)
    
    # 生成树木点
    n_trees = 20
    n_points_per_tree = n_points // (n_buildings + n_trees + 10)  # 分配点数
    
    tree_points = []
    for i in range(n_trees):
        # 随机树木位置
        tx = np.random.uniform(-18, 18)
        ty = np.random.uniform(-18, 18)
        tr = np.random.uniform(0.3, 0.8)
        th = np.random.uniform(1, 3)
        
        # 找到树木底部的地面高度
        tree = cKDTree(ground_points[:, :2])
        _, idx = tree.query([tx, ty])
        base_height = ground_points[idx, 2]
        
        # 生成树木点
        tree = generate_tree(tx, ty, tr, th, base_height, n_points_per_tree)
        tree_points.append(tree)
    
    # 合并所有点
    all_points = np.vstack([ground_points] + building_points + tree_points)
    
    # 生成标签
    labels = np.zeros(len(all_points))
    labels[:len(ground_points)] = 1  # 地面点标签为1
    
    # 添加一些噪声
    noise = np.random.normal(0, 0.05, size=all_points.shape)
    all_points += noise
    
    # 保存为LAS文件
    save_to_las(all_points, labels, output_file)
    
    return all_points, labels

def generate_urban_terrain(output_file, n_points=300000):
    """
    生成城市地形点云数据
    
    Parameters
    ----------
    output_file : str
        输出文件路径
    n_points : int
        总点数
    """
    print("生成城市地形点云数据...")
    
    # 生成地面点
    def z_func(x, y):
        return 0.1 * np.sin(0.2 * x) * np.cos(0.2 * y)
    
    ground_points = generate_ground_points((-30, 30), (-30, 30), z_func, resolution=0.5)
    
    # 生成建筑物点
    n_buildings = 15
    n_points_per_building = n_points // (n_buildings + 20)  # 分配点数
    
    building_points = []
    for i in range(n_buildings):
        # 随机建筑物位置
        bx = np.random.uniform(-25, 25)
        by = np.random.uniform(-25, 25)
        bw = np.random.uniform(2, 5)
        bl = np.random.uniform(2, 5)
        bh = np.random.uniform(5, 15)
        
        # 找到建筑物底部的地面高度
        tree = cKDTree(ground_points[:, :2])
        _, idx = tree.query([bx, by])
        base_height = ground_points[idx, 2]
        
        # 生成建筑物点
        building = generate_building(bx, by, bw, bl, bh, base_height, n_points_per_building)
        building_points.append(building)
    
    # 生成树木点
    n_trees = 30
    n_points_per_tree = n_points // (n_buildings + n_trees + 15)  # 分配点数
    
    tree_points = []
    for i in range(n_trees):
        # 随机树木位置
        tx = np.random.uniform(-28, 28)
        ty = np.random.uniform(-28, 28)
        tr = np.random.uniform(0.3, 0.8)
        th = np.random.uniform(1, 3)
        
        # 找到树木底部的地面高度
        tree = cKDTree(ground_points[:, :2])
        _, idx = tree.query([tx, ty])
        base_height = ground_points[idx, 2]
        
        # 生成树木点
        tree = generate_tree(tx, ty, tr, th, base_height, n_points_per_tree)
        tree_points.append(tree)
    
    # 生成道路点
    n_roads = 5
    n_points_per_road = n_points // (n_buildings + n_trees + n_roads + 10)  # 分配点数
    
    road_points = []
    for i in range(n_roads):
        # 随机道路起点和终点
        x_start = np.random.uniform(-25, 25)
        y_start = np.random.uniform(-25, 25)
        x_end = np.random.uniform(-25, 25)
        y_end = np.random.uniform(-25, 25)
        
        # 找到道路底部的地面高度
        tree = cKDTree(ground_points[:, :2])
        _, idx = tree.query([(x_start + x_end)/2, (y_start + y_end)/2])
        base_height = ground_points[idx, 2]
        
        # 生成道路点
        road = generate_road(x_start, y_start, x_end, y_end, 3.0, base_height, n_points_per_road)
        road_points.append(road)
    
    # 合并所有点
    all_points = np.vstack([ground_points] + building_points + tree_points + road_points)
    
    # 生成标签
    labels = np.zeros(len(all_points))
    labels[:len(ground_points)] = 1  # 地面点标签为1
    
    # 添加一些噪声
    noise = np.random.normal(0, 0.05, size=all_points.shape)
    all_points += noise
    
    # 保存为LAS文件
    save_to_las(all_points, labels, output_file)
    
    return all_points, labels

def generate_forest_terrain(output_file, n_points=400000):
    """
    生成森林地形点云数据
    
    Parameters
    ----------
    output_file : str
        输出文件路径
    n_points : int
        总点数
    """
    print("生成森林地形点云数据...")
    
    # 生成地面点
    def z_func(x, y):
        return 0.3 * np.sin(0.1 * x) * np.cos(0.1 * y) + 0.1 * np.sin(0.5 * x) * np.cos(0.5 * y)
    
    ground_points = generate_ground_points((-40, 40), (-40, 40), z_func, resolution=0.5)
    
    # 生成建筑物点
    n_buildings = 5
    n_points_per_building = n_points // (n_buildings + 100)  # 分配点数
    
    building_points = []
    for i in range(n_buildings):
        # 随机建筑物位置
        bx = np.random.uniform(-35, 35)
        by = np.random.uniform(-35, 35)
        bw = np.random.uniform(2, 4)
        bl = np.random.uniform(2, 4)
        bh = np.random.uniform(3, 6)
        
        # 找到建筑物底部的地面高度
        tree = cKDTree(ground_points[:, :2])
        _, idx = tree.query([bx, by])
        base_height = ground_points[idx, 2]
        
        # 生成建筑物点
        building = generate_building(bx, by, bw, bl, bh, base_height, n_points_per_building)
        building_points.append(building)
    
    # 生成树木点
    n_trees = 100
    n_points_per_tree = n_points // (n_buildings + n_trees + 5)  # 分配点数
    
    tree_points = []
    for i in tqdm(range(n_trees), desc="生成树木"):
        # 随机树木位置
        tx = np.random.uniform(-38, 38)
        ty = np.random.uniform(-38, 38)
        tr = np.random.uniform(0.3, 1.0)
        th = np.random.uniform(2, 5)
        
        # 找到树木底部的地面高度
        tree = cKDTree(ground_points[:, :2])
        _, idx = tree.query([tx, ty])
        base_height = ground_points[idx, 2]
        
        # 生成树木点
        tree = generate_tree(tx, ty, tr, th, base_height, n_points_per_tree)
        tree_points.append(tree)
    
    # 合并所有点
    all_points = np.vstack([ground_points] + building_points + tree_points)
    
    # 生成标签
    labels = np.zeros(len(all_points))
    labels[:len(ground_points)] = 1  # 地面点标签为1
    
    # 添加一些噪声
    noise = np.random.normal(0, 0.05, size=all_points.shape)
    all_points += noise
    
    # 保存为LAS文件
    save_to_las(all_points, labels, output_file)
    
    return all_points, labels

def parse_arguments():
    """
    解析命令行参数
    
    Returns
    -------
    args : argparse.Namespace
        解析后的命令行参数
    """
    parser = argparse.ArgumentParser(description='生成示例LAS点云数据')
    
    parser.add_argument('--output-dir', type=str, default='sample_data',
                        help='输出目录，默认为sample_data')
    parser.add_argument('--terrain', type=str, choices=['simple', 'complex', 'urban', 'forest', 'all'],
                        default='all', help='要生成的地形类型，默认为all')
    parser.add_argument('--points', type=int, default=100000,
                        help='每种地形的点数，默认为100000')
    
    return parser.parse_args()

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 生成点云数据
    if args.terrain == 'simple' or args.terrain == 'all':
        output_file = os.path.join(args.output_dir, 'simple_terrain.las')
        generate_simple_terrain(output_file, args.points)
    
    if args.terrain == 'complex' or args.terrain == 'all':
        output_file = os.path.join(args.output_dir, 'complex_terrain.las')
        generate_complex_terrain(output_file, args.points * 2)
    
    if args.terrain == 'urban' or args.terrain == 'all':
        output_file = os.path.join(args.output_dir, 'urban_terrain.las')
        generate_urban_terrain(output_file, args.points * 3)
    
    if args.terrain == 'forest' or args.terrain == 'all':
        output_file = os.path.join(args.output_dir, 'forest_terrain.las')
        generate_forest_terrain(output_file, args.points * 4)
    
    print("\n所有点云数据生成完成！")
    print(f"数据保存在目录: {os.path.abspath(args.output_dir)}")

if __name__ == "__main__":
    main() 