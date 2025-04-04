# Enhanced CSF (Cloth Simulation Filter)

Enhanced CSF是一个基于布料模拟滤波(Cloth Simulation Filter)的点云地面提取算法库，提供了多种改进版本的CSF算法实现。

## 特点

- 多种CSF算法实现，包括基础版本和多种改进版本
- 基于Numba的高性能计算
- 丰富的数学模型和几何分析
- 简单易用的API
- 详细的文档和示例

## 安装

```bash
pip install -r requirements.txt
```

## 示例数据

本库提供了一个示例数据生成脚本 `generate_sample_data.py`，用于生成不同场景的LAS格式点云数据：

```bash
# 生成所有类型的地形数据
python generate_sample_data.py

# 生成特定类型的地形数据
python generate_sample_data.py --terrain simple
python generate_sample_data.py --terrain complex
python generate_sample_data.py --terrain urban
python generate_sample_data.py --terrain forest

# 指定输出目录和点数
python generate_sample_data.py --output-dir my_data --points 200000
```

生成的数据包括：

1. **简单地形**：平坦地形，有少量建筑物和树木
2. **复杂地形**：起伏地形，有多个建筑物和树木
3. **城市地形**：城市环境，有高楼、道路和树木
4. **森林地形**：森林环境，有大量树木和少量建筑物

## 算法实现

本库提供了以下CSF算法实现：

1. **基础CSF算法** (`CSF`)
   - 原始的布料模拟滤波算法
   - 适用于简单地形

2. **数学增强CSF算法** (`MathematicalCSF`)
   - 结合黎曼几何和拓扑分析
   - 使用高斯曲率和拓扑特征增强分类效果
   - 适用于复杂地形

3. **微分几何CSF算法** (`DifferentialCSF`)
   - 基于微分几何理论
   - 使用第一基本形式、第二基本形式、主曲率和平均曲率
   - 适用于具有复杂曲率变化的地形

4. **流形学习CSF算法** (`ManifoldCSF`)
   - 结合流形学习和几何分析
   - 使用局部线性嵌入(LLE)将点云映射到低维流形空间
   - 计算测地线曲率、法曲率和持久同伦特征
   - 适用于具有复杂拓扑结构的地形
   - 自适应权重根据局部几何和拓扑特征动态调整

## 使用方法

```python
import numpy as np
from enhanced_csf.core import CSF, MathematicalCSF, DifferentialCSF, ManifoldCSF

# 加载点云数据
points = np.load('point_cloud.npy')  # shape: (n_points, 3)

# 使用基础CSF算法
csf = CSF(cloth_resolution=2, time_step=0.65, class_threshold=0.5, iterations=500, rigidness=3)
labels = csf.classify(points)

# 使用数学增强CSF算法
math_csf = MathematicalCSF(cloth_resolution=2, time_step=0.65, class_threshold=0.5, 
                          iterations=500, rigidness=3, curvature_weight=0.3, topological_weight=0.2)
labels = math_csf.classify(points)

# 使用微分几何CSF算法
diff_csf = DifferentialCSF(cloth_resolution=2, time_step=0.65, class_threshold=0.5, 
                          iterations=500, rigidness=3, mean_curvature_weight=0.3, principal_curvature_weight=0.2)
labels = diff_csf.classify(points)

# 使用流形学习CSF算法
manifold_csf = ManifoldCSF(cloth_resolution=2, time_step=0.65, class_threshold=0.5, 
                          iterations=500, rigidness=3, manifold_dim=2, k_neighbors=10,
                          geodesic_weight=0.3, normal_weight=0.2, persistence_weight=0.1)
labels = manifold_csf.classify(points)
```

## 参数说明

### 基础参数

- `cloth_resolution`: 布料分辨率，控制布料的精细程度
- `time_step`: 模拟时间步长，控制收敛速度
- `class_threshold`: 分类阈值，控制地面点的判定标准
- `iterations`: 最大迭代次数
- `rigidness`: 布料刚性参数，控制布料的刚性程度

### 数学增强CSF参数

- `curvature_weight`: 曲率权重，控制曲率特征的影响程度
- `topological_weight`: 拓扑权重，控制拓扑特征的影响程度

### 微分几何CSF参数

- `mean_curvature_weight`: 平均曲率权重，控制平均曲率的影响程度
- `principal_curvature_weight`: 主曲率权重，控制主曲率的影响程度

### 流形学习CSF参数

- `manifold_dim`: 流形嵌入维度，控制嵌入空间的维度
- `k_neighbors`: 近邻点数量，控制局部几何特征的计算范围
- `geodesic_weight`: 测地线曲率权重，控制测地线曲率的影响程度
- `normal_weight`: 法曲率权重，控制法曲率的影响程度
- `persistence_weight`: 持久同伦特征权重，控制拓扑特征的影响程度

## 性能优化

所有算法实现都使用Numba进行性能优化，支持并行计算。对于大规模点云数据，建议使用以下参数设置：

```python
# 大规模点云数据的高性能设置
csf = CSF(cloth_resolution=5, time_step=0.65, class_threshold=0.5, iterations=200, rigidness=3)
```

## 算法测试与对比

本库提供了一个测试脚本 `test_csf_algorithms.py`，用于同时运行所有CSF算法实现，并进行多方面对比分析。

### 测试功能

1. **算法性能对比**：比较不同算法的计算时间和分类效果
2. **可视化结果**：可视化原始点云、分类结果和地面真值
3. **参数敏感性分析**：分析不同参数对算法性能的影响
4. **混淆矩阵分析**：分析分类结果的混淆矩阵

### 使用方法

```bash
# 运行测试脚本
python test_csf_algorithms.py

# 使用自定义点云数据
python test_csf_algorithms.py --data path/to/your/point_cloud.las
```

### 测试结果

测试脚本会生成以下结果：

1. **点云可视化**：显示原始点云、分类结果和地面真值
2. **混淆矩阵**：显示每个算法的分类混淆矩阵
3. **性能对比图**：比较不同算法的计算时间和分类指标
4. **参数敏感性图**：分析不同参数对算法性能的影响

## 参考文献

1. Zhang, J., Lin, X., & Ning, X. (2016). Cloth simulation filter for ground extraction from airborne LiDAR data. IEEE Geoscience and Remote Sensing Letters, 13(5), 692-696.
2. 微分几何与流形学习相关理论
3. 拓扑数据分析与持久同伦理论

## 许可证

MIT 