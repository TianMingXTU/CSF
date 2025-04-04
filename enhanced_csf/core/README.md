# Enhanced CSF Algorithm Library

这是一个增强版的CSF（Cloth Simulation Filter）算法库，提供了多种改进的CSF算法实现。

## 算法特点

1. **基础CSF算法**
   - 使用布料模拟方法对点云进行地面点分类
   - 自适应更新布料位置
   - 参数可调，适应不同场景

2. **数学CSF算法**
   - 结合黎曼几何、拓扑分析和谱分析
   - 使用黎曼度量张量和协变导数分析局部几何特征
   - 使用持久同伦分析提取拓扑特征
   - 使用拉普拉斯算子分析全局结构

3. **微分CSF算法**
   - 使用微分几何方法分析点云表面
   - 计算Weingarten映射和主曲率
   - 自适应权重更新

4. **流形CSF算法**
   - 将点云视为流形进行处理
   - 计算测地线路径和局部曲率
   - 使用流形特征进行地面点分类

5. **自适应CSF算法**
   - 根据局部坡度动态调整更新步长
   - 使用SVD计算局部平面法向量
   - 自适应参数优化

6. **多尺度CSF算法**
   - 使用不同尺度的局部特征
   - 根据局部曲率动态调整尺度权重
   - 多尺度特征融合

## 性能优化

1. **Numba加速**
   - 使用Numba JIT编译加速计算密集型操作
   - 支持并行计算
   - 优化内存使用

2. **缓存机制**
   - 使用LRU缓存减少重复计算
   - 优化布料网格创建
   - 提高算法效率

3. **进度显示**
   - 使用tqdm显示计算进度
   - 提供详细的日志信息
   - 方便调试和监控

## 使用方法

```python
from enhanced_csf.core import CSF, MathematicalCSF, DifferentialCSF, ManifoldCSF, AdaptiveCSF, MultiscaleCSF

# 创建算法实例
csf = CSF(
    cloth_resolution=0.5,
    time_step=0.65,
    class_threshold=0.5,
    iterations=500,
    rigidness=1
)

# 对点云进行分类
labels = csf.classify(points)
```

## 参数说明

### 通用参数
- `cloth_resolution`: 布料分辨率
- `time_step`: 模拟时间步长
- `class_threshold`: 分类阈值
- `iterations`: 最大迭代次数
- `rigidness`: 布料刚性参数

### 特殊参数
- `n_neighbors`: 计算局部特征时使用的邻居点数量
- `curvature_weight`: 曲率权重
- `topological_weight`: 拓扑权重
- `spectral_weight`: 谱权重
- `manifold_weight`: 流形特征权重
- `scales`: 多尺度分析的尺度列表

## 依赖库
- NumPy
- SciPy
- scikit-learn
- Numba
- tqdm

## 注意事项
1. 确保输入点云数据的格式正确（numpy数组，shape为(n_points, 3)）
2. 根据具体应用场景选择合适的算法和参数
3. 对于大规模点云数据，建议使用批处理方式
4. 注意内存使用，适当调整参数以避免内存溢出

## 未来改进
1. 添加GPU加速支持
2. 实现更多特征提取方法
3. 优化内存使用
4. 添加更多评估指标
5. 提供可视化工具 