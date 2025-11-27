# SMPL重建问题诊断报告

## 问题描述

运行多视角SMPL重建时，SMPL模型"躺在地上"，与真实人体姿态完全不符。

## 诊断结果

### 核心问题：相机内参标定严重异常 ⚠️

| 相机 | fx | fy | fy/fx比值 | 状态 |
|-----|----|----|----------|------|
| 01 | 497.1 | 284.3 | **0.572** | 严重异常 |
| 02 | 731.8 | 1028.5 | **1.405** | 严重异常 |
| 03 | 1335.7 | 211.3 | **0.158** | 极度异常 |
| 04 | 928.8 | 180.1 | **0.194** | 极度异常 |

**正常值**：fy/fx应该接近1.0 (0.95-1.05之间)

### 三角化结果分析

虽然内参异常，但三角化结果的坐标系方向是**正确的**：

```
世界坐标系范围:
  X轴: [-0.331, 0.318] m  (跨度: 0.65m) - 左右
  Y轴: [-1.127, 0.435] m  (跨度: 1.56m) - 上下 ✓
  Z轴: [-0.002, 0.410] m  (跨度: 0.41m) - 前后
```

- Y轴跨度1.56m，符合人体身高
- 坐标系定义正确（Y轴竖直向上）

关键部位3D位置：
```
  Nose    : X=-0.008, Y= 0.404, Z= 0.155  (头部，最高)
  Neck    : X=-0.005, Y= 0.243, Z=-0.002
  MidHip  : X=-0.002, Y=-0.248, Z= 0.014  (髋部，中间)
  Ankles  : Y≈-1.08              (脚踝，最低)
```

### SMPL拟合失败

```
Rh (全局旋转): [0.000, 0.000, 0.000]  - 无旋转
Th (全局平移): [0.000, 0.000, 0.000]  - 无平移
```

SMPL参数全为0，说明优化过程失败或未正常运行。

## 问题原因

### 1. 标定分辨率不匹配

**最可能的原因**：标定时使用的图像分辨率与实际运行时不同。

- 实际图像：1280 x 720
- 标定图像：可能是其他分辨率（如640x480, 1920x1080等）

fx和fy的异常比值表明：
- 某些相机标定时图像被拉伸或压缩
- 横纵比不匹配

### 2. 标定质量问题

内参畸变系数也显示异常：
```
dist_02: [0.076615, -7.124058, ...]  - 异常大的畸变系数
```

## 影响

1. **三角化精度下降**：虽然能得到大致正确的3D位置，但精度很低
2. **SMPL拟合失败**：由于3D关键点不够准确，SMPL无法正确拟合
3. **可视化错误**：SMPL模型姿态错误

## 解决方案

### 方案1：重新标定（强烈推荐） ⭐⭐⭐⭐⭐

**步骤**：

1. **确保使用正确的图像分辨率**
   ```bash
   # 确认实际图像分辨率
   file data/examples/my_multiview/images/01/000000.jpg
   # 或
   identify data/examples/my_multiview/images/01/000000.jpg
   ```

2. **录制标定视频/图像**（使用1280x720分辨率）
   - 内参标定：每个相机单独录制，移动棋盘格
   - 外参标定：所有相机同时拍摄静止棋盘格

3. **运行标定**
   ```bash
   # 提取图像
   python scripts/preprocess/extract_video.py ${data} --no2d
   
   # 检测棋盘格
   python apps/calibration/detect_chessboard.py ${data} --out ${data}/output/calibration --pattern 9,6 --grid 0.025
   
   # 标定内参
   python apps/calibration/calib_intri.py ${data} --step 1
   
   # 标定外参
   python apps/calibration/calib_extri.py ${data} --intri ${data}/output/calibration/intri.yml --step 1
   ```

### 方案2：缩放内参（临时方案） ⭐⭐⭐

如果无法重新标定，尝试缩放内参矩阵以匹配当前分辨率。

**需要知道**：
- 标定时使用的分辨率（旧）
- 当前使用的分辨率（新）= 1280x720

**缩放公式**：
```python
scale_x = new_width / old_width
scale_y = new_height / old_height

fx_new = fx_old * scale_x
fy_new = fy_old * scale_y
cx_new = cx_old * scale_x
cy_new = cy_old * scale_y
```

### 方案3：检查标定图像分辨率

查看标定时使用的图像：

```bash
# 检查标定目录
ls data/calib/intri/images/
ls data/calib/extri/images/

# 查看图像分辨率
identify data/calib/intri/images/01/000000.jpg
```

如果发现分辨率不同，使用方案2缩放内参。

## 快速验证方法

### 检查当前图像分辨率

```bash
cd /home/bupt630/Dabai/AmmWave/EasyMocap
identify data/examples/my_multiview/images/01/000000.jpg
identify data/examples/my_multiview/images/02/000000.jpg
identify data/examples/my_multiview/images/03/000000.jpg
identify data/examples/my_multiview/images/04/000000.jpg
```

### 检查标定图像分辨率（如果有）

```bash
# 查找标定数据
find data/calib -name "*.jpg" | head -1 | xargs identify
```

### 对比分辨率

如果发现不同，计算缩放比例并修改内参文件。

## 检查配置

### 图像格式配置 ✓

```yaml
# config/datasets/mvimage.yml
reader:
  images:
    ext: .jpg  # 正确
```

### 可视化缩放配置 ✓

```yaml
# config/mv1p/detect_triangulate_fitSMPL.yml
vis2d:
  args:
    scale: 0.5  # 可视化时缩放，不影响计算
```

这些配置是正确的。

## 其他潜在问题

### 1. HRNet模型分辨率

HRNet默认输入384x288，会自动处理缩放，不需要修改。

### 2. SMPL优化参数

配置文件中的SMPL优化参数看起来正常：
- `fitShape`: 拟合形状
- `init_RT`: 初始化旋转平移
- `refine_poses`: 优化姿态

但由于输入的3D关键点精度低，优化可能失败。

### 3. 三角化模式

```yaml
triangulate:
  args:
    mode: iterative  # 迭代模式，更鲁棒
```

这个配置是合理的。

## 总结

### 主要问题
**相机内参标定与实际图像分辨率不匹配**，导致：
1. fx/fy比值严重异常
2. 三角化精度下降
3. SMPL拟合失败

### 根本解决
**使用1280x720分辨率重新标定所有相机**

### 临时方案
找到标定时使用的分辨率，缩放内参矩阵

## 下一步行动

1. **立即检查**：
   ```bash
   identify data/examples/my_multiview/images/*/000000.jpg
   ```

2. **如果能找到标定数据**：
   ```bash
   find data/calib -name "*.jpg" -o -name "*.png" | head -5 | xargs identify
   ```

3. **对比分辨率**，计算缩放比例

4. **决定采用方案1（重新标定）或方案2（缩放内参）**

---

**报告生成时间**：2025-11-27  
**分析帧数**：1帧（代表性）  
**问题严重程度**：高 - 需要立即修复

