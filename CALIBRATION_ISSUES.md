# 标定问题分析与正确方法

## 🔍 当前标定问题

### 1. 图像数量严重不足 ⚠️⚠️⚠️

```
内参标定: 每个相机仅8张图
外参标定: 每个相机仅8张图
```

**问题**：图像太少，无法充分约束相机参数
**推荐**：15-20张以上

### 2. 标定方法错误 ⚠️⚠️⚠️

**你的方法**：把棋盘格放在地面上拍几张

**问题**：
- ❌ 视角变化太小（都是俯视角度）
- ❌ 深度变化不足（棋盘格始终在地面）
- ❌ 角度变化不足（棋盘格平放）
- ❌ 覆盖范围小（只覆盖画面下方）

**结果**：导致fx和fy估计严重不准确（fy/fx比值0.16-1.41，正常应该接近1.0）

## ✅ 正确的标定方法

### 内参标定（每个相机单独）

**目标**：估计相机内参K和畸变系数dist

**步骤**：

1. **手持棋盘格**，在相机前方移动
2. **覆盖整个视野**（上、下、左、右、中心）
3. **变化深度**（近距离0.5m - 远距离3m）
4. **变化角度**（正对、倾斜、旋转）
5. **拍摄15-20张**不同位置的图像

**关键点**：
```
✓ 手持移动棋盘格（不要固定）
✓ 每张图都要有明显的位置/角度/深度变化
✓ 棋盘格要占据画面的不同区域
✓ 包含近景（棋盘格占80%画面）
✓ 包含远景（棋盘格占20%画面）
✓ 包含倾斜角度（不只是正对相机）
```

**示意图**：
```
移动路径（俯视图）:

        相机📷
         ↑
    ┌────┼────┐
    │ ◻️  ◻️  ◻️ │  上方（远）
    │ ◻️  ◻️  ◻️ │  中间
    │ ◻️  ◻️  ◻️ │  下方（近）
    └──────────┘
    
侧视图（深度变化）:
    
    📷 ← 近 ◻️ ← 中 ◻️ ← 远 ◻️
```

### 外参标定（多相机同时）

**目标**：估计相机之间的相对位置关系（R, T）

**步骤**：

1. **将棋盘格放在场景中央**（所有相机都能看到）
2. **保持棋盘格静止不动**
3. **所有相机同时拍照**
4. **变换棋盘格位置**，重复步骤1-3
5. **拍摄8-10组**不同位置（每组4张图，一个相机一张）

**关键点**：
```
✓ 棋盘格要被所有相机同时看到
✓ 可以平放地面（外参标定可以）
✓ 也可以手持或支架固定（更好）
✓ 不同位置和角度
✓ 确保棋盘格在所有相机视野内
```

**特别注意**：
- 外参标定时棋盘格定义了世界坐标系
- 如果棋盘格平放地面，世界坐标系的Z轴（或Y轴）会垂直地面
- 要确保坐标系定义合理（通常Y轴向上）

## 📋 正确的标定命令

### 步骤1：准备棋盘格图像

**内参标定**：
```bash
# 目录结构
data/calib/intri/images/
├── 01/  # 相机1，15-20张，手持移动棋盘格
├── 02/  # 相机2，15-20张
├── 03/  # 相机3，15-20张
└── 04/  # 相机4，15-20张
```

**外参标定**：
```bash
# 目录结构（每组是同时拍的）
data/calib/extri/images/
├── 01/  # 8-10张，对应下面8-10个位置
├── 02/  # 8-10张
├── 03/  # 8-10张
└── 04/  # 8-10张

# 每组4张图（01、02、03、04）要是同一时刻拍的
```

### 步骤2：检测棋盘格

**你的棋盘格参数**：
- `--pattern 5,3`：5列3行的内角点
- `--grid 0.066`：格子边长6.6cm

```bash
# 内参 - 检测棋盘格
python apps/calibration/detect_chessboard.py data/calib/intri/ \
    --out data/calib/intri/chessboard \
    --pattern 5,3 \
    --grid 0.066

# 外参 - 检测棋盘格
python apps/calibration/detect_chessboard.py data/calib/extri/ \
    --out data/calib/extri/chessboard \
    --pattern 5,3 \
    --grid 0.066
```

### 步骤3：标定

```bash
# 内参标定
python apps/calibration/calib_intri.py data/calib/intri

# 外参标定
python apps/calibration/calib_extri.py data/calib/extri \
    --intri data/calib/intri/output/intri.yml
```

## 🎯 你的命令问题检查

### 你的命令
```bash
python apps/calibration/detect_chessboard.py data/calib/intri/ \
    --out data/calib/intri/chessboard --pattern 5,3 --grid 0.066

python3 apps/calibration/calib_intri.py ${DATA_DIR}/intri

python apps/calibration/detect_chessboard.py data/calib/extri/ \
    --out data/calib/extri/chessboard --pattern 5,3 --grid 0.066

python3 apps/calibration/calib_extri.py ${DATA_DIR}/extri \
    --intri ${DATA_DIR}/intri/output/intri.yml
```

**命令本身是对的！** ✅

**问题在于**：
1. ❌ 输入图像质量差（只有8张，变化不够）
2. ❌ 内参标定方法错误（棋盘格放地面）

## 💡 快速修复方案

### 方案1：重新采集内参图像（推荐）⭐⭐⭐⭐⭐

**只需重新采集内参**，外参可以保留：

1. **准备**：
   - 棋盘格一张（你已有的5x3格）
   - 相机固定不动

2. **每个相机拍摄15-20张**：
   ```
   手持棋盘格，移动到：
   画面上方、下方、左侧、右侧、中心（5个）
   × 
   近距离、中距离、远距离（3个）
   × 
   正对、左倾、右倾（至少）
   = 至少15张
   ```

3. **保存图像**：
   ```bash
   data/calib/intri_new/images/01/*.jpg
   data/calib/intri_new/images/02/*.jpg
   data/calib/intri_new/images/03/*.jpg
   data/calib/intri_new/images/04/*.jpg
   ```

4. **运行标定**：
   ```bash
   python apps/calibration/detect_chessboard.py data/calib/intri_new/ \
       --out data/calib/intri_new/chessboard \
       --pattern 5,3 --grid 0.066
   
   python apps/calibration/calib_intri.py data/calib/intri_new
   
   # 使用新内参重新标定外参
   python apps/calibration/calib_extri.py data/calib/extri \
       --intri data/calib/intri_new/output/intri.yml
   ```

### 方案2：完全重新标定

如果外参也有问题（比如世界坐标系定义），全部重做：

1. **内参**：按方案1
2. **外参**：
   - 8-10个不同位置
   - 每个位置所有相机同时拍照
   - 棋盘格要被所有相机看到

## 🔍 如何判断标定质量

### 重投影误差

标定完成后，检查输出：
```
Reprojection error: 0.3 pixels  ← 好
Reprojection error: 0.5 pixels  ← 可接受  
Reprojection error: 1.0 pixels  ← 勉强
Reprojection error: 2.0+ pixels ← 差，需要重做
```

### fx/fy比值

```python
# 检查内参
import yaml
import numpy as np

with open('data/calib/intri/output/intri.yml', 'r') as f:
    # 读取并检查
    pass

# 理想情况
fx ≈ fy (比值0.95-1.05)
```

### 视觉检查

运行：
```bash
python apps/calibration/check_calib.py data/calib/extri \
    --mode cube --show
```

看投影的3D立方体是否对齐。

## 📚 参考资料

### 标定原理

- 内参标定：需要大量不同角度和深度的图像
- 外参标定：需要同时看到同一个标定板

### 常见错误

1. ❌ 图像太少
2. ❌ 视角变化不够（如只平放棋盘格）
3. ❌ 深度变化不够
4. ❌ 未覆盖画面边缘
5. ❌ 棋盘格检测失败（模糊、光照不均）

### EasyMocap标定教程

参考官方文档：
```
https://github.com/zju3dv/EasyMocap/blob/master/doc/calibration.md
```

## 🎯 总结

**你的问题根源**：
1. 内参标定图像太少（8张）
2. 内参标定方法错误（平放地面，视角单一）
3. 导致fx/fy严重异常
4. 进而导致SMPL拟合失败

**解决方法**：
- 重新采集内参图像（15-20张，手持移动棋盘格）
- 使用正确的移动方式（覆盖各个位置、角度、深度）
- 重新标定内参和外参

**预期结果**：
- fx/fy比值接近1.0
- 重投影误差 < 0.5像素
- SMPL能正确拟合

---

**文档版本**：1.0  
**创建日期**：2025-11-27  
**适用场景**：4相机多视角人体重建

