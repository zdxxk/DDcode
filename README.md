# 固定翼无人机对地侦察与打击系统 — 代码说明文档

## 项目总览

该项目是一个 **CUADC（中国大学生无人机竞赛）/ CADC** 参赛项目的完整软件系统，分为 **三大功能板块**：

1. **C++ 端（地面站计算机）**：视觉目标检测与识别
2. **Python 端**：飞行航线规划、Pixhawk飞控通信、打击航线生成与上传
3. **Python 主控脚本**：整合侦察与打击全流程

### 整体工作流程

> 无人机起飞 → 侦察航线飞行（Python上传侦察航点） → **视觉检测靶标与数字识别（C++）** → 计算靶标GPS坐标（C++） → 通过共享内存传给Python → **自动规划打击航线并上传飞控（Python）**

---

## 一、C++ 端：视觉检测模块（机载实时处理）

### 1. `main.cpp` — 主程序入口与核心循环

**核心功能**：实时视频流处理、两阶段目标检测、GPS坐标计算与共享内存通信。

**关键流程**：

1. **加载两个 YOLO ONNX 模型**（`best_target.onnx` 用于检测靶标，`best_number.onnx` 用于识别靶标上的数字编号）
2. **打开 RTSP 视频流** (`rtsp://192.168.144.25:8554/main.264`)，以独立线程持续拉流
3. **主循环**逐帧处理：
   - 可选畸变矫正（`UNDISTORT` 宏）
   - 第一帧时创建带时间戳的 AVI 录像文件
   - **第一阶段**：用 `net1` 检测靶标位置（`Detect` with `model_flag=0`）
   - **靶标后处理**（`muti_target`）：裁剪靶标区域、调用 `shift.CorrectDirection` 旋转转正
   - **第二阶段**：对转正后的每个靶标用 `net2` 识别数字编号（`model_flag=1`）
   - 从 Python 端共享内存读取飞行数据（高度、航向、GPS坐标、GPS锁定状态）
   - 调用 `target_gps()` 计算靶标的实际经纬度
4. **统计频率**：当检测到的数字 ID 出现次数 ≥ `NUMBER_COUNT`（3次），且前三名都足够，判定侦察完成
5. **计算靶标中位数GPS**，写入共享内存 `gps_shared_memory` 回传给 Python
6. **自动切换**到纯画面推流模式（第二段 while 循环）
7. 按**空格键**启动300秒计时，按 **q** 退出

**重要宏定义**：

| 宏 | 值 | 说明 |
|---|---|---|
| `NUMBER_COUNT` | 3 | 检测到3次即判定侦察完成 |
| `SWITCH_TIME` | 300 | 空格后300秒上限 |
| `USE_CUDA` | true | 启用CUDA加速 |
| `UNDISTORT` | 0 | 默认不畸变矫正 |

**线程安全帧队列 `FrameQueue`**：使用 `std::mutex` + `std::condition_variable` 实现生产者-消费者模式，最大缓存10帧，防止内存暴涨。

---

### 2. `global.h` — 全局数据结构定义

定义了核心数据传输结构体 `Data`：

```cpp
struct Data {
    int id;                 // 检测到的靶标数字ID
    double altitude;        // 无人机高度
    double heading;         // 无人机航向角
    double latitude;        // 无人机GPS纬度
    double longitude;       // 无人机GPS经度
    double target_latitude; // 计算出的靶标纬度
    double target_longitude;// 计算出的靶标经度
    int fluency;            // 流畅度标记
};
```

---

### 3. `yolo.h` / `yolo.cpp` — YOLO 目标检测模块

#### `yolo.h` — 数据结构与类声明

**检测结果结构体**：

```cpp
struct Output {
    int id;             // 结果类别id
    float confidence;   // 结果置信度
    cv::Rect box;       // 矩形框
    cv::Mat img;        // 目标图像
    cv::Point center;   // 中心点坐标
};

struct Numbers {
    std::vector<Output> targets;  // 数字检测结果列表
    int id;                       // 拼接后的数字ID
    float confidence;             // 平均置信度
    cv::Mat img;                  // 标注后的图像
};
```

**`Yolo` 类关键成员**：

| 成员 | 说明 |
|---|---|
| `netWidth/netHeight` | 模型输入尺寸（640×640 P5 / 1280×1280 P6） |
| `netStride[]` | 特征图步长 {8, 16, 32, 64} |
| `netAnchors[][]` | 锚框参数 |
| `boxThreshold` | 框置信度阈值 0.25 |
| `classThreshold` | 类别置信度阈值 0.25 |
| `nmsThreshold` | NMS阈值 0.45 |
| `className` | 类别名：靶标["target"]，数字["0"~"9"] |

#### `yolo.cpp` — 核心实现

| 方法 | 功能 |
|---|---|
| `readModel()` | 加载 ONNX 模型，支持 CUDA/CPU 后端切换 |
| `Detect()` | 执行推理，支持 YOLOv5/v7 两种架构，包含 NMS 非极大值抑制。`model_flag=0` 靶标检测（置信度阈值0.7），`=1` 数字检测（置信度阈值0.8） |
| `drawPred()` | 在图像上绘制检测框、标签、置信度 |
| `drawPred_number()` | 识别两位数编号（如"12"），将左右两个数字按x坐标排序后拼接，计算平均置信度 |
| `target()` | 裁剪单个靶标区域（取正方形外接框，以长边为准扩展） |
| `muti_target()` | 多靶标后处理：裁剪每个靶标 → 调用 `Shift::CorrectDirection` 转正 → 保存转正后的靶标图像 |
| `show_target()` | 对置信度>0.5的数字创建独立窗口显示并为该ID的计数+1，最多展示10个窗口 |

---

### 4. `shift.h` / `shift.cpp` — 靶标转正（方向校正）模块

这是**最关键的后处理模块**，用于将任意朝向拍摄的靶标旋转到正立方向。

#### `Shift` 类核心方法

| 方法 | 功能 |
|---|---|
| `CorrectDirection()` | **主入口**：输入靶标图像 → 输出转正后的图像。先缩放到400×400，双边上滤波去噪 |
| `ColorProcess()` | HSV颜色空间提取红色（H:0~5, 160~180）和蓝色（H:100~124），合并为"天井颜色区域" |
| `sift_area()` | 保留最大连通域，去除噪点 |
| `findSingleRectangleCenter()` | 在二值图中查找唯一矩形的中心点 |
| `calculateAngle()` / `angle()` | 计算几何角度 |

#### 转正算法详细流程

1. **颜色分割**：提取红+蓝区域（靶标底座的天井颜色），形态学开运算去噪
2. **凸包填充**：对颜色区域进行凸包检测并填充
3. **白色数字区域提取**：G通道阈值提取白色数字区域
4. **纯颜色区域 = 凸包填充(红+蓝) - 白色区域**
5. 尝试用 `approxPolyDP` 拟合五边形：
   - **找到五边形**：确定最小角度的顶点 → 从图像中心指向该顶点的方向作为"上方" → 计算旋转角度
   - **找不到五边形**：计算红色重心 + 白色数字矩形中心 → 白→红连线方向即为"上方" → 计算旋转角度
6. 用 `warpAffine` 旋转图像至正立

---

### 5. `location.h` / `location.cpp` — GPS 坐标计算模块

#### 核心函数 `target_gps()`

将像素坐标转换为靶标GPS经纬度。

**计算原理**：

```
1. 像素坐标 → 相机坐标系
   - 像元尺寸: 3.45μm, 焦距: 6mm
   - 图像中心: 960×540
   - camerax = (pixel_x * 3.45e-6 - 960 * 3.45e-6) / 0.006 * height
   - cameray = (540 * 3.45e-6 - pixel_y * 3.45e-6) / 0.006 * height

2. 相机坐标系 → 地面坐标系（考虑航向角旋转）
   - delta_weidu = (camerax*sin(yaw) - cameray*cos(yaw)) * rateweidu
   - delta_jingdu = -(cameray*sin(yaw) + camerax*cos(yaw)) * ratejingdu

3. 最终靶标经纬度 = 无人机GPS + 偏移量
```

**关键常量**：

| 常量 | 值 | 说明 |
|---|---|---|
| `ratejingdu` | 0.0000104062 | 经度1米对应的度数 |
| `rateweidu` | 0.00000899798 | 纬度1米对应的度数 |
| `camera_F` | 0.006m | 相机焦距 |
| `camera_size` | 3.45e-6m | 像元尺寸 |
| `xmax/ymax` | 1920/1080 | 图像分辨率 |

---

### 6. `MY_DFT.h` / `MY_DFT.cpp` — 傅里叶变换图像增强模块

提供两个函数：

- **`My_DFT()`**：完整DFT流程——扩展边界（最优DFT尺寸）→ 创建复数矩阵 → DFT → 计算幅值 → 对数归一化 → 象限重排（中心化），同时输出频谱图和复数域结果

- **`Homo()`**：同态滤波——对B/G通道分别进行 ln → DFT → 高斯高通滤波 → IDFT → exp → 归一化，保留R通道原值。用于增强图像对比度、抑制光照不均

---

### 7. `CMakeLists.txt` — 构建配置

```cmake
cmake_minimum_required(VERSION 3.1)
project(main)
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED)
set(CMAKE_CXX_STANDARD 11)
add_executable(main main.cpp location.cpp MY_DFT.cpp yolo.cpp shift.cpp)
target_link_libraries(main PRIVATE ${OpenCV_LIBS} ${Boost_LIBRARIES})
```

- 依赖：**OpenCV**（图像处理 + DNN推理）+ **Boost**（共享内存 interprocess）
- C++11 标准
- 编译生成 `main` 可执行文件

---

## 二、Python 端：航线规划与飞控通信模块

### 8. `main.py` — Python 端主控程序

**整个系统的 Python 侧入口**，串联侦察→打击全流程：

```python
连接飞控(/dev/ttyUSB0, 57600波特率)
  → UploadScout()：上传预存侦察航线到飞控
  → 解锁无人机
  → 循环写入共享内存(高度/航向/GPS/锁定状态)
  → 等待C++端发回靶标GPS
  → Line(targetlat, targetlon, vehicle)：自动规划打击航线并上传
```

**通信方式**：

| 共享内存名称 | 方向 | 内容 |
|---|---|---|
| `shared_memory` | Python → C++ | 飞控状态（高度/航向/经纬度/GPS锁定状态），5个double |
| `gps_shared_memory` | C++ → Python | 靶标经纬度，2个double |

使用 `multiprocessing.shared_memory` 实现进程间通信。

---

### 9. `Line.py` — 航线规划核心（最新完整版，~1230行）

这是整个项目的**第二个核心模块**，负责根据靶标位置自动规划完整的打击+降落航线。

#### 核心数据结构

- **`Location`**：相对坐标点（x, y, 航向hdg, 高度alt），在本地坐标系中工作
- **`WayPoint`**：Mission Planner 格式的航点，支持属性访问和下标访问

| command值 | 含义 |
|---|---|
| 16 | WAYPOINT（普通航点） |
| 183 | DO_SET_SERVO（投弹舵机释放） |
| 21 | LAND（降落） |
| 178 | DO_CHANGE_SPEED（调速） |

#### 主函数 `Line(targetlat, targetlon, vehicle)` 流程

```
1. 读取当前位置 → 转换为当地相对坐标（原点为预设的LocalLatitude/LocalLongitude）
2. check()：GPS/位置合法性检查
3. checkZoneTarget()：检查目标是否在预设目标区内，否则修正到边界内
4. classifyLocation()：判断是A型（可直接打击）还是B型（需先导航到远处）
5. 若B型：
   → PassPattern()：选择最佳导航点并转向
   → StraightPattern()：若距离不够，再直飞一段
6. TargetPattern()：转向目标 + AttackCourse() 打击航线（含投弹指令）
7. LandingPattern()：规划降落航线（支持预转向回角机制）
8. checkZoneLine()：航线超界检查
9. WPwrite()：写入waypoints文件
10. upload_mission_waypoints()：通过dronekit上传飞控
```

#### 关键子函数详解

| 函数 | 功能 |
|---|---|
| `updateLocation()` | 从飞控GPS读取并转换为当地（LocalLatitude/LocalLongitude为原点）的米制相对坐标 |
| `loadTarget()` | 将靶标经纬度转换为当地米制坐标 |
| `ConvertToGlobal()` | 将当地坐标转回经纬度 |
| `calPoint()` | 计算从某点沿某方向前进dist米后的新点（深拷贝防bug） |
| `calSetAlt()` | 根据高度变化率约束（AltSlope）计算实际可达高度 |
| `refPoint()` | 计算左右转向中心（距当前位置TurnRadius=30m的左右两点） |
| `calBearing()` | 计算从location指向target的方位角（0~360°） |
| `calHeadingDiff()` | 计算两航向角之差（-180~180°） |
| `calDist()` | 两点间欧氏距离 |
| `inZone()` | **射线法**判断点是否在多边形区域内 |
| `classifyLocation()` | 判断是A型（转弯后距离目标 > sqrt(TurnRadius²+StableLength²)）还是B型 |
| `selectPasspoint()` | 从预设PASSPOINT中选择最优导航点（基于转向距离和角度评分） |
| `TurnCourseCore()` | **转向核心算法**：计算左/右转向中心、切点角度、生成弧形航点 |
| `TurnCourse()` | 转向至朝向目标 |
| `TurnHeadingCourse()` | 转向至指定航向（通过在远处生成虚拟目标实现） |
| `StraightCourse()` | 直飞线段，间距≤PointDist(30m)，含高度渐变 |
| `AttackCourse()` | 打击航线：从当前位置飞向投弹点（距目标LeadLength=30m），投弹点写入command=183（舵机释放） |
| `TargetPattern()` | 转向+打击的组合模式，若航线超界则尝试反向方案 |
| `StraightPattern()` | 用余弦定理计算需要直飞的距离，使后续转向后有足够StableLength |
| `PassPattern()` | 转向导航点模式 |
| `LandingPattern()` | **复杂的降落航线生成器**：枚举多种转向方案（含TurnReversal预转向），按分值选择最优 |
| `LandingCourse()` | 生成降落梯度航点组（LANDINGGROUP比例缩放） |
| `checkZoneLine()` | 检查所有航点是否在安全飞行区域内 |
| `showLine()` | 用matplotlib可视化绘制航线 |
| `WPwrite()` | 输出QGC WPL 110格式航点文件 |

#### 关键参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `TurnRadius` | 30m | 固定翼转向半径 |
| `StableLength` | 80m | 投弹前稳定段总长度 |
| `LeadLength` | 30m | 投弹点到目标距离 |
| `AttackAlt` | 12m | 投弹高度 |
| `CruiseAlt` | 30m | 巡航高度 |
| `PointDist` | 30m | 航点最大间距 |
| `LandLength` | 150m | 降落稳定段长度 |
| `AltSlope` | (-0.2, 0.5) | 直飞时高度变化斜率限制 |
| `AltSlopeTurn` | (-0.1, 0.2) | 转向时高度变化斜率限制 |
| `TURNRATIO` | 1.8 | 评分权重比，越大越倾向于直线多的方案 |
| `MAX_UPLOAD_TIME` | 15s | 上传超时警告阈值 |
| `LANDPOINT` | (0,-20,3°,0.5m) | 平飘点参数 |
| `TurnReversalAngle_step` | 5° | 降落回角步长 |
| `TurnReversalAngle_max` | 150° | 降落回角最大角度 |

#### 声音提示系统

支持蜂鸣（Windows: `winsound.Beep` / Linux: `sox` + `aplay`）和人声（播放`.wav`文件）两种模式，默认使用人声。声音类型包括：

| 类型 | 含义 |
|---|---|
| `Emergency` | 飞手立刻接管，出大事了 |
| `WarningTarget` | 目标超界 |
| `WarningLine` | 航线超界 |
| `Uploading` | 开始上传航线到飞控 |
| `Uploaded` | 航线上传完成 |
| `Abnormal` | 较小的正常不利情况 |

---

### 10. `Line_1111.py` — 航线规划（中期版本，~1060行）

大体逻辑与 `Line.py` 一致，主要区别：
- 降落航线枚举方案更少（4种，无TurnReversal预转向机制）
- 没有 `WPdata_temp` 机制（打击航线无法自动在超界时尝试反向方案）
- 默认场地参数为定州（`LocalLatitude=38.5582, LocalLongtitude=115.1406`）
- 上传超时时间为30s（vs 最新版15s）
- 追加了 `Line_fixed()` 函数用于CADC固定目标模式

---

### 11. `Line(9.5).py` — 航线规划（早期版本，~870行）

最原始的版本，主要特点：
- 参数列表较不完整
- `LandingPattern` 中 score 计算简化（无TurnRadius加权）
- 没有 `checkZoneLine_temp` 机制
- `addPoint` 函数缺少 `SetSpeed` 指令
- 默认场地为"桥头堡"测试场（`LocalLatitude=30.307560, LocalLongtitude=120.427237`）
- Waypoint写入格式为空格分隔而非Tab分隔

---

### 12. `beep.py` — 音频提示模块（独立版，~45行）

独立于 `Line.py` 的简化蜂鸣/人声提示模块。使用 `sox` 生成蜂鸣音（Linux/Mac）或用 `winsound.PlaySound` 播放人声 `.wav` 文件（Windows）。

支持的提示类型：`Emergency`, `Warning`, `Pass`, `Uploading`, `Uploaded`, `Abnormal`, `Bug`

---

## 三、数据文件

### 13. `models/` — ONNX 深度学习模型

| 文件 | 用途 |
|---|---|
| `best_target.onnx` / `best_target(2023).onnx` | YOLO 靶标检测模型（检测地面靶标位置） |
| `best_number.onnx` / `best_number(2023).onnx` | 数字识别模型（0-9数字分类，识别靶标编号） |

---

### 14. `SoundPack/` — 人声提示音效

| 分类 | 文件 |
|---|---|
| 数字播报 | `0.wav` ~ `9.wav`（中文数字）、`10.wav`、`100.wav` |
| 清晰数字 | `0_clear.wav`, `1_clear.wav`, `2_clear.wav`, `7_clear.wav`, `9_clear.wav` |
| 计数单位 | `x000.wav`（"千"）、`x0000.wav`（"万"）、`ge.wav`（"个"）、`dian.wav`（"点"） |
| 状态提示 | `emergency.wav`, `warningtarget.wav`, `warningline.wav`, `uploaded.wav`, `uploading.wav`, `abnormal.wav` |

---

### 15. `.waypoints` 文件 — QGC WPL 110 格式航点

| 文件 | 说明 |
|---|---|
| `8.waypoints` | 当前使用的打击航线输出文件（程序自动生成覆盖） |
| `test.waypoints` | 定州测试场的侦察航线（25个航点 + DO_JUMP循环回到第12个航点） |
| `Q/1.waypoints` ~ `Q/30.waypoints` | 多个备用航点文件 |
| `D__/wp1test.waypoints`, `D__/wp2test.waypoints` | 测试时输出的打击航线文件 |

**文件格式**（每行由Tab分隔）：
```
index  currentwp  frame  command  param1  param2  param3  param4  lat  lon  alt  autocontinue
```

---

### 16. `videos/` — 录像存储目录

C++ 端实时录制带时间戳的 `.avi` 视频文件（MJPG编码，10fps，1920×1080），文件名格式：`YYYY-MM-DD_HH:MM:SS.avi`

---

### 17. `build/` — CMake 编译构建目录

包含编译生成的 `main` 可执行文件及所有 CMake 中间件（`.o` 目标文件、`CMakeCache.txt` 等）。

---

## 四、系统架构总图

```
┌──────────────────────────────────────────────────────────────────┐
│                     无人机 (Pixhawk飞控)                          │
│  GPS / 姿态 / 高度 / 航向 / 舵机                                  │
└────────────────┬─────────────────────────────────────────────────┘
                 │ MAVLink (USB/串口 57600bps)
                 ↓
┌──────────────────────────────────────────────────────────────────┐
│                         地面站计算机                               │
│                                                                  │
│  ┌─────────────────────────┐    共享内存     ┌─────────────────┐  │
│  │    Python 端 (main.py)  │ ←───────────→  │  C++ 端 (main)  │  │
│  │                         │   飞控数据(5×d) │                 │  │
│  │  • dronekit 飞控通信    │ ─────────────→ │  • RTSP视频流   │  │
│  │  • 侦察航线上传          │   靶标GPS(2×d) │  • YOLO靶标检测 │  │
│  │  • 打击航线自动规划      │ ←───────────── │  • 数字编号识别 │  │
│  │  • 降落航线自动规划      │                 │  • 靶标转正     │  │
│  │  • Line.py 航线引擎     │                 │  • GPS坐标解算  │  │
│  │  • 音频提示系统          │                 │  • 视频录制     │  │
│  └───────────┬─────────────┘                 └─────────────────┘  │
│              │ MAVLink 上传航点                                    │
│              ↓                                                    │
│  飞控接收 → 切AUTO模式 → 执行打击航线                              │
│  (巡航30m → 降至12m → 投弹DO_SET_SERVO → 爬升 → 降落LAND)        │
└──────────────────────────────────────────────────────────────────┘
```

### 任务执行时间线

```
起飞 ──→ 侦察航线 (UploadScout) ──→ 靶标检测 (C++ YOLO) ──→ 数字识别 ──→ GPS解算
                                                                              │
                                                                              ↓
降落 ←── 飞控执行AUTO ←── 航线上传 ←── WPwrite ←── 航线规划 ←── 靶标GPS回传
 LAND                  MAVLink       .waypoints   Line.py    gps_shared_memory
```

---

## 五、关键设计特点

1. **C++/Python 混合架构**：C++ 负责计算密集型的视觉推理，Python 负责灵活的航线规划逻辑，通过共享内存高效通信
2. **两阶段级联检测**：先检测大尺度的靶标，裁剪转正后再识别小尺度的数字编号，提高数字识别准确率
3. **靶标朝向自适应**：通过颜色分割+几何分析自动判断靶标朝向并旋转转正，不依赖固定安装方向
4. **航线安全校验**：生成的航线必须通过目标区域检查（`checkZoneTarget`）和飞行区域检查（`checkZoneLine`），超界航线不会被上传
5. **多方案择优**：降落航线枚举多种转向方案（含预转向回角），按加权评分选择最优方案
6. **防呆机制**：GPS检查（距离>100km报警）、上传超时报警（飞手接管提示）、人声/蜂鸣双重提示
7. **多场地适配**：通过修改 `LocalLatitude/LocalLongitude/TargetZone/LineZone` 等参数即可适配不同比赛场地（桥头堡/杭电/定州）
