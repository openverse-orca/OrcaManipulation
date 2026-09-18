# G1 OmniPicker VR Teleoperation Guide

本说明用于指导在 **OrcaLab / OrcaStudio + PICO VR** 环境下运行 `G1 OmniPicker` 机器人遥操作与数据采集程序。

## 1. 前置条件

运行前请确认以下内容：

- OrcaLab 已正常启动。
- 已在 OrcaLab / OrcaStudio 中加载机器人场景 JSON 文件。
- 场景 JSON 文件位于：

```text
src/examples/dataCollection/test-env
```

- PICO 头显已连接并正常工作。
- PICO 中已打开 `OrcaGymCtrl`，并保持应用处于运行状态。
- Python / Conda 环境 `orcalab` 已正确安装。

## 2. 启动程序

### 2.1 激活 Conda 环境

打开终端并执行：

```bash
conda activate orcalab
```

### 2.2 进入 OrcaManipulation 工程目录


### 2.3 启动 G1 OmniPicker 遥操作程序

执行：

```bash
python src/examples/dataCollection/g1_omnipicker_collection_tele.py
```

程序启动后，请保持终端运行，不要关闭当前窗口。

## 3. 初次启动操作

程序初次运行后，需要先完成一次控制输入初始化：

1. 按下左摇杆。
2. 按下右摇杆。
3. 再次按下右摇杆。
4. 完成后即可开始操作机器人。

> 如果机器人没有响应，请优先确认 PICO 中的 `OrcaGymCtrl` 是否处于前台运行状态，以及 OrcaLab 是否已正确加载场景。

## 4. VR 手柄控制说明

| VR 操作 | 机器人功能 |
| --- | --- |
| 左手柄移动 / 旋转 | 控制左机械臂 |
| 右手柄移动 / 旋转 | 控制右机械臂 |
| 左摇杆左右 | 车辆转向 |
| 右摇杆上下 | 车辆前进 / 后退 |
| 左 Grip 按住 | 机器人上半身持续上升 |
| 松开左 Grip | 停止上升并保持当前位置 |
| 右 Grip 按住 | 机器人上半身持续下降 |
| 松开右 Grip | 停止下降并保持当前位置 |
| 按下右摇杆 | 腰部自动复位到最低位置 |
| 按下左摇杆 | 开始 / 结束数据采集 |
| 左 Trigger | 左夹爪控制 |
| 右 Trigger | 右夹爪控制 |

## 5. 推荐操作顺序

建议按照以下顺序进行操作：

1. 启动 OrcaLab / OrcaStudio。
2. 加载 `src/examples/dataCollection/test-env` 中对应的 JSON 场景文件。
3. 打开 PICO 头显。
4. 启动 PICO 中的 `OrcaGymCtrl`。
5. 在终端执行指令
6. 完成初次摇杆输入初始化。
7. 先测试底盘前进、后退和转向。
8. 再测试左右机械臂和夹爪。
9. 测试腰部上升、下降和复位。
10. 确认所有控制正常后，再开始正式数据采集。

## 6. 腰部升降控制

### 上升

按住左 Grip：

```text
机器人上半身持续上升
```

松开后立即停止，并保持当前高度。

### 下降

按住右 Grip：

```text
机器人上半身持续下降
```

松开后立即停止，并保持当前高度。

### 复位

按下右摇杆：

```text
当前位置
   ↓
自动下降
   ↓
最低位置
```

复位过程中请避免同时进行其他腰部升降操作。


