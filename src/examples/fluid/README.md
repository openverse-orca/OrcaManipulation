# Fluid-MuJoCo 耦合仿真示例

SPH 流体与 MuJoCo 刚体耦合，经 OrcaLink 与 OrcaLab / OrcaStudio 通信。以下以 **天工机器人 + 流体场景** 为例。

---

## 一、订阅相关资产

1. 资产库链接  
   https://simassets.orca3d.cn/

2. 资产中心 → 生活资产 → 订阅 **FluidTest_Hotel_Bar_02** 流体场景资产

3. 资产中心 → 机器人资产 → 订阅 **tiangong2 机器人资产**

---

## 二、Pico 的安装

1. 将 `PicoController.apk` 安装包下载到电脑上  
   安装包路径（将 `<仓库根>` 换为你本机克隆路径，例如 `~/OrcaManipulation`）：

       <仓库根>/src/examples/超市场景青龙机器人数采案例/pico安装包

2. 打开 Pico，并使用 USB 线将 Pico 连接到刚刚下载 apk 安装包的电脑。

3. 将 apk 安装包拷贝到 Pico 目录下。

4. 在 VR 视角中查看安装包目录，使用右手手柄点击 **A 键**，确认安装刚刚下载的 apk 包。

5. 仍在 VR 视角中，点击：  
   **设置 → 关于本机 → 软件版本号**，  
   使用确认键 **A 键** 对着软件版本号。

6. 连续点击 **6–7 次**，调出开发者模式。  
   至此 Pico 已配置完成。

---

## 三、在 OrcaLab 中的使用

1. 终端进入 **OrcaLab** 使用的 conda 环境（名称以你本机为准，示例为 `orcalab`）：

       conda activate orcalab

2. 执行命令启动 OrcaLab：

       orcalab

3. 启动过程中会自动下载第一步中订阅的资产，请等待下载与同步完成。

---

## 四、数据采集

### 4.1 安装 adb 工具

在 Ubuntu 终端中执行：

       sudo apt install android-tools-adb android-tools-fastboot

---

### 4.2 配置 Pico 与 PC 的通信

1. 使用 USB 数据线连接 Pico 设备与 PC。

2. 在终端中执行 adb 命令：

       adb reverse tcp:8001 tcp:8001

---

### 4.3 启动 VR 控制程序

1. 确认在 VR 中打开 **OrcaGymCtrl** 控制程序。

2. 将 VR 开机，选择底部的 **资源库**。  
   本项目使用的是 **Ultra4** 版本，在资源库中可直接看到 **OrcaGymCtrl**。

3. VR 启动后会提示设置安全边界，可选择重设或保持原有边界（VR 默认配置）。

4. 启动 OrcaGymCtrl 后，会进入一个 3D 界面：
   - 左侧显示红 / 蓝 / 绿三维坐标轴
   - 中间显示一系列红色文字  

   该界面为**正常现象**，后续 VR 遥操作需一直保持在该界面。

---

### 4.4 启动仿真

1. 点击界面右上角 **绿色启动按钮**
2. 选择 **FluidTest_Hotel_Bar_02** 场景打开，使用默认布局
3. 选择 **无仿真程序（手动启动）**

---

### 4.5 开启数采脚本

1. 激活 **Python 数采工程** 的 conda 环境（需已安装本项目依赖；环境名以你本机为准，例如 `orca`、`orcalab` 等）：

       conda activate <你的数采环境名>

2. 进入数采脚本目录并启动（**仿真须已就绪**；`SceneManager` 默认连接 `localhost:50051`，若 OrcaLab 使用其它端口，需与代码或端口映射一致）：

       cd <仓库根>/src/examples/dataCollection
       python data_collection_fluid_tele.py --level fluid_demo --agent_name tiangong2

   参数说明：

   - `--level`：场景 / 关卡标签，决定数据保存目录 `dataset/tiangong2/<level>/` 下的子文件夹名；示例中的 `fluid_demo` 仅为示例，请与你希望的目录命名保持一致。
   - `--agent_name`：天工机器人为 `tiangong2`。

3. 数据默认写入：

       <仓库根>/src/examples/dataCollection/dataset/tiangong2/<level>/<回合uuid>/record/proprio_stats.hdf5

---

### 4.6 手柄操作说明（开始采集）

1. **机械臂复位 / 进入抓取模式**
   - 依次按压 **左摇杆**、**右摇杆**
   - 仿真环境中机械臂可移动，机器人进入抓取操作模式

2. **开始 / 结束一条轨迹记录（与代码一致）**

   - **左手柄握持键（`L_GRIPBUTTON`）**：在「未开始 → 采集中 → 已结束」状态间切换；用于**开始**与**结束**本回合 HDF5 记录（任务成功则保留数据，失败则丢弃，逻辑见 `TaskStatusController`）。

3. **夹爪与双臂（与 `data_collection_fluid_tele.py` 绑定一致）**

   - **左手柄**：`X`、`Y`、**左扳机** 控制左夹爪。
   - **右手柄**：`A`、`B`、**右扳机** 控制右夹爪。
   - **双臂末端**：左手柄 **左 Transform**、右手柄 **右 Transform** 控制双臂 OSC 位姿（VR 中保持 OrcaGymCtrl 界面即可）。

   **务必掌握握持键**以正确开停录制。

---

## 配置要求

- **操作系统**：Ubuntu（本示例未针对 Windows 验证）。
- **GPU**：支持 **CUDA 12.1 及以上** 的 NVIDIA 显卡及匹配驱动。

## 依赖简述

- **OrcaLab（推荐）或 OrcaStudio**：用于加载流体场景、接收仿真与（可选）粒子流。
- **OrcaLink、OrcaSPH**：安装后默认由脚本自动拉起；也可使用 `--manual-fluid` 手动启动。
- **Python**：先安装仓库根目录 `requirements.txt`，再安装本目录额外依赖（若根目录已含 orca-sph 可跳过）。

## 安装

```bash
# 在 OrcaManipulation 仓库根目录
pip install -r requirements.txt

# 或仅安装流体额外依赖
pip install -r src/examples/fluid/requirements.txt
```

## 可选参数

| 参数 | 说明 |
|------|------|
| `--fluid_config` | 流体 JSON 配置，默认 `src/examples/fluid/fluid_sim_config.json` |
| `--manual-fluid` | 不自动启动 OrcaLink / OrcaSPH |
| `--gui` `--sph-gui` | 启用 OrcaSPH 原生 GUI 窗口（粒子/刚体可视化） |
| `--use-all-cpu` | 禁用 OrcaSPH CPU 亲和性 |
| `--build-mode` | `release`（默认，关闭 debug/CSV 开销）或 `debug` |
| `--frame-skip` | MuJoCo frame_skip（默认 20，与 OrcaLink 50Hz 对齐） |
| `--time-step` | MuJoCo 子步 dt（秒，默认 0.001） |
| `--bench` | 基准测试输出 JSON 路径（启用逐帧计时） |

## 配置文件

| 文件 | 说明 |
|------|------|
| `fluid_sim_config.json` | MuJoCo 侧主配置（OrcaLink / OrcaSPH 启动、耦合模式） |
| `sph_sim_config_force_position.json` | SPH 程序配置模板 |
| `scene_config.json` | 流体块 / 墙体等场景生成模板 |

## 代码结构

```
src/envs/fluid/          # 流体耦合核心（OrcaLinkBridge、场景生成、启动编排）
src/envs/fluid_stats/    # 性能统计查看器
src/examples/fluid/      # 独立示例与配置
```

Python API：

```python
from envs.fluid import run_simulation_with_config, start_fluid_coupling, load_fluid_config
```
