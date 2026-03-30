# OrcaLab 数采示例

以下以 **青龙机器人 + 超市场景** 为例。

---

## 一、订阅相关资产

1. 资产库链接  
   https://simassets.orca3d.cn/

2. 资产中心 → 生活资产 → 订阅 **shop 场景资产**

3. 资产中心 → 机器人资产 → 订阅 **openloong 机器人资产**

---

## 二、Pico 的安装

1. 将 `PicoController.apk` 安装包下载到电脑上  
   安装包路径：

       ~/OrcaManipulation/src/examples/超市场景青龙机器人数采案例/pico安装包

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

1. 终端进入 OrcaLab 安装的 conda 环境：

       conda activate orcalab

2. 执行命令启动 OrcaLab：

       orcalab

3. 启动过程中会自动下载第一步中订阅的资产，请等待下载与同步完成。

4. 选择 **shop 场景** 打开，使用默认布局。

5. 在 OrcaLab 中选择 **打开布局**，加载 `shop_openloong.json` 文件。

   JSON 文件路径：

       /home/orcatest/OrcaManipulation/src/examples/超市场景青龙机器人数采案例/shop_openloong.json

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
2. 选择 **无仿真程序（手动启动）**

---

### 4.5 开启数采脚本

1. 激活数采脚本所需 conda 环境：

       conda activate OrcaManipulation

2. 进入数采脚本目录并启动：

       cd ~/OrcaManipulation/src/examples/dataCollection
       python data_collection_tele.py  --level shop_scaning --agent_name openloong  --task_config example.yaml

---

### 4.6 手柄操作说明（开始采集）

1. **机械臂复位 / 进入抓取模式**
   - 依次按压 **左摇杆**、**右摇杆**
   - 仿真环境中机械臂可移动，机器人进入抓取操作模式

2. **手柄功能映射**

   **左手手柄**
   - **Y 键长按**：左手抓取
   - **X 键**：左手放开

   **右手手柄**
   - **B 键长按**：右手抓取
   - **A 键长按**：右手放开

## 五、FAQ
1. 在 OrcaLab 中选择 **打开布局**，加载 `shop_openloong.json` 文件时出现，布局加载失败，eg： Request failed. Spawnable not found., asset_path: assets/e071469a36d3c8aa/shopscene_scaning/shop_scene/els/basket

   去看JSON文件中asset_path的路径是否与订阅资产路径一致，不一致请修改：

       eg：JSON文件中"asset_path": " assets/e071469a36d3c8aa/shopscene_scaning/shop_scene/els/basket"与实际订阅不同，修改路径为：
       "asset_path": "assets/e071469a36d3c8aa/default_project/shop_scene/els/basket"
2. 在启动数采脚本时报找不到spawnable物品
    去看(../dataCollection/example.yaml)文件中actor:spawnable路径是否与订阅资产路径一致，不一致请修改：

       eg：yaml文件中"assets/e071469a36d3c8aa/default_project/shop_scene/prefabs/goods/jar_01",与实际订阅不同，修改路径为：
       "assets/e071469a36d3c8aa/shopscene_scaning/shop_scene/prefabs/goods/jar_01",
---