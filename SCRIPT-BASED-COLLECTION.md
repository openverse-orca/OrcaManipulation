# 基于脚本的数据采集

本文说明如何仅使用 **`data_collection_scripted.py`**，在仿真中按 **预定义末端轨迹（基座系 B）+ 夹爪指令** 驱动 **OpenLoong** 双臂（OSC），完成数据采集相关流程。**不需要 VR 头显与 Pico 手柄**。

Python 侧与仿真通过 **gRPC** 连接，默认 **`localhost:50051`**，与遥操作采集脚本一致。

---

## 一、数据采集软件硬件环境准备
 - **OrcaGym OrcaManipulation** 开源代码库下载
 - **场景资源** 订阅
 
### 1.1 从github获取OrcaManipulation代码仓库。

```bash
git clone https://github.com/openverse-orca/OrcaManipulation.git
# 进入项目目录
cd OrcaManipulation

# 激活 OrcaLab 的 conda 环境（根据你的环境名称调整）
conda activate orcalab  # 激活你创建的 OrcaLab 环境名称

# 安装项目依赖
pip install -r requirements.txt
```

### 1.2 场景资产包准备 

 以下内容，以 `青龙机器人` + `超市场景` 为例，展示一个数据采集场景任务，必需要准备的机器人资产和场景资产。
  
 **订阅相关资产**

1. 资产库链接: https://simassets.orca3d.cn/

2. 资产中心 → 订阅 **ShopScene_Scaning**
![](../../img/shucai/shop_scan.png)

1. 资产中心 → 订阅 **openloong**
![](../../img/shucai/openloong.png)

## 二、数据采集

完成以上准备工作后，以 `青龙机器人` + `超市场景` 为例，添加布局后，开始数据采集任务。

 - **资产包订阅成功**：个人中心有已订阅`ShopScene_Scaning`、`openloong`资产包，

#### 2.1 打开 OrcaLab 中的场景与布局

1. 终端进入 OrcaLab 安装的 conda 环境：
```bash
#激活orcalab conda环境
conda activate orcalab
```
2. 执行命令启动 OrcaLab：
```bash
#启动OrcaLab
orcalab
```
3. 启动过程中会自动下载已订阅的资产，请等待下载与同步完成。
![](../../img/shucai/shop_download.png)

4. 资产下载完成后，在选择场景弹框中，选择 **shop 场景** ，选择默认布局。
![](../../img/shucai/shop_select.png)

5. 在 OrcaLab 客户端菜单栏中，选择 **打开布局**，加载 `shop_openloong_script_based.json` 文件。
```bash
#布局JSON 文件路径（注：布局文件定义了机器人初始位置及姿态）
  ~/OrcaManipulation/src/examples/超市场景青龙机器人数采案例/shop_openloong_script_based.json
```
![](../../img/shucai/shop_layout.png)

6.shopScense示例场景的yaml文件已配置为scriped_example.yaml，如需详细了解配置参数含义，请参考第三章节 数据采集任务配置文件说明
```bash
#scriped_example.yaml文件路径
~/OrcaManipulation/src/examples/dataCollection
```
![](../../img/shucai/shop_example1.png)

### 2.2 启动仿真

1. 点击界面右上角 **绿色启动按钮**
2. 选择 **无仿真程序（手动启动）**
![](../../img/shucai/shop_sim1.png)

### 2.3 开启数采脚本

1. 激活数采脚本所需 conda 环境：
```bash
conda activate orcalab
```

2. 进入数采脚本目录并启动
1) 确认 **`scripted-example.yaml`**、**`pose.yaml`** 已按任务配置（或直接使用仓库内示例）。
2) 在 **`src/examples/dataCollection`** 下执行：
```bash
cd ~/OrcaManipulation/src/examples/dataCollection
#启动数据采集脚本,确保example.yaml文件中资产配置正确
#参数说明（根据实际填写）:level 场景名称，agent_name 机器人名称，task_config 任务文件
python data_collection_scripted.py --level shop_scaning --task_config scripted-example.yaml --pose_file pose.yaml
```

3) 在仿真窗口中观察机器人是否按分段轨迹运动；异常时查看 **`logs/scripted_grasp.log`**（见 **附录 D**）。

**常用参数**：

| 参数 | 含义 |
|------|------|
| `--level` | 数据集逻辑名，对应 `dataset/openloong/<level>/` 等路径 |
| `--task_config` | 场景/物体任务 YAML（与遥操作脚本相同格式），路径相对 **本脚本所在目录** |
| `--pose_file` | 末端轨迹与夹爪配置（YAML/JSON），路径相对 **本脚本所在目录** |

更多参数见 **附录 A** 与源码中 `argparse` 定义。

---

### 2.4（可选）：标定位姿 `--dump_pose`

根据**当前场景**填写或微调 **`pose.yaml`** 中的 **`r_target_b` / `l_target_b`** 时：

```bash
python data_collection_scripted.py --level shop_scaning --task_config scripted-example.yaml --pose_file pose.yaml --dump_pose
```

场景就绪后，脚本会打印基座系下左/右臂末端及名称含 **`bottle`**、**`basket`** 的 body 位置，随后退出。将数值抄入 **`pose.yaml`** 后，再回到 **步骤 4** 正式执行轨迹。

---

### 2.5（可选）：录制 HDF5 `--record_hdf5`

需要生成与 **`data_collection_tele`** 相同结构的 HDF5，供 **`data_collection_aug.py`** 回放时，在 **步骤 4** 的命令末尾增加 **`--record_hdf5`**：

```bash
python data_collection_scripted.py --level shop_scaning --task_config scripted-example.yaml --pose_file pose.yaml --record_hdf5
```

- 任务**成功**结束时写入：`dataset/openloong/<level>/<uuid>/record/proprio_stats.hdf5`
- 任务**失败**则丢弃本回合缓冲；默认**不**另存视频（仅 HDF5）

脚本行为与 HDF5 细节见 **附录 B**。

---

### 2.6（可选）：回放 / 增强采集 `data_collection_aug.py`

在 **OrcaLab 仿真已运行**、**`--level`** 与录制时一致、且 **`dataset/openloong/<level>/`** 下已有单元目录时：

```bash
cd src/examples/dataCollection
python data_collection_aug.py --level shop_scaning --agent_name openloong --task_config scripted-example.yaml
```

新数据默认写入 **`aug_dataset/openloong/<level>/`**。机制说明见 **附录 C**。

---

## 附录 A：命令行参数补充

- **`--dump_pose`**：仅打印位姿，不执行轨迹（见步骤 5）。
- **`--delta_b BX BY BZ`**：无分段文件时，双臂同加基座系位移。
- **`--l_target_b` / `--r_target_b`**、**`--l_quat_b` / `--r_quat_b`**、**`--steps`**、**`--gripper_open` / `--gripper_close`**：单段模式或覆盖文件默认值；详见 **`data_collection_scripted.py`** 文件头与 `argparse`。

---

## 附录 B：脚本行为与 HDF5

1. 加载场景配置 → **`SceneManager`** → **`env.reset()`**、**`update_scene()`**。
2. **`--dump_pose`**：打印 B 系末端与相关物体位置后退出。
3. 否则根据 **`pose_file`** 生成离散轨迹（位置插值、姿态 SLERP），由 **`ScriptedTrajectoryDevice`** 逐步下发 OSC 与夹爪，执行 **`run_episode()`**。
4. 位姿均在 **`base_link`**（仿真中常解析为 **`openloong_base_link`**）定义的 **B 系** 下。
5. **默认**不写 HDF5；**`--record_hdf5`** 时在任务成功时 **`save_data`**，HDF5 布局与遥操作采集兼容；失败则 **`clear_data`**。

---

## 附录 C：数据回放（`data_collection_aug.py`）要点

- **`DataDevice`** 从 **`dataset/openloong/<level>/`** 各单元子目录读取 **`record/proprio_stats.hdf5`**，将 **`/action/end/position`**、**`/action/end/orientation`** 等按时间步送入 OSC；**`AUGMENTATION`** 模式下依据 **`scene_info`** 等恢复场景。
- **`--level`** 必须与 HDF5 所在路径一致；**`--task_config`** 宜与采集时一致或等价，避免场景恢复异常。
- 与「仅脚本采集」的区别：回放由 **磁盘 HDF5** 驱动，**不再读取** **`pose.yaml`**。

---

## 附录 D：日志路径

| 脚本 | 日志文件（相对 `src/examples/dataCollection`） |
|------|-----------------------------------------------|
| `data_collection_scripted.py` | `logs/scripted_grasp.log` |
| `data_collection_aug.py` | `logs/data_collection.log` |

---

## 附录 E：`pose.yaml` 分段字段

典型包含 **`gripper_open` / `gripper_close`** 与 **`segments`**。每一段可包含：

| 字段 | 说明 |
|------|------|
| `steps` | 本段步数；段内从段初插值到段末 |
| `l_hold` / `r_hold` | 为 `true` 时该臂本段保持段初位置 |
| `l_target_b` / `r_target_b` | 段末 B 系绝对位置 `[x,y,z]`（米） |
| `l_delta_b` / `r_delta_b` | 相对本段起点的位移 |
| `l_quat_b` / `r_quat_b` | 可选，段末四元数 `x,y,z,w` |
| `gripper_l` / `gripper_r` | `open` / `close` / `hold` 或数值 |

多段首尾相接；仓库内 **`pose.yaml`** 可作为示例。

---

## 附录 F：相关文件

| 路径 | 说明 |
|------|------|
| `src/examples/dataCollection/data_collection_scripted.py` | 脚本采集入口 |
| `src/examples/dataCollection/data_collection_aug.py` | HDF5 回放与增强采集 |
| `src/examples/超市场景青龙机器人数采案例/shop_openloong_script_based.json` | 推荐 OrcaLab 布局（步骤 2） |
| `src/examples/dataCollection/pose.yaml` | 分段位姿与夹爪示例 |
| `src/examples/dataCollection/scripted-example.yaml` | 场景任务配置示例 |
| `src/conf/openloong_conf.py` | OpenLoong 臂、末端 site、夹爪、基座 |
| `README.md` | 仓库总览与依赖 |
| `QUICK_START.md` | 通用数据采集搭建说明 |
| `DEVELOPER_GUIDE.md` | 开发者说明 |

---
