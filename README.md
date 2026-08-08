# OrcaManipulation

本仓库基于 OrcaLab / OrcaGym，提供人形机器人数据采集、回放与在线推理工具，支持以下两类机器人：

| 机器人 | 支持场景 | 采集文档 | 推理文档 |
|--------|----------|----------|----------|
| 智元 G1 OmniPicker | 四色按按钮、工具整理（脚本化 + Pico 遥操作） | [docs/g1_omnipicker_collection.md](docs/g1_omnipicker_collection.md) | [docs/g1_omnipicker_inference.md](docs/g1_omnipicker_inference.md) |
| 宇树 G1 | 四色按按钮（TeleVuer 遥操作 / 脚本化） | [docs/unitree_g1_collection.md](docs/unitree_g1_collection.md) | [docs/unitree_g1_inference.md](docs/unitree_g1_inference.md) |

请按上表进入对应文档查看完整流程。下文说明两套机器人共用的环境配置与数据格式。

如需基于本仓库采集的数据进行策略训练或部署推理服务，请参阅 [docs/openpi_deployment.md](docs/openpi_deployment.md)（RTC 异步推理为其中可选章节）。

---

## 已验证交付基线

此分支面向 **Linux x86-64**，以当前稳定运行环境的实际导入结果为准：

| 组件 | 精确版本 |
|------|----------|
| OrcaLab / OrcaGym | 6.3 / 26.6.3 |
| Python | 3.12.13 |
| NumPy / SciPy | 2.4.6 / 1.15.3（Conda 管理） |
| Pinocchio / CasADi | 3.9.0 / 3.7.2（Conda 管理） |
| Gymnasium / MuJoCo | 1.2.1 / 3.7.0 |
| PyTorch | 2.7.1+cpu，torchvision 0.22.1+cpu |
| LeRobot / TeleVuer / OpenPI client | 仓库内 `third_party/` 固定源码 |
| 宇树 IK 模型 | 仓库内 `src/examples/dataCollection/assets/g1/` |
| 视频 | OpenCV 4.13.0.92 QT5、PyAV 17.0.1、`av1_nvenc` |

系统仍需提供 OrcaLab 6.3、NVIDIA 驱动/GPU 和 ADB（TeleVuer 遥操）。
三份场景 JSON 还引用 OrcaLab 资产命名空间
`assets/e071469a36d3c8aa/...` 与 `assets/13951baeb514b4b9/...`；
这些商业资产不在 Git 仓库中，交付机的 OrcaLab 资产库必须已包含它们。

宇树 IK 所需的 G1 描述文件（`g1_body29_hand14.urdf` 与它引用的 49 个 STL）
已随仓库提供，来源为 Unitree `unitree_ros` 的 `g1_description`（BSD-3-Clause，
见该目录下的 `LICENSE`）。Pinocchio 建模会解析全部网格，缺一个就无法启动，
所以不要裁剪该目录。

---

## 环境安装

必须新建环境，不要在已有环境上执行 `conda env update`，也不要复制开发机的
`site-packages`：

```bash
cd /path/to/OrcaManipulation
conda env create -f environment-unitree.yml
conda activate orcalab_lerobot
bash scripts/install_runtime.sh
```

`install_runtime.sh` 会先验证 Conda 管理的数值 ABI，再按哈希锁安装 pip wheel；
`orca-gym==26.6.3` 因发布元数据与已验证 NumPy/SciPy 不一致而有意使用
`--no-deps`；最后从本仓库以非 editable 方式安装 LeRobot、TeleVuer 和
OpenPI client，并执行数据集写入、编码器、GUI 与来源路径自检。

禁止单独执行 `pip install pin`、`pip install pinocchio`、`pip install casadi`
或普通的 `pip install -r requirements.txt`。它们可能让 cmeel/PyPI 文件覆盖
Conda 的 Pinocchio ABI。安装成功后的统一复检命令是：

```bash
python scripts/verify_environment.py
```

`requirements.in` 是人工维护的顶层输入，`requirements.txt` 是 Linux/Python
3.12 的哈希锁；正常交付安装只读取后者。仓内依赖的来源与本地修改见
[third_party/README.md](third_party/README.md)。

---

## 目录结构

```text
OrcaManipulation/
├── README.md
├── requirements.in               # pip 顶层输入
├── requirements.txt              # 含哈希的 pip 锁
├── constraints.txt               # 传递依赖锁定输入（仅生成锁时使用）
├── environment-unitree.yml       # Conda 数值 ABI 精确构建
├── scripts/                      # 安装与环境验证
├── third_party/
│   ├── lerobot/                  # LeRobot 数据集运行时
│   ├── televuer/                 # TeleVuer XR 运行时
│   └── openpi-client/            # OpenPI 推理客户端
├── docs/
│   ├── g1_omnipicker_collection.md   # 智元 · 采集
│   ├── g1_omnipicker_inference.md    # 智元 · 推理
│   ├── unitree_g1_collection.md      # 宇树 · 采集
│   ├── unitree_g1_inference.md       # 宇树 · 推理
│   └── openpi_deployment.md          # OpenPI 完整部署（训练 / 推理服务 / 可选 RTC）
├── pyproject.toml
└── src/
    ├── conf/
    ├── controllers/
    ├── dataCollectionManager/
    ├── dataStorage/
    ├── devices/
    ├── envs/dataCollection/
    ├── scene/
    ├── task/
    └── examples/
        ├── dataCollection/
        │   ├── assets/g1/              # 宇树 G1 URDF + STL（IK 必需）
        │   ├── common/                 # 共用 example.yaml / scripted 基座
        │   ├── g1_omnipicker/          # 智元采集入口与布局
        │   └── unitree_g1/             # 宇树采集入口与布局
        └── inference/
            ├── g1_omnipicker/          # 智元推理入口
            └── unitree_g1/             # 宇树推理入口
```

---

## 入口脚本

### 数据采集（`src/examples/dataCollection/`）

| 脚本 | 机器人 | 用途 |
|------|--------|------|
| `g1_omnipicker/g1_omnipicker_collection_scripted_tool_lerobot.py` | 智元 | 工具整理脚本化自动采集 |
| `g1_omnipicker/g1_omnipicker_collection_scripted_button_lerobot.py` | 智元 | 四色按钮脚本化自动采集 |
| `g1_omnipicker/g1_omnipicker_collection_tele_lerobot.py` | 智元 | Pico VR 遥操作采集（左臂锁定） |
| `g1_omnipicker/g1_omnipicker_replay_lerobot.py` | 智元 | LeRobot Parquet 数据集回放 |
| `unitree_g1/g1_pick_collection_tele_lerobot.py` | 宇树 | TeleVuer 遥操作采集（双臂） |
| `unitree_g1/g1_pick_collection_scripted_button_lerobot.py` | 宇树 | 四色按钮脚本化自动采集 |

### 在线推理（`src/examples/inference/`）

| 脚本 | 机器人 | 用途 |
|------|--------|------|
| `g1_omnipicker/eval_g1_omnipicker_lerobot.py` | 智元 | 按钮任务在线推理（OpenPI） |
| `g1_omnipicker/eval_g1_omnipicker_tool_lerobot.py` | 智元 | 工具任务在线推理（OpenPI） |
| `unitree_g1/eval_g1_pick_lerobot.py` | 宇树 | 按钮任务在线推理（OpenPI） |

---

## 数据集格式

采集结果保存为 LeRobot v2.1 格式：

```text
<dataset_root>/
├── meta/
│   ├── info.json              # 数据集元信息（fps / 维度 / 相机键等）
│   ├── episodes.jsonl         # 每集的 index / length / task
│   ├── episodes_stats.jsonl
│   └── tasks.jsonl            # 语言指令列表
├── data/chunk-000/
│   └── episode_XXXXXX.parquet # action / observation.state / timestamp
└── videos/chunk-000/
    ├── observation.images.cam_head/
    ├── observation.images.cam_wrist_l/    # 智元专有
    └── observation.images.cam_wrist_r/
```

- 默认相机分辨率为 480×640，默认帧率为 20 FPS。
- 视频编码为 MP4（`av1_nvenc`）。
- `action` 与 `observation.state` 的维度及字段含义因机器人而异，详见各机器人文档。
