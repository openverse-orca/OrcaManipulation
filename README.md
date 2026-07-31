# OrcaManipulation

本仓库基于 OrcaLab / OrcaGym，提供人形机器人数据采集、回放与在线推理工具。当前交付覆盖两类机器人：

| 机器人 | 场景 | 说明文档 |
|--------|------|----------|
| 智元 G1 OmniPicker | 四色按按钮、工具整理 | [docs/g1_omnipicker.md](docs/g1_omnipicker.md) |
| 宇树 G1 | 四色按按钮遥操作采集 | [docs/unitree_g1.md](docs/unitree_g1.md) |

请按上表进入对应文档。下文只写两套流程共用的环境与数据格式。

## 兼容环境

| 组件 | 版本 / 要求 |
|------|-------------|
| OrcaLab | 6.3 |
| Python | 3.12 及以上 |
| orca-gym | 与 OrcaLab 6.3 配套的 26.6.x |
| lerobot | 0.3.4（更低版本缺少所需数据集接口） |
| gymnasium | 1.2.1（与 orca-gym 一致） |
| GPU 视频编码 | NVIDIA，需支持 av1_nvenc |

## 环境安装

```bash
conda activate orcalab_lerobot
pip install -r requirements.txt
```

安装时若出现 `lerobot` 要求 `gymnasium<1.0` 一类提示，可以忽略。本项目只使用 lerobot 的数据集读写能力，运行时以 orca-gym 所需版本为准。若 gymnasium 被降级，请执行：

```bash
pip install lerobot==0.3.4 --no-deps
pip install gymnasium==1.2.1
```

在线推理还需要 `openpi_client`：

```bash
export OPENPI_CLIENT_SRC=~/openpi/packages/openpi-client/src
```

## 目录结构

```text
OrcaManipulation/
├── README.md
├── docs/
│   ├── g1_omnipicker.md      # 智元流程
│   └── unitree_g1.md         # 宇树流程
├── requirements.txt
└── src/
    ├── conf/                 # 机器人配置
    ├── controllers/          # 臂/手/任务状态控制
    ├── dataCollectionManager/
    ├── dataStorage/          # LeRobot 存储与相机
    ├── devices/              # 手柄输入
    └── examples/dataCollection/   # 入口脚本与场景布局
```

## 数据集格式

采集结果为 LeRobot v2.1：

```text
<dataset_root>/
├── meta/          # info.json、episodes.jsonl、tasks.jsonl 等
├── data/          # episode_XXXXXX.parquet
└── videos/        # 各路相机 MP4
```

默认分辨率为 480×640，默认帧率为 20 FPS，视频编码为 av1_nvenc。`action` 与 `observation.state` 的维度与字段含义见各机器人文档。
