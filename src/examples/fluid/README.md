# Fluid-MuJoCo 耦合仿真示例

SPH 流体与 MuJoCo 刚体耦合，经 OrcaLink 与 OrcaLab / OrcaStudio 通信。

## 配置要求

- **操作系统**：Ubuntu（本示例未针对 Windows 验证）。
- **GPU**：支持 **CUDA 12.1 及以上** 的 NVIDIA 显卡及匹配驱动。

## 依赖简述

- **OrcaLab（推荐）或 OrcaStudio**：用于加载流体场景、接收仿真与（可选）粒子流。
- **场景**：资产库中订阅并加载 **`water_example`**（或等价带 SPH 标记的流体场景）。
- **OrcaLink、OrcaSPH**：安装后默认由脚本自动拉起；也可使用 `--manual-mode` 手动启动。
- **Python**：先安装仓库根目录 `requirements.txt`，再安装本目录额外依赖（若根目录已含 orca-sph 可跳过）。

## 安装

```bash
# 在 OrcaManipulation 仓库根目录
pip install -r requirements.txt

# 或仅安装流体额外依赖
pip install -r src/examples/fluid/requirements.txt
```

## 运行前（OrcaLab 侧）

1. 启动 OrcaLab，加载 **`water_example`** 场景。
2. 在界面中将 **「无仿真程序」** 切换为 **启动仿真**，再运行下方命令。

## 独立流体仿真

在 **`src`** 目录下执行（PYTHONPATH 需包含 `src`）：

```bash
cd src/examples/fluid
python run_fluid_sim.py
python run_fluid_sim.py --mode record
python run_fluid_sim.py --mode playback --h5 particle_records/foo.h5
python run_fluid_sim.py --manual-mode
```

## 与数据采集框架集成

带流体耦合的 VR 遥操作采集：

```bash
cd src/examples/dataCollection
python data_collection_fluid_tele.py --level fluid_demo --agent_name openloong --task_config example.yaml
```

可选参数：

| 参数 | 说明 |
|------|------|
| `--fluid_config` | 流体 JSON 配置，默认 `src/examples/fluid/fluid_sim_config.json` |
| `--manual-fluid` | 不自动启动 OrcaLink / OrcaSPH |
| `--use-all-cpu` | 禁用 OrcaSPH CPU 亲和性 |

## 配置文件

| 文件 | 说明 |
|------|------|
| `fluid_sim_config.json` | MuJoCo 侧主配置（OrcaLink / OrcaSPH 启动、耦合模式） |
| `sph_sim_config.json` | SPH 程序配置模板 |
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
