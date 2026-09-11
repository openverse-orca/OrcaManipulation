# LeRobot 采集 / 回放 / 推理

本目录集中存放新增 LeRobot 链路的环境锁定文件与说明。不要与根目录 `requirements.txt`（HDF5 旧入口 `data_collection_tele.py`）混用。

## 安装

需要 Ubuntu 22.04/24.04、Conda、Python 3.12.13、NVIDIA 驱动（采集视频要 RTX 40 系及以上的 AV1 NVENC）。请从仓库根目录执行，并从新环境创建，不要更新已有环境。

```bash
conda env create -f docs/lerobot/environment.yml
conda activate orcalab_lerobot
bash docs/lerobot/install_runtime.sh
```

装完检查：

```bash
python docs/lerobot/verify_environment.py
```

`install_runtime.sh` 按 `requirements.txt` 哈希锁定安装 pip 依赖，再装 `orca-gym==26.7.3`、`orca-lab==26.7.3`、`third_party/lerobot`、`third_party/openpi-client`。本仓遥操作走 PicoJoystick，不安装 televuer。

重新生成锁定文件（在本目录下）：

```bash
uv pip compile requirements.in --constraint constraints.txt \
  --python-version 3.12 --python-platform x86_64-unknown-linux-gnu \
  --generate-hashes --emit-index-url --index-strategy unsafe-best-match \
  --no-emit-package numpy --no-emit-package scipy \
  --no-emit-package argparse --output-file requirements.txt
```

## 文档

| 文件 | 内容 |
|------|------|
| [dataset.md](dataset.md) | LeRobot v2.1 写入约定与 Schema |
| [policy_inference.md](policy_inference.md) | 策略 Schema / Client / Device |
| [cameras_and_video.md](cameras_and_video.md) | 相机拉起与 NVENC |
| [openpi_deployment.md](openpi_deployment.md) | OpenPI 训练与推理服务部署 |
| [examples/inference/README.md](../../src/examples/inference/README.md) | 通用推理入口参数 |
| [examples/southgrid/README.md](../../src/examples/southgrid/README.md) | 南网任务入口 |

## 本目录文件

| 文件 | 用途 |
|------|------|
| `environment.yml` | Conda 环境 `orcalab_lerobot`（Python 3.12.13、NumPy、SciPy） |
| `requirements.in` | 直接依赖 |
| `constraints.txt` | 传递依赖锁定 |
| `requirements.txt` | 带哈希的 pip lock |
| `install_runtime.sh` | 安装脚本 |
| `verify_environment.py` | 版本与关键能力检查 |
