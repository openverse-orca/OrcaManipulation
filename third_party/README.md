# 本仓库自带的 lerobot（给客户直接装，不必去网上找 0.3.4）

来源：Hugging Face 官方 lerobot（本机对齐提交约 `882c80d4` / 标称版本 0.3.4）

相对官方干净树，仅额外包含数采需要的两处补丁：

1. `src/lerobot/datasets/lerobot_dataset.py`
   - 新增 `save_episode_data_only()`（先写 parquet/meta，视频编码可后台做）
   - 兼容 datasets 3.x / 4.x 的 timestamp 读取
   - `clear_episode_buffer` 前先 flush 图像写入
2. `src/lerobot/datasets/compute_stats.py`
   - 统计时支持 episode_buffer 里直接存 numpy 图像（NVENC 流式编码路径）

未放入本目录的本地脏东西（有意排除）：

- `.git`、测试、文档媒体、`__pycache__`
- SO101 仿真/调试脚本、`view_camera_*.py`
- 相机/电机/record.py 等与 OrcaManipulation 数采无关的本地改动

安装（在仓库根目录）：

```bash
pip install -e ./third_party/lerobot
```

或直接：

```bash
pip install -r requirements.txt
```
