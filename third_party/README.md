# LeRobot 依赖说明

本目录基于 Hugging Face LeRobot 0.3.4。

本项目在上游版本基础上包含以下兼容性修改：

- 新增 `save_episode_data_only()`，用于保存 Parquet 和元数据而不执行视频编码；
- 兼容 `datasets` 3.x 和 4.x 的时间戳读取；
- 清空 episode buffer 前等待图像写入完成；
- 统计流程支持读取 episode buffer 中的 NumPy 图像数组。

在仓库根目录执行：

```bash
pip install -e ./third_party/lerobot
```

也可以通过项目依赖文件安装：

```bash
pip install -r requirements.txt
```
