# 仓内第三方运行时

交付环境只从本目录安装以下三个源码包，不读取开发机 checkout、HOME 或环境变量
指定的源码路径。统一安装入口是仓库根目录的
`bash scripts/install_runtime.sh`；安装采用非 editable wheel，不生成指向源码树的
`.pth` 文件。

| 目录 | 来源基线 | 许可证 | 本地版本 |
|------|----------|--------|----------|
| `lerobot/` | Hugging Face LeRobot 0.3.4 | Apache-2.0 | `0.3.4+orca.1` |
| `televuer/` | Unitree xr_teleoperate `766de45e74373ae0ea66321d942ce538385655a5` | MIT | `4.0.0+orca.1` |
| `openpi-client/` | Physical Intelligence openpi `981483dca0fd9acba698fea00aa6e52d56a66c58` 的 `packages/openpi-client` | Apache-2.0 | `0.1.0+orca.1` |
| `openpi-rtc/` | 本项目自研，基于 openpi `981483dca0fd9acba698fea00aa6e52d56a66c58`，提供 RTC 软前缀伪逆引导服务端扩展 | Apache-2.0 | `0.1.0` |

LeRobot 仅保留本项目使用的数据集运行依赖，并包含以下兼容修改：

- 新增 `save_episode_data_only()`，保存 Parquet 和元数据而不执行视频编码；
- 兼容 `datasets` 3.x/4.x 时间戳读取；
- 清空 episode buffer 前等待图像写入完成；
- 统计流程支持 episode buffer 中的 NumPy 图像数组。

TeleVuer 的交付修改：

- 依赖版本与 NumPy 2.4.6/OpenCV 4.13.0.92 基线一致；不使用 `vuer[all]` 额外项，
  因为它硬钉 `aiohttp==3.10.5`，而已验证栈运行 3.13.1，该额外项的三个包已显式列出；
- 新增 `host` 参数，默认 `127.0.0.1`，供 `adb reverse` 使用；
- 不搜索 `XR_TELEOP_CERT`、`~/.config` 或源码目录证书；
- 回环地址默认 HTTP/WS；非回环 `host` 必须同时显式提供证书和私钥，
  因为 WebXR 只在安全上下文中进入沉浸式会话；
- 私钥和证书不进入仓库。

OpenPI client 只包含远程策略通信客户端，不包含策略服务、模型或 checkpoint。

OpenPI RTC（`openpi-rtc/`）是本项目自研的服务端扩展，在不修改任何 openpi 上游源码的
前提下为 Pi0/Pi0.5 策略服务器启用 RTC 软前缀伪逆引导推理。需与宿主 openpi 环境（锁定
commit `981483d`）配合使用，详见 `docs/openpi_deployment.md`。
