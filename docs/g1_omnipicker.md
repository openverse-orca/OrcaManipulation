# 智元 G1 OmniPicker

本文说明智元 G1 OmniPicker 的场景准备、脚本化采集、Pico 遥操作、回放与在线推理。

## 场景与相机

本节说明如何在 OrcaLab 中准备智元场景。

1. 请启动 OrcaLab 6.3。
2. 请按任务加载布局：四色按按钮使用 `g1_button.json`，工具整理使用 `g1_tool.json`。
3. 请确认场景名称与 `example.yaml` 中的 `level_name: "example"` 一致。
4. 请配置三路相机后，再点击运行，等待服务就绪（默认地址 `localhost:50051`）。

| 相机 | 代码名称 | Color Port |
|------|----------|------------|
| 左腕 | camera_wrist_l_color | 7070 |
| 右腕 | camera_wrist_r_color | 7080 |
| 头部 | camera_head_color | 7090 |

每路相机请勾选 UseNvEnc 与 Color Camera，并设置上表端口。三路全部配完后，再勾选 Recording。Recording 勾选后请勿取消，也不要写回布局文件；每次重新打开布局后需要再勾一次。

## 运行准备

本节给出后续命令的公共前置步骤。

```bash
conda activate orcalab_lerobot
cd src/examples/dataCollection
```

## 工具整理脚本化采集

本节说明如何自动采集工具整理数据。请先加载 `g1_tool.json`。

```bash
python g1_omnipicker_collection_scripted_tool_lerobot.py \
    --task_config example.yaml \
    --lerobot_out ~/datasets/g1_tool_scripted \
    --repo_id local/g1_omnipicker_tool \
    --num_episodes 20 \
    --fps 20
```

断点续采时请在命令中打开续采开关。

## 四色按钮脚本化采集

本节说明如何按颜色配额自动采集按钮数据。请先加载 `g1_button.json`。

```bash
python g1_omnipicker_collection_scripted_button_lerobot.py \
    --task_config example.yaml \
    --lerobot_out ~/datasets/g1_button_scripted \
    --repo_id local/g1_omnipicker_button \
    --counts 25,25,25,25 \
    --fps 20 --clock wall
```

颜色数量参数的顺序为红、绿、黄、蓝。非交互环境请直接在命令中写明四色数量；交互终端也可由脚本询问。

## Pico 遥操作采集

本节说明如何用 Pico 手柄采集。智元遥操作走 Pico 原生手柄链路，端口为 8001。

1. 请加载与任务对应的布局文件。
2. 请新开终端执行：`adb reverse tcp:8001 tcp:8001`
3. 请启动采集脚本：

```bash
python g1_omnipicker_collection_tele_lerobot.py \
    --task_config example.yaml \
    --lerobot_out ~/datasets/g1_tele \
    --repo_id local/g1_omnipicker \
    --task "按红色按钮" \
    --fps 20 --clock wall
```

`--task` 可填 `按红色按钮` / `按绿色按钮` / `按黄色按钮` / `按蓝色按钮`，或工具任务描述。

### 按键

| 功能 | 操作 |
|------|------|
| 右臂移动 | 右手柄位姿（左臂锁定） |
| 夹爪 | 主键张开、副键闭合，或用扳机连续闭合 |
| 开始当前集 | 轻按左侧握键 |
| 结束并保存 | 再次轻按左侧握键 |
| 放弃当前集 | 轻按右侧握键 |
| 结束全部采集 | 左右侧握键同时按下 |

手柄连接成功后，机器人仍保持静止。请再按一次左侧握键，机器人才会跟随手柄。未开始采集时，仅左侧握键有效。

### 续采

续采时请在命令中打开续采开关。脚本启动后应打印已加载的集数与帧数，并周期性提示正在采集第几集。若只有连接提示而没有上述信息，说明数据集加载失败，请查看终端报错。异常退出可能导致元数据与视频不一致；此时请用备份覆盖数据集后再续采，或不用续采直接重采。

## 数据回放

本节说明如何回放已采集数据。请加载与采集时一致的布局。

```bash
python g1_omnipicker_replay_lerobot.py \
    --dataset_dir /path/to/lerobot_dataset \
    --task_config example.yaml \
    --episode 1 \
    --steps_per_frame 3 \
    --render_every 5
```

集号从 1 开始；省略集号则顺序播放全部。需要循环播放时请在命令中打开循环开关。

## 在线推理

本节说明如何连接策略服务做在线推理。

1. 请在策略侧启动服务（端口示例为 8010）。
2. 请设置 `OPENPI_CLIENT_SRC`。
3. 请按场景选择脚本：

按钮任务：

```bash
python eval_g1_omnipicker_lerobot.py \
    --task_config example.yaml \
    --host localhost --port 8010 \
    --prompt "按红色按钮" \
    --max_steps 500 --episodes 3
```

工具任务：

```bash
python eval_g1_omnipicker_tool_lerobot.py \
    --task_config example.yaml \
    --host localhost --port 8010 \
    --prompt "整理工具" \
    --max_steps 10000 --episodes 1
```

提示语须与训练数据中的任务描述一致。

## 状态维度说明

智元数据中 `action` 与 `observation.state` 均为 18 维：左右臂末端位置与四元数，以及左右夹爪归一化开度。夹爪归一化公式为 `(电机值 + 1) / 3`。基座坐标系为 `g1_omnipicker_body_link1`。

## 故障排查

现象 — 相机连不上。原因 — 端口未按上表配置，或未勾选 Recording。处理 — 请重新配置三路相机并确认端口监听。

现象 — 续采后机器人不动。原因 — 手柄已连接但尚未开始采集。处理 — 请再按左侧握键进入采集。

现象 — 续采报错或只见连接提示。原因 — 上次异常退出导致数据集不完整。处理 — 请用备份覆盖后再续采，或重新采集。
