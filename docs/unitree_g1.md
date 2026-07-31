# 宇树 G1

本文说明宇树 G1 四色按按钮任务的 VR 遥操作采集。宇树侧通过 TeleVuer 读取 Pico 手柄位姿，端口为 8012；智元侧流程见 [g1_omnipicker.md](g1_omnipicker.md)，请勿混用。

## 场景与相机

本节说明如何在 OrcaLab 中准备宇树按钮场景。

1. 请启动 OrcaLab 6.3。
2. 请加载布局文件 `unitree_button.json`。
3. 请确认场景中机器人为 `unitree_humanoid_robot_1`。
4. 请配置两路相机后，再点击运行，等待服务就绪（默认地址 `localhost:50051`）。

| 相机 | Color Port |
|------|------------|
| 头部 head_cam | 7070 |
| 右腕 wrist_cam | 7080 |

每路相机请勾选 UseNvEnc、Color Camera、Enable，并设置上表端口。两路全部配完后，再勾选 Recording。Recording 勾选后请勿取消，也不要写回布局文件。

## 连接头显

本节说明如何建立手柄位姿链路。

1. 请用 USB 连接 Pico 头显，并确认 `adb devices` 能看到设备。
2. 请执行：`adb reverse tcp:8012 tcp:8012`
3. 请在头显浏览器打开脚本打印的访问地址。关闭加密传输时，地址形如 `http://127.0.0.1:8012/`。
4. 请进入透传模式，并晃动手柄，确认位姿已经上报。

## 启动采集

本节给出完整可复制命令。请先完成上一节连接。

```bash
conda activate orcalab_lerobot
cd src/examples/dataCollection

adb reverse tcp:8012 tcp:8012

python -u g1_pick_collection_tele_lerobot.py \
    --level default \
    --task_config example.yaml \
    --task "按红色按钮" \
    --lerobot_out ~/datasets/g1_unitree_button \
    --repo_id local/g1_unitree_button \
    --fps 20 --clock wall \
    --cameras head,wrist_r \
    --orcagym_addr localhost:50051 \
    --agent_name unitree_humanoid_robot_1 \
    --tv_no_tls \
    --tv_goal_mode rebased_tv \
    --tv_ee_dx 0.03
```

脚本会打印头显访问地址与操作提示。正常进入采集后，终端会周期性出现「正在采集第 N 集」。若只有连接类提示而没有该信息，请停止并检查报错。

## 按键

本节说明采集时的手柄操作。双臂均可跟随手柄；扳机控制灵巧手开合；腿部与腰部保持锁定。

| 功能 | 操作 |
|------|------|
| 双臂跟随 | 左右手柄位姿 |
| 左手抓握 | 左扳机 |
| 右手抓握 | 右扳机 |
| 开始当前集 | 轻按左手柄侧握键 |
| 结束并保存 | 再次轻按左手柄侧握键 |
| 放弃当前集 | 轻按右手柄侧握键 |
| 结束全部采集 | 左右侧握键同时按下 |

未开始采集时，手臂保持静止。请先按左侧握键，机器人才会跟随。握持时请避免左右侧握同时按下，以免误结束全部采集。

## 续采

本节说明如何在已有数据集上继续采集。请在命令中打开续采开关。

脚本启动后应提示已加载的集数与帧数。续采成功后，仍须再按左侧握键才会开始新的一集。若数据集因上次异常退出而不完整，请用备份覆盖后再续采，或不用续采直接重采。

## 故障排查

现象 — 头显打不开页面或一直重连。原因 — 端口转发未建立，或误用了加密地址。处理 — 请重做 `adb reverse`，并按脚本提示打开对应地址。

现象 — 相机超时。原因 — 头部或右腕端口未设为 7070 / 7080，或未启用推流。处理 — 请按上表重配相机并确认 Recording。

现象 — 关节名报错。原因 — 机器人前缀与场景不一致。处理 — 请确认布局中的机器人名，并在启动命令中填写相同的机器人前缀。

现象 — 手柄已连接但机器人不动。原因 — 尚未开始当前集。处理 — 请轻按左手柄侧握键。
