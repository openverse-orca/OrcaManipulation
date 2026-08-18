# G1 OSC 关节剥离采集

本目录是可复现包：编译期删掉下肢 / 腰 / 左臂 / 左爪关节，只保留右臂和右爪参与 `mj_step`。左臂侧平举在删关节前烘进 body，渲染和相机挂点不变。

| 文件 | 作用 |
| --- | --- |
| `g1_pick_osc_collection_tele_lerobot_strip.py` | 数采脚本（相对原版增加 `--joint_strip`） |
| `mj_joint_strip.py` | XML 注入：删关节、烘姿态、qpos 补全、安全检查 |
| `uni_osc.json` | 当前实测场景布局（桌、柜、工具箱、工具、G1） |
| `example.yaml` | 任务配置（agent 前缀已改成 `g1_pick_southgrid_usda_1_`） |
| `run_strip.sh` | 与 `g1_osc_strip2` 相同的启动命令 |

机器人名称必须是 **`g1_pick_southgrid_usda_1`**。

---

## 1. 取代码

```bash
git clone -b g1_lerobot_osc https://github.com/openverse-orca/OrcaManipulation.git
cd OrcaManipulation
conda activate orcalab_lerobot
```

环境安装与仓库根目录 `README.md` 一致，不要自行拆 `pip`。

---

## 2. 加载布局（OrcaLab）

1. 启动 OrcaLab，订阅资产 `SouthGrid_Competition_2026` 以及场景里的 G1 / 工具。
2. 打开布局：

```
src/examples/dataCollection/unitree_g1_osc/joint_strip/uni_osc.json
```

3. 确认机器人名为 `g1_pick_southgrid_usda_1`。
4. 点击运行，等到 OrcaGym 就绪，地址默认 `localhost:50051`。不要勾选相机 Recording。

---

## 3. Pico 手柄

```bash
adb reverse tcp:8001 tcp:8001
adb reverse --list
```

应看到 `tcp:8001 tcp:8001`。

---

## 4. 采集（与本次实测相同）

```bash
cd src/examples/dataCollection/unitree_g1_osc/joint_strip
bash run_strip.sh /path/to/output_dataset
```

或直接：

```bash
cd src/examples/dataCollection/unitree_g1_osc/joint_strip
OMP_NUM_THREADS=1 python g1_pick_osc_collection_tele_lerobot_strip.py \
    --task_config example.yaml \
    --agent_name g1_pick_southgrid_usda_1 \
    --lerobot_out /path/to/output_dataset \
    --repo_id local/g1_pick_osc_strip \
    --task "按红色按钮" \
    --fps 20 --clock wall \
    --cameras head,wrist_r --camera_source websocket \
    --dls_lambda 0.2 \
    --joint_strip on --strip_col off \
    --time_step 0.001 --frame_skip 5
```

启动后日志应出现：

```
[STRIP] 已烘入 7 个关节角 → 31 个 body
[STRIP] ✓ 已启用
  规模: nq 113→76  nv 104→68  nu 45→15
[STRIP] 生效 nq=76 nv=68 nu=15
```

安全检查不过会自动回退原始 XML，采集仍能跑，只是没有加速。

仅遥操、不落盘：在上面命令末尾加 `--teleop_only`，并去掉 `--lerobot_out`。

---

## 5. 按键

- 右臂：右手柄位姿（持握激活）
- 右爪：A / B 或右扳机
- 左 Grip：开始 / 保存
- 右 Grip：丢弃本集
- 左臂已锁定侧平举，不响应手柄

---

## 原理（给排障用）

Python 里 `pin` / `JointHold` 只改 `mjData`，全身自由度仍参与 `mj_step`。`--joint_strip on` 在编译前删除非右臂 `<joint>`，body / geom / camera / site 保留。`nq` 变短后由 `mj_joint_strip` 在 `update_local_env` 上补全到原长度，OrcaLab 渲染不错位。
