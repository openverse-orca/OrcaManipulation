# 夹爪纯遥操作 · 测试最短流程

目标：从空机器到加载 `uni_test.json` 并跑通 `g1_pick_teleop_gripper_test.py`。本流程不落盘、不采相机。

请把下面的 `/path/to/OrcaManipulation` 换成仓库实际路径。命令请逐条执行，不要合并成脚本。

---

## 1. 安装 OrcaLab

安装 **OrcaLab 6.3**（OrcaGym **26.6.3**）。

在 OrcaLab 资产库订阅：

- `SouthGrid_Competition_2026`
- `g1_pick_with_gripper_usda`（草稿箱搜索该名；场景里 actor 名为 `g1_pick_with_gripper_usda_1`）

---

## 2. 获取代码并建环境

```
git clone -b g1_lerobot https://github.com/openverse-orca/OrcaManipulation.git
```

```
cd /path/to/OrcaManipulation
```

```
conda env create -f environment-unitree.yml
```

```
conda activate orcalab_lerobot
```

```
python -m pip install --no-deps --require-hashes -r requirements.txt
```

```
python -m pip install --no-deps orca-gym==26.6.3
```

```
python -m pip install --no-deps --no-build-isolation ./third_party/lerobot
```

```
python -m pip install --no-deps --no-build-isolation ./third_party/televuer
```

```
python -m pip install --no-deps --no-build-isolation ./third_party/openpi-client
```

```
python scripts/verify_environment.py
```

通过后终端应出现 `Environment verification OK`。

不要执行：`pip install pin`、`pip install pinocchio`、`pip install casadi`、`pip install -r requirements.txt`（无 `--no-deps`）。

本机若还没有 adb：

```
sudo apt install adb
```

---

## 3. 加载场景（OrcaLab）

1. 启动 OrcaLab 6.3。
2. 打开布局文件：

`src/examples/dataCollection/unitree_g1/test/uni_test.json`

3. 确认机器人名称为 `g1_pick_with_gripper_usda_1`。
4. 点击运行，等到 OrcaGym 就绪。地址默认：

```
localhost:50051
```

纯遥操作不需要勾选相机 Recording。

---

## 4. 连接 Pico（每条单独执行）

USB 连接 Pico，在电脑上授权调试。

```
adb devices
```

应看到设备序列号，状态为 `device`。把下面的 `<序列号>` 换成该值。

**每台 Pico 只需做一次**（恢复出厂后需重做）：

```
adb -s <序列号> shell pm disable-user --user 0 com.pvr.roomcapture
```

每次遥操作前：

```
adb reverse tcp:8012 tcp:8012
```

先启动第 5 步的 Python 脚本，等到终端打印访问地址后再执行：

```
adb shell am start -a android.intent.action.VIEW -d http://127.0.0.1:8012/
```

头显里打开的必须是：

```
http://127.0.0.1:8012/
```

不要用 `https://`。

---

## 5. 启动遥操作

另开一个已激活环境的终端：

```
conda activate orcalab_lerobot
```

```
cd /path/to/OrcaManipulation/src/examples/dataCollection/unitree_g1/test
```

```
python -u g1_pick_teleop_gripper_test.py --level default --task_config ../../common/example.yaml --agent_name g1_pick_with_gripper_usda_1 --task "抓取测试" --orcagym_addr localhost:50051 --xr_backend televuer --tv_no_tls --tv_goal_mode rebased_tv --tv_ee_dx 0.03
```

终端出现 `TeleVuer visit URL` 和 `http://127.0.0.1:8012/` 后，再执行上一节的 `adb shell am start ...`。

---

## 6. 操作

| 操作 | 效果 |
|------|------|
| 左 squeeze | 开始 / 结束本轮跟手 |
| 右 squeeze | 重置本轮 |
| 左右 squeeze 同按 | 退出 |
| 右手柄 | 右臂跟随 |
| 左臂 | 侧平举锁定 |
| 左 / 右扳机 | 对应夹爪开合 |

先左 squeeze 开始跟手，再移动右手柄。退出用左右 squeeze 同按，或终端 `Ctrl+C`。
