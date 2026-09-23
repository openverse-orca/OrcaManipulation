# Euler 刚体通道（MuJoCoFlow）

本目录只驱动 OrcaGym Euler 后端的 GPU 刚体。不加布、不连 OrcaLink、不连 `:50451`。

旧布料入口在 `dataCollection_cloth/`，CPU MuJoCo 入口在 `dataCollection/`。不要混用。

## 前置

1. 分支：Gym / Manipulation 用 `jimmy/deformable`。
2. Studio 先打开关卡并 **Play**。关卡里有机器人，不要放布。确认 `:50051` 在听。
3. conda：本机用能 `import orca.flow`、`import orca.euler`、且没有 NVIDIA `warp` 的环境（团队常叫 `OrcaFlow_Flow`）。不要在 Manipulation 的 `orca` 环境里硬跑 MuJoCoFlow。

```bash
source ~/miniconda3/etc/profile.d/conda.sh   # 按本机 conda 根目录改
conda activate OrcaFlow_Flow
python -c "import orca.flow; import orca.euler; import orca_gym"
```

## 工作目录

```bash
cd /home/hjadmin/OrcaApr24/OrcaManipulation/src/examples/dataCollection_euler
```

日志写在本目录 `logs/`。`--mjc-agent-prefix` 必须与 Studio 里机器人实例名一致，脚本不写死预制体名。`--sim-device` 必须由命令行传入，不写死 `cuda:0`。

## 回放（先做）

Studio 已 Play 后：

```bash
python data_collection_euler_rigid_tele.py \
  --level <关卡名> \
  --agent_name g1_omnipicker \
  --mjc-agent-prefix <Studio预制体名> \
  --sim-device cuda:0 \
  --replay \
  --replay_data <本机JSON路径> \
  --no-collect
```

`--sim-device` 按机器改成 `cuda:0` 或 `hip:0`。回放 JSON 用 PicoJoystick 格式，路径自己传，不要改脚本去写死文件名。

## PICO 实时

同一条命令去掉 `--replay` / `--replay_data`。USB 手柄需要：

```bash
adb reverse tcp:8001 tcp:8001
```

坐标只经过 `PicoJoystickDevice` 那一层。

## 不要做的事

- 不要 `from envs.cloth ...`
- 不要起 OrcaLink / 独立 XPBD
- 不要给本脚本加 `--esdf-path` 或 `:50451`
- 加布时另开入口，不要改本 rigid 脚本
