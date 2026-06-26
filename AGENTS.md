# OrcaManipulation AI 开发指南

本文件为 AI 代理（如 Trae、Cursor 等）在本仓库工作时提供强制规则。AI 代理必须严格遵守。

## 规则 1：测试与调试环境

AI 代理执行测试、调试、运行脚本时，**必须使用 `orca` conda 环境**。

```bash
# 正确
conda activate orca
python some_script.py

# 错误 — 不要使用 base 或其他环境
conda activate base
python some_script.py
```

`orca` 是 OrcaGym README 推荐的环境名称，已安装本项目所有依赖。使用其他环境会导致依赖缺失或版本不一致。

## 规则 2：依赖 OrcaGym 架构约束

本项目依赖 `orca-gym`，Env 子类基于 OrcaGym 体系开发。Euler 体系的架构约束参考 OrcaGym 仓库的架构文档：

`../OrcaGym/docs/design/architecture/orca_gym_euler_architecture.md`

该文档定义了：

- `OrcaGymEulerEnv` 的公共 API 契约（状态读取 / 写入 / 仿真步进 / 求解器配置 / 名称空间）
- 封装隔离机制（M0-M7 多层封装隔离机制）
- 用户代码的正确使用模式

### 冲突处理

若开发过程中发现 OrcaGym 体系缺少所需功能，或架构约束与本项目需求存在冲突，**请联系 OrcaGym 开发者寻求协助**，不要在本项目代码中绕过封装隔离机制。

## 规则 3：API 隔离强制

本仓库采用 `_` 前缀社区约定 + ruff SLF001 静态检查，引导 AI 和用户走公共 API（OrcaGym 架构 §7）。

### 禁止穿墙访问

不得访问以下 `_` 前缀内部属性（类内部合法的 `self._xxx` 委托除外）：

- `env._gym` / `env._stub` / `env._channel` / `env._studio_bridge`
- `env._gym._sim` / `env._gym._sim._mjData` / `env._gym._sim._mjModel`
- 任何自研类（含本仓库 controller/task/storage/device 子类）的 `_` 前缀属性

### 必须使用公共 API

| 操作 | 正确 | 禁止 |
|------|------|------|
| 读取状态 | `env.data.qpos` / `env.data.body_xpos(name)` / `env.query_*()` | `env._gym._sim._mjData.qpos` |
| 写入状态 | `env.set_joint_qpos()` / `env.apply_body_force()` | `env._gym._sim._mjData.xfrc_applied[...]` |
| 步进 | `env.do_simulation(ctrl, n_frames)` / `env.step()` | `env._gym._sim._mjData.step()` |
| 求解器配置 | `env.sim_config.timestep = 0.002` | `env._gym._sim._mjModel.opt.timestep = 0.002` |

### 必须执行 ruff

提交代码前必须执行，零报警方可提交：

    <conda-base>/envs/orca/bin/python -m ruff check --select SLF001 src/

### 缺失功能时扩展公共方法

若公共 API 不满足需求，**暂停并提交用户决策**，不要穿墙访问内部属性。扩展途径：
- 在 OrcaGym 侧添加公共方法（联系 OrcaGym 开发者）
- 在本仓库 controller/task/storage/device 子类中添加公共访问器
