# 拐角往返诊断：corner-recovery-loop

- Status: OPEN
- progress_status: collecting（状态机离线对照通过，等待原拐角场景）
- 项目根：/home/user/Desktop/g1_lidar/OrcaManipulation
- 外部证据目录：/tmp/OrcaManipulation-corner-recovery-loop
- HTTP 日志：/tmp/OrcaManipulation-corner-recovery-loop/trae-debug-log-corner-recovery-loop.ndjson
- env：/tmp/OrcaManipulation-corner-recovery-loop/corner-recovery-loop.env
- HTTP：http://127.0.0.1:7777/event，健康检查及上报探测成功。
- 受阻事件同时写入项目既有日志：src/examples/dataCollection/logs/g1_autopilot.log。
- 已有证据：用户提供 Terminal#944-1017，反复 arc blocked → recovery 1/2 → backed 0.60/0.61m。

## 假设与证据
| ID | 假设 | 状态 | 证据 |
|---|---|---|---|
| A | 预测弧线受阻触发恢复 | CONFIRMED（触发链）；具体几何 INCONCLUSIVE | 用户终端；新增采样日志仅完成合成障碍验证 |
| B | 窗口位移清零恢复次数，使往返无上限 | CONFIRMED | NDJSON第2行：原方法0.2m前进清零次数；第4行：修复后保留次数且第三次恢复锁停 |
| C | 重规划重复不可执行拐角 | INCONCLUSIVE | 尚无用户原场景完整路径坐标 |
| D | 同步定位耗时造成卡顿 | INCONCLUSIVE | 尚无原场景定位耗时 |

## 修改
- 首次现有代码修改仅插桩，在原 `_stalled` 上执行离线复现。
- 删除0.15m窗口位移清零恢复次数的逻辑；单次导航累计上限通过 --max-recoveries 配置，默认2，0禁止倒车。此处是保守总预算，不宣称可以识别不同拐角。
- 预测刹停同步清零 last_v_cmd 并重置卡死窗口。
- 受阻时记录完整路径、位姿、前视点、期望及限幅曲率、命中格、采样距离、恢复计数、两次受阻报告之间的最大定位耗时。HTTP超时20ms，仅受阻分支上报，项目日志保留副本。

## 离线前后对照
内联Python AST提取当前实际类执行，未落盘测试文件、未下发仿真驱动。
- 修改前：恢复次数1，前进0.2m后变为0。
- 修改后：相同输入仍为1；穿插前进的两次恢复后，第三次请求锁定停车。
- 禁止恢复、后方受阻均停车。
- 调用实际update、pure pursuit和arc检查：合成障碍触发几何日志，速度记账为0，后续不会将主动停车认作卡死。
- Python编译、git diff --check通过。
- NDJSON第3行为离线合成障碍数据，不是用户拐角证据；第4行为post-fix-unit结果。

## 下一步与最终证据门
沿用用户原启动命令、目标、地图及初始场景复现。未知用户当前具体目标与场景摆放，不擅自启动另一导航进程。获取 [DEBUG] nav_blocked 及 recovery 日志，判断受阻几何；诊断尚未关闭，保留采集服务及插桩。
证据来源：用户终端、上述NDJSON、内联离线测试输出。progress_status: collecting。尚不能宣称原拐角导航已经修好。
