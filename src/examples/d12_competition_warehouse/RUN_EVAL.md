# 推理命令速查

训练与策略服务部署见 [docs/训练和推理.md](../../../docs/训练和推理.md)。

```bash
# 1. SSH 隧道转发策略服务端口（远程服务时）
ssh -p <ssh端口> -L 8010:localhost:8010 <user>@<服务器IP>

# 2. 本地推理
python eval.py \
    --task_config competition_warehouse.yaml \
    --orcagym_addr localhost:50051 \
    --host localhost --port 8010 \
    --prompt "pick up the electronic component and place it into the box" \
    --max_steps 500 --episodes 1
```

| 参数 | 说明 |
|------|------|
| `--host` / `--port` | 策略服务地址（需先建隧道） |
| `--prompt` | 任务语言指令 |
| `--max_steps` / `--episodes` | 每轮步数 / 轮数 |
| `--sleep` | 按 real_time_step 节奏运行 |
