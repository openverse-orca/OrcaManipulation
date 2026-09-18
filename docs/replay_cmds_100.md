# G1 Pick OSC Replay — 100 条调参命令

> 生成规则：kp∈{150,200,250,300}，ki∈[0,2]（0.5–0.7 高权重），
> steps∈{10,20,25,40,50,100}（50 附近多）。01-11 沿用历史参数。

## 001  kp=300  ki=0  steps=50

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 300 --track_ki 0 --steps 50 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay001.txt
```

## 002  kp=300  ki=0  steps=10

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 300 --track_ki 0 --steps 10 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay002.txt
```

## 003  kp=300  ki=0.03  steps=10

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 300 --track_ki 0.03 --steps 10 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay003.txt
```

## 004  kp=300  ki=0.03  steps=20

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 300 --track_ki 0.03 --steps 20 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay004.txt
```

## 005  kp=300  ki=0.1  steps=20

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 300 --track_ki 0.1 --steps 20 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay005.txt
```

## 006  kp=300  ki=0.5  steps=20

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 300 --track_ki 0.5 --steps 20 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay006.txt
```

## 007  kp=300  ki=0.5  steps=20

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 300 --track_ki 0.5 --steps 20 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay007.txt
```

## 008  kp=300  ki=0.5  steps=40

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 300 --track_ki 0.5 --steps 40 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay008.txt
```

## 009  kp=200  ki=0.5  steps=25

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 200 --track_ki 0.5 --steps 25 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay009.txt
```

## 010  kp=200  ki=0.75  steps=25

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 200 --track_ki 0.75 --steps 25 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay010.txt
```

## 011  kp=200  ki=0.75  steps=50

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 200 --track_ki 0.75 --steps 50 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay011.txt
```

## 012  kp=150  ki=0.35  steps=25

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 150 --track_ki 0.35 --steps 25 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay012.txt
```

## 013  kp=200  ki=0.6  steps=10

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 200 --track_ki 0.6 --steps 10 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay013.txt
```

## 014  kp=200  ki=0.75  steps=100

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 200 --track_ki 0.75 --steps 100 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay014.txt
```

## 015  kp=250  ki=0.5  steps=25

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 250 --track_ki 0.5 --steps 25 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay015.txt
```

## 016  kp=300  ki=0.0  steps=20

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 300 --track_ki 0.0 --steps 20 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay016.txt
```

## 017  kp=200  ki=0.25  steps=50

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 200 --track_ki 0.25 --steps 50 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay017.txt
```

## 018  kp=200  ki=0.6  steps=25

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 200 --track_ki 0.6 --steps 25 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay018.txt
```

## 019  kp=300  ki=0.7  steps=20

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 300 --track_ki 0.7 --steps 20 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay019.txt
```

## 020  kp=300  ki=0.75  steps=50

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 300 --track_ki 0.75 --steps 50 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay020.txt
```

## 021  kp=200  ki=0.55  steps=50

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 200 --track_ki 0.55 --steps 50 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay021.txt
```

## 022  kp=150  ki=0.45  steps=10

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 150 --track_ki 0.45 --steps 10 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay022.txt
```

## 023  kp=200  ki=0.45  steps=100

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 200 --track_ki 0.45 --steps 100 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay023.txt
```

## 024  kp=200  ki=0.25  steps=100

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 200 --track_ki 0.25 --steps 100 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay024.txt
```

## 025  kp=250  ki=0.6  steps=40

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 250 --track_ki 0.6 --steps 40 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay025.txt
```

## 026  kp=200  ki=0.35  steps=25

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 200 --track_ki 0.35 --steps 25 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay026.txt
```

## 027  kp=250  ki=0.3  steps=20

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 250 --track_ki 0.3 --steps 20 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay027.txt
```

## 028  kp=200  ki=0.2  steps=20

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 200 --track_ki 0.2 --steps 20 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay028.txt
```

## 029  kp=250  ki=0.32  steps=50

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 250 --track_ki 0.32 --steps 50 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay029.txt
```

## 030  kp=150  ki=0.8  steps=40

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 150 --track_ki 0.8 --steps 40 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay030.txt
```

## 031  kp=300  ki=0.9  steps=50

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 300 --track_ki 0.9 --steps 50 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay031.txt
```

## 032  kp=250  ki=0.4  steps=40

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 250 --track_ki 0.4 --steps 40 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay032.txt
```

## 033  kp=300  ki=0.72  steps=40

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 300 --track_ki 0.72 --steps 40 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay033.txt
```

## 034  kp=200  ki=0.95  steps=50

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 200 --track_ki 0.95 --steps 50 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay034.txt
```

## 035  kp=200  ki=0.2  steps=40

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 200 --track_ki 0.2 --steps 40 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay035.txt
```

## 036  kp=200  ki=0.55  steps=100

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 200 --track_ki 0.55 --steps 100 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay036.txt
```

## 037  kp=250  ki=0.5  steps=25

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 250 --track_ki 0.5 --steps 25 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay037.txt
```

## 038  kp=200  ki=0.65  steps=100

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 200 --track_ki 0.65 --steps 100 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay038.txt
```

## 039  kp=200  ki=0.15  steps=20

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 200 --track_ki 0.15 --steps 20 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay039.txt
```

## 040  kp=150  ki=0.55  steps=50

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 150 --track_ki 0.55 --steps 50 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay040.txt
```

## 041  kp=300  ki=0.75  steps=50

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 300 --track_ki 0.75 --steps 50 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay041.txt
```

## 042  kp=250  ki=0.85  steps=50

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 250 --track_ki 0.85 --steps 50 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay042.txt
```

## 043  kp=250  ki=0.75  steps=10

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 250 --track_ki 0.75 --steps 10 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay043.txt
```

## 044  kp=200  ki=0.65  steps=20

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 200 --track_ki 0.65 --steps 20 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay044.txt
```

## 045  kp=250  ki=1.1  steps=40

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 250 --track_ki 1.1 --steps 40 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay045.txt
```

## 046  kp=250  ki=0.45  steps=50

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 250 --track_ki 0.45 --steps 50 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay046.txt
```

## 047  kp=250  ki=1.8  steps=25

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 250 --track_ki 1.8 --steps 25 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay047.txt
```

## 048  kp=300  ki=0.6  steps=40

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 300 --track_ki 0.6 --steps 40 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay048.txt
```

## 049  kp=250  ki=0.65  steps=50

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 250 --track_ki 0.65 --steps 50 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay049.txt
```

## 050  kp=300  ki=0.7  steps=50

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 300 --track_ki 0.7 --steps 50 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay050.txt
```

## 051  kp=250  ki=0.6  steps=50

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 250 --track_ki 0.6 --steps 50 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay051.txt
```

## 052  kp=200  ki=1.0  steps=40

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 200 --track_ki 1.0 --steps 40 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay052.txt
```

## 053  kp=300  ki=0.1  steps=20

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 300 --track_ki 0.1 --steps 20 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay053.txt
```

## 054  kp=300  ki=0.8  steps=25

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 300 --track_ki 0.8 --steps 25 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay054.txt
```

## 055  kp=250  ki=0.5  steps=50

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 250 --track_ki 0.5 --steps 50 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay055.txt
```

## 056  kp=250  ki=0.55  steps=20

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 250 --track_ki 0.55 --steps 20 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay056.txt
```

## 057  kp=150  ki=1.5  steps=25

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 150 --track_ki 1.5 --steps 25 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay057.txt
```

## 058  kp=200  ki=0.5  steps=40

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 200 --track_ki 0.5 --steps 40 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay058.txt
```

## 059  kp=200  ki=0.6  steps=50

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 200 --track_ki 0.6 --steps 50 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay059.txt
```

## 060  kp=300  ki=0.65  steps=25

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 300 --track_ki 0.65 --steps 25 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay060.txt
```

## 061  kp=200  ki=0.4  steps=10

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 200 --track_ki 0.4 --steps 10 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay061.txt
```

## 062  kp=300  ki=0.7  steps=50

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 300 --track_ki 0.7 --steps 50 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay062.txt
```

## 063  kp=250  ki=0.02  steps=25

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 250 --track_ki 0.02 --steps 25 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay063.txt
```

## 064  kp=200  ki=0.7  steps=10

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 200 --track_ki 0.7 --steps 10 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay064.txt
```

## 065  kp=150  ki=0.5  steps=50

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 150 --track_ki 0.5 --steps 50 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay065.txt
```

## 066  kp=200  ki=0.5  steps=50

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 200 --track_ki 0.5 --steps 50 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay066.txt
```

## 067  kp=300  ki=0.85  steps=10

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 300 --track_ki 0.85 --steps 10 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay067.txt
```

## 068  kp=250  ki=0.4  steps=10

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 250 --track_ki 0.4 --steps 10 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay068.txt
```

## 069  kp=300  ki=1.4  steps=50

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 300 --track_ki 1.4 --steps 50 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay069.txt
```

## 070  kp=200  ki=0.5  steps=50

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 200 --track_ki 0.5 --steps 50 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay070.txt
```

## 071  kp=300  ki=0.52  steps=20

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 300 --track_ki 0.52 --steps 20 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay071.txt
```

## 072  kp=150  ki=0.7  steps=50

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 150 --track_ki 0.7 --steps 50 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay072.txt
```

## 073  kp=150  ki=2.0  steps=50

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 150 --track_ki 2.0 --steps 50 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay073.txt
```

## 074  kp=250  ki=0.75  steps=10

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 250 --track_ki 0.75 --steps 10 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay074.txt
```

## 075  kp=150  ki=0.7  steps=50

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 150 --track_ki 0.7 --steps 50 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay075.txt
```

## 076  kp=200  ki=0.5  steps=25

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 200 --track_ki 0.5 --steps 25 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay076.txt
```

## 077  kp=300  ki=0.82  steps=40

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 300 --track_ki 0.82 --steps 40 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay077.txt
```

## 078  kp=300  ki=0.9  steps=40

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 300 --track_ki 0.9 --steps 40 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay078.txt
```

## 079  kp=300  ki=0.62  steps=20

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 300 --track_ki 0.62 --steps 20 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay079.txt
```

## 080  kp=200  ki=0.8  steps=50

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 200 --track_ki 0.8 --steps 50 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay080.txt
```

## 081  kp=250  ki=0.7  steps=100

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 250 --track_ki 0.7 --steps 100 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay081.txt
```

## 082  kp=300  ki=0.12  steps=50

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 300 --track_ki 0.12 --steps 50 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay082.txt
```

## 083  kp=250  ki=1.0  steps=50

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 250 --track_ki 1.0 --steps 50 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay083.txt
```

## 084  kp=300  ki=0.7  steps=50

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 300 --track_ki 0.7 --steps 50 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay084.txt
```

## 085  kp=300  ki=0.7  steps=50

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 300 --track_ki 0.7 --steps 50 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay085.txt
```

## 086  kp=300  ki=0.22  steps=40

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 300 --track_ki 0.22 --steps 40 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay086.txt
```

## 087  kp=200  ki=0.92  steps=100

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 200 --track_ki 0.92 --steps 100 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay087.txt
```

## 088  kp=200  ki=0.5  steps=40

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 200 --track_ki 0.5 --steps 40 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay088.txt
```

## 089  kp=200  ki=0.3  steps=50

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 200 --track_ki 0.3 --steps 50 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay089.txt
```

## 090  kp=200  ki=1.3  steps=25

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 200 --track_ki 1.3 --steps 25 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay090.txt
```

## 091  kp=250  ki=0.6  steps=100

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 250 --track_ki 0.6 --steps 100 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay091.txt
```

## 092  kp=250  ki=1.2  steps=40

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 250 --track_ki 1.2 --steps 40 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay092.txt
```

## 093  kp=250  ki=0.6  steps=40

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 250 --track_ki 0.6 --steps 40 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay093.txt
```

## 094  kp=300  ki=0.6  steps=100

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 300 --track_ki 0.6 --steps 100 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay094.txt
```

## 095  kp=250  ki=0.7  steps=50

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 250 --track_ki 0.7 --steps 50 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay095.txt
```

## 096  kp=300  ki=0.55  steps=25

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 300 --track_ki 0.55 --steps 25 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay096.txt
```

## 097  kp=200  ki=0.65  steps=25

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 200 --track_ki 0.65 --steps 25 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay097.txt
```

## 098  kp=150  ki=0.5  steps=20

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 150 --track_ki 0.5 --steps 20 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay098.txt
```

## 099  kp=200  ki=0.6  steps=100

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 200 --track_ki 0.6 --steps 100 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay099.txt
```

## 100  kp=250  ki=0.42  steps=20

```bash
python src/examples/southgrid/unitree_g1/g1_pick_osc_replay_dual.py \
  --task_config src/examples/southgrid/unitree_g1/example.yaml --lerobot_out /home/dht/hebing/g1_pick_scripted_dual --episode_index 0 --agent_name g1_pick --dls_lambda 0.2 --dls_sigma_th 0.12 --null_kp 10 --track_clamp 0.08 --grasp_x_bias 0 \
  --kp 250 --track_ki 0.42 --steps 20 \
  --log_txt /home/dht/hebing/g1_pick_scripted_dual/replay100.txt
```
