# 宇树 g1_pick 四色按钮 waypoint

目录内每个颜色一个文件：`red.yaml` / `green.yaml` / `yellow.yaml` / `blue.yaml`。

每个文件包含若干接触 `waypoints`（建议每色 3 个），字段为 28 维关节角 `q`。

录制（一次只写一个颜色文件，换色改 `--color`）：

```bash
python -u record_g1_pick_button_waypoints.py \
  --color red --per_color 3 --output_dir pose_g1_pick_button ...
```

数采：

```bash
python -u g1_pick_collection_scripted_button_lerobot.py --pose_candidates pose_g1_pick_button ...
```
