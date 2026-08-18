#! /bin/bash


cd src/examples/dataCollection/unitree_g1_osc

# 遥操模式
python g1_pick_osc_collection_tele_lerobot.py  --agent_name g1_pick_southgrid_usda_1 --teleop_only
# 采集模式
python g1_pick_osc_collection_tele_lerobot.py --lerobot_out ~/dataset --agent_name g1_pick_southgrid_usda_1