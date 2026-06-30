# 回放数据集
python src/examples/dataCollection/g1_omnipicker_collection_replay.py --level nfdw629  --task_config example.yaml --data_root dataset

# 采集数据
python src/examples/dataCollection/g1_omnipicker_collection_tele.py --level nfdw629 

# 数据增强
python src/examples/dataCollection/g1_omnipicker_collection_aug.py --level nfdw629 --aug_count 10 --task_config example.yaml