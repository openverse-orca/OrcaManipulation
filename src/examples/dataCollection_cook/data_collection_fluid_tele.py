"""已迁移：流体 SPH 遥操请用 ../dataCollection/data_collection_fluid_tele.py。"""
import sys

print(
    "dataCollection_cloth 不包含流体耦合。\n"
    "  布料遥操/轨迹回放: data_collection_cloth_tele.py\n"
    "  流体 VR 遥操:      ../dataCollection/data_collection_fluid_tele.py",
    file=sys.stderr,
)
raise SystemExit(2)
