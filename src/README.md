操作说明：
确认 Orcalab 已经启动，并且加载json文件（该文件位于src/examples/dataCollection/test-env）。
PICO 头显需要打开 OrcaGymCtrl，并保持应用处于运行状态。
打开终端：
conda activate orcalab
进入 OrcaManipulation
运行：
python src/examples/dataCollection/g1_omnipicker_collection_tele.py 
初次运行需按下左右摇杆并再次按下右摇杆，开始操作机器人
| VR 操作             | 机器人功能       |
| 左手柄移动/旋转      | 控制左机械臂      |
| 右手柄移动/旋转      | 控制右机械臂      |
| 左摇杆左右           | 车辆转向      |
| 右摇杆上下           | 车辆前进 / 后退   |
| 左 Grip 按住         | 机器人上半身持续上升  |
| 松开左 Grip          | 停止上升并保持当前位置 |
| 右 Grip 按住         | 机器人上半身持续下降  |
| 松开右 Grip          | 停止下降并保持当前位置 |
| 按下右摇杆           | 腰部自动复位到最低位置 |
| 按下左摇杆           | 开始 / 结束数据采集 |
|左 Trigger            | 左夹爪控制     |  
|右 Trigger            | 右夹爪控制       |

