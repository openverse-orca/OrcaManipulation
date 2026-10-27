1. 代码版本

建议固定使用 WB-WaM/WB-WAM-Official 提交：

6a9bf12d6c36d42ec66ee2aa7235ea2058175c40

需要将原始训练配置中的：

model:
  _target_: fastwam.runtime.create_fastwam_optional_idm

改为：

model:
  _target_: wbwam.runtime.create_wbwam_optional_idm

其余模型参数保留，并使用这份权重自带的 dataset_stats.json。

2. SONIC 权重与配置

按公开部署说明，使用 nvidia/GEAR-SONIC 根目录的 Default SONIC 原始版本，四个文件配套：

model_encoder.onnx
model_decoder.onnx
observation_config.yaml
planner_sonic.onnx

3. 推理配置和启动脚本
迁移后的训练配置 (wb_wam_push_cart_review/config.public.yaml)
push_cart 部署配置 (wb_wam_push_cart_review/deploy.push_cart.yaml)
检查与离线推理脚本 (wb_wam_push_cart_review/run_push_cart.sh)
部署配置使用 wbwam 首帧动作推理模式，任务指令为 Push the chair or cart forward.，图像为 224×224，采用 absolute 动作表示

准备好正式 bridge 环境、SONIC encoder、Wan2.2 VAE 和文本编码资源后，可运行真实模型的无硬件推理：
export WBWAM_REPO=/你的完整WB-WAM仓库
export WBWAM_BRIDGE_PYTHON="$WBWAM_REPO/bridge/.venv-wam/bin/python"
export DIFFSYNTH_MODEL_BASE_PATH=/你的基础模型目录
export WBWAM_BRIDGE_ENCODER_MODEL=/你的SONIC目录/model_encoder.onnx
bash /home/xaai/wb_wam_push_cart_review/run_push_cart.sh dry-run

脚本在这里：run_push_cart.sh (wb_wam_push_cart_review/run_push_cart.sh)

同目录还有两个配套配置：
- deploy.push_cart.yaml (wb_wam_push_cart_review/deploy.push_cart.yaml)：推理部署配置
- config.public.yaml (wb_wam_push_cart_review/config.public.yaml)：迁移后的训练配置