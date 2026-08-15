# pi05_gt_slot01_rewp_integral_kp200_n1000_lora

官方 openpi 上跑这个 checkpoint，只改下面两处。

checkpoint：`29999/`（含 `params/` 和 `assets/gt_slot01_rewp_integral_kp200_n1000/norm_stats.json`）

---

## 1. 新建 `src/openpi/policies/waic_policy.py`

```python
import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class WaicInputs(transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        in_images = data["images"]
        base_image = _parse_image(in_images["cam_head"])
        left_wrist = _parse_image(in_images["cam_wrist_l"])
        right_wrist = _parse_image(in_images["cam_wrist_r"])
        inputs = {
            "state": np.asarray(data["state"], dtype=np.float32),
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist,
                "right_wrist_0_rgb": right_wrist,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class G1OmnipickerOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :18])}
```

---

## 2. 改 `src/openpi/training/config.py`

**import**（`libero_policy` 那行后面）：

```python
import openpi.policies.waic_policy as waic_policy
```

**类**（`_CONFIGS = [` 之前）：

```python
@dataclasses.dataclass(frozen=True)
class LeRobotG1OmnipickerDataConfig(DataConfigFactory):
    default_prompt: str | None = None

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "cam_head": "observation.images.cam_head",
                            "cam_wrist_l": "observation.images.cam_wrist_l",
                            "cam_wrist_r": "observation.images.cam_wrist_r",
                        },
                        "state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        data_transforms = _transforms.Group(
            inputs=[waic_policy.WaicInputs(model_type=model_config.model_type)],
            outputs=[waic_policy.G1OmnipickerOutputs()],
        )
        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("action",),
        )
```

**TrainConfig**（`_CONFIGS` 列表末尾）：

```python
    TrainConfig(
        name="pi05_gt_slot01_rewp_integral_kp200_n1000_lora",
        model=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            action_dim=32,
            action_horizon=50,
            max_token_len=200,
            discrete_state_input=True,
        ),
        data=LeRobotG1OmnipickerDataConfig(
            repo_id="gt_slot01_rewp_integral_kp200_n1000",
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=32,
        ema_decay=None,
        num_train_steps=30_000,
    ),
```

---

## 3. 启动

```bash
uv run scripts/serve_policy.py \
    --port 8010 \
    policy:checkpoint \
    --policy.config=pi05_gt_slot01_rewp_integral_kp200_n1000_lora \
    --policy.dir=/path/to/29999
```
