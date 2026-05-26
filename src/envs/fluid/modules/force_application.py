"""
Module for applying forces to MuJoCo
"""

import numpy as np
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class ForceApplicationModule:
    """Module for applying forces to MuJoCo rigid bodies or SITE points"""
    
    def __init__(self, env, orcalink_client, loop):
        """
        Args:
            env: OrcaGym environment instance
            orcalink_client: OrcaLinkClient instance
            loop: Event loop for async operations
        """
        import sys
        print("[PRINT-DEBUG] ForceApplicationModule.__init__() - START", file=sys.stderr, flush=True)
        logger.debug("[DEBUG] ForceApplicationModule.__init__() - Start")
        self.env = env
        self.client = orcalink_client
        self.loop = loop
        
        # 脉冲力方案：只需记录上一帧施加过力的 site 名称
        # 用于在下一帧开始时清零这些 site 对应的 body 的外力
        self._previous_site_names = set()
        
        print("[PRINT-DEBUG] ForceApplicationModule.__init__() - END", file=sys.stderr, flush=True)
        logger.debug("[DEBUG] ForceApplicationModule.__init__() - Completed")
    
    def subscribe_and_apply_forces(self):
        """Subscribe to rigid body-level forces and apply (ForcePositionMode)"""
        if not self.client or not self.loop:
            return
        
        try:
            forces = self.loop.run_until_complete(
                self.client.subscribe_forces()
            )
            
            if not forces:
                return
            
            for force_data in forces:
                # Apply force to rigid body
                self._apply_force_to_body(force_data)
        except Exception as e:
            logger.error(f"Error applying forces: {e}", exc_info=True)
    
    def subscribe_and_apply_site_forces(self):
        """Subscribe to multi-point forces and apply to SITE (MultiPointForceMode)
        
        脉冲力方案（避免累积误差）：
        - Step 1: 订阅新力
        - Step 2: 如果没有新力，保持上一帧的力不变（SPH 侧未更新）
        - Step 3: 如果有新力，清零旧 sites + 应用新 sites 的力
        
        优势：
        - 无累积误差：每次更新时从零开始设置力
        - 逻辑简单：不需要存储和计算旧力的反向力
        - 保持现状：未收到更新时，维持上一帧的力
        """
        if not self.client or not self.loop:
            logger.debug("[DEBUG] subscribe_and_apply_site_forces - client or loop not available")
            return
        
        try:
            # Step 1: 订阅新力
            logger.debug("[DEBUG] subscribe_and_apply_site_forces - About to call subscribe_forces()...")
            forces = self.loop.run_until_complete(
                self.client.subscribe_forces()
            )
            
            # Step 2: 如果没有新数据，保持上一帧的力不变（SPH 侧未发送更新）
            if not forces:
                logger.debug("[DEBUG] subscribe_and_apply_site_forces - No forces received, keeping previous forces")
                return
            
            # Step 3: 有新数据，统计并输出日志
            site_names = [f.object_id for f in forces]
            logger.debug(f"[DEBUG] subscribe_and_apply_site_forces - Received {len(forces)} SITE forces: {site_names}")
            
            # Step 4: 清零上一帧施加过力的 site 对应的 body
            if self._previous_site_names:
                if hasattr(self.env, 'mj_clear_xfrc_applied_for_site'):
                    for site_name in self._previous_site_names:
                        self.env.mj_clear_xfrc_applied_for_site(site_name)
                    logger.debug(f"[DEBUG] Cleared xfrc_applied for {len(self._previous_site_names)} sites")
                else:
                    logger.warning("Environment does not support mj_clear_xfrc_applied_for_site")
            
            # Step 5: 清空旧记录，准备记录本帧的 site
            self._previous_site_names.clear()
            
            # Step 6: 应用所有新力（在已清零的基础上累加，等价于直接设置）
            for force_data in forces:
                site_name = force_data.object_id  # SITE point ID
                
                # 直接使用接收到的力（已经是MuJoCo Z-up坐标系）
                # 坐标转换已经在C++的GrpcDataMapper中完成
                force_mujoco = np.array(force_data.force, dtype=np.float64)
                torque_mujoco = np.zeros(3, dtype=np.float64)
                
                # 记录 site 名称（下次更新时需要清零）
                self._previous_site_names.add(site_name)
                
                # 性能优化：如果是0值力，跳过 mj_applyFT 调用（已经清零了）
                force_norm = np.linalg.norm(force_mujoco)
                torque_norm = np.linalg.norm(torque_mujoco)
                if force_norm < 1e-9 and torque_norm < 1e-9:
                    continue  # 0 值力，已经清零，不需要应用
                
                if hasattr(self.env, 'mj_apply_force_at_site'):
                    self.env.mj_apply_force_at_site(site_name, force_mujoco, torque_mujoco)
                else:
                    logger.warning(f"Environment does not support mj_apply_force_at_site")
            
            logger.debug(f"[DEBUG] Applied {len(forces)} impulse forces to sites")
            
        except Exception as e:
            logger.error(f"Error applying site forces: {e}", exc_info=True)
    
    def _apply_force_to_body(self, force_data):
        """Apply force to a rigid body using OrcaGym API"""
        try:
            body_name = force_data.object_id
            force = np.array(force_data.force, dtype=np.float64)
            torque = np.array(force_data.torque, dtype=np.float64) if hasattr(force_data, 'torque') else np.zeros(3)
            
            # Apply force using OrcaGym API (required)
            if not hasattr(self.env, 'apply_force_to_body'):
                raise AttributeError(
                    f"Environment does not provide 'apply_force_to_body' method. "
                    f"Cannot apply forces to rigid bodies. "
                    f"Environment type: {type(self.env).__name__}"
                )
            
            self.env.apply_force_to_body(body_name, force, torque)
            logger.debug(f"Applied force to body '{body_name}': F={force}, τ={torque}")
            
        except Exception as e:
            logger.error(f"Error applying force to body '{body_name}': {e}", exc_info=True)
            raise

