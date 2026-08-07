"""
Module for publishing positions to SPH
"""

import numpy as np
import logging
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)


class PositionPublishModule:
    """Module for publishing rigid body or SITE positions to SPH"""
    
    def __init__(self, env, orcalink_client, loop, rigid_bodies_config: List[Dict[str, Any]]):
        """
        Args:
            env: OrcaGym environment instance
            orcalink_client: OrcaLinkClient instance
            loop: Event loop for async operations
            rigid_bodies_config: List of rigid body configurations
        """
        import sys
        print(f"[PRINT-DEBUG] PositionPublishModule.__init__() - START", file=sys.stderr, flush=True)
        logger.debug("PositionPublishModule.__init__() - Start")
        self.env = env
        self.client = orcalink_client
        self.loop = loop
        self.rigid_bodies = rigid_bodies_config
        print(f"[PRINT-DEBUG] PositionPublishModule.__init__() - END (rigid_bodies count: {len(rigid_bodies_config)})", file=sys.stderr, flush=True)
        logger.debug(f"PositionPublishModule.__init__() - Initialized with {len(rigid_bodies_config)} rigid bodies")
    
    def publish_positions(self):
        """Publish rigid body positions (ForcePositionMode, SpringConstraintMode)"""
        if not self.client or not self.loop:
            return
        
        try:
            positions = self._collect_body_positions()
            if positions:
                self.loop.run_until_complete(
                    self.client.publish_positions(positions)
                )
        except Exception as e:
            logger.error(f"Error publishing positions: {e}", exc_info=True)
    
    def publish_site_positions(self):
        """Publish SITE positions (MultiPointForceMode)"""
        logger.debug(f"publish_site_positions - START")
        logger.debug(f"publish_site_positions - client={self.client is not None}, loop={self.loop is not None}")
        logger.debug(f"publish_site_positions - rigid_bodies count: {len(self.rigid_bodies)}")
        
        if not self.client or not self.loop:
            logger.warning("publish_site_positions - client or loop not available, returning")
            return
        
        try:
            positions = self._collect_site_positions()
            logger.debug(f"publish_site_positions - Collected {len(positions)} positions")
            
            if positions:
                logger.debug(f"Publish_site_positions - Publishing {len(positions)} positions to channel {self.client.position_channel_id}")
                self.loop.run_until_complete(
                    self.client.publish_positions(positions)
                )
                logger.debug(f"Publish_site_positions - Successfully published {len(positions)} positions")
            else:
                logger.warning("publish_site_positions - No positions collected, not publishing")
        except Exception as e:
            logger.error(f"Error publishing site positions: {e}", exc_info=True)
    
    def _collect_body_positions(self) -> List:
        """Collect rigid body positions from MuJoCo using OrcaGym API"""
        positions = []
        
        try:
            # 1. 收集所有需要查询的 body 名称
            body_names = []
            body_to_object_id = {}  # 映射 body_name -> object_id (用于 OrcaLink)
            
            for body_config in self.rigid_bodies:
                body_name = body_config.get('mujoco_body', '')
                object_id = body_config.get('object_id', body_name)
                if body_name:
                    body_names.append(body_name)
                    body_to_object_id[body_name] = object_id
            
            if not body_names:
                logger.debug("_collect_body_positions - No body names to query")
                return positions
            
            logger.debug(f"_collect_body_positions - Querying {len(body_names)} bodies")
            
            # 2. 更新 MuJoCo 数据
            self.env.mj_forward()
            
            # 3. 批量查询 body 位置、旋转矩阵和四元数（使用 OrcaGym API）
            # 返回值是三个扁平数组的元组: (xpos, xmat, xquat)
            xpos_flat, xmat_flat, xquat_flat = self.env.get_body_xpos_xmat_xquat(body_names)
            
            # 4. 解析扁平数组
            num_bodies = len(body_names)
            xpos_list = xpos_flat.reshape(num_bodies, 3)
            xquat_list = xquat_flat.reshape(num_bodies, 4)
            
            # 5. 转换为 OrcaLink 格式
            for i, body_name in enumerate(body_names):
                object_id = body_to_object_id.get(body_name, body_name)
                pos = xpos_list[i]
                quat = xquat_list[i]
                
                # Convert to position data structure
                try:
                    from data_structures import RigidBodyPosition
                    position_data = RigidBodyPosition()
                    position_data.object_id = object_id
                    position_data.position = np.array(pos, dtype=np.float32)
                    position_data.rotation = np.array(quat, dtype=np.float32)
                except ImportError:
                    # Fallback: create dict-like object
                    position_data = type('RigidBodyPosition', (), {
                        'object_id': object_id,
                        'position': np.array(pos, dtype=np.float32),
                        'rotation': np.array(quat, dtype=np.float32)
                    })()
                
                positions.append(position_data)
                logger.debug(f"_collect_body_positions - Collected body '{body_name}' -> object_id '{object_id}', pos={pos}")
            
            logger.debug(f"_collect_body_positions - Collected {len(positions)} positions")
            return positions
            
        except Exception as e:
            logger.error(f"Error collecting body positions: {e}", exc_info=True)
            return []
    
    def _collect_site_positions(self) -> List:
        """Collect SITE positions from MuJoCo using OrcaGym API"""
        logger.debug(f"_collect_site_positions - START")
        logger.debug(f"_collect_site_positions - rigid_bodies count: {len(self.rigid_bodies)}")
        positions = []
        
        try:
            # 1. 收集所有需要查询的 SITE 名称
            site_names = []
            site_to_object_id = {}  # 映射 site_name -> object_id (用于 OrcaLink)
            
            for body_config in self.rigid_bodies:
                connection_points = body_config.get('connection_points', [])
                for cp in connection_points:
                    site_name = cp.get('site_name', '')
                    # 使用 point_id 作为 object_id，如果没有则使用 site_name
                    object_id = cp.get('point_id', cp.get('object_id', site_name))
                    if site_name:
                        site_names.append(site_name)
                        site_to_object_id[site_name] = object_id
            
            if not site_names:
                logger.debug("_collect_site_positions - No site names to query")
                return positions
            
            logger.debug(f"_collect_site_positions - Querying {len(site_names)} sites")
            
            # 2. 更新 MuJoCo 数据
            self.env.mj_forward()
            
            # 3. 批量查询 SITE 位置和四元数（使用 OrcaGym API）
            site_dict = self.env.query_site_pos_and_quat(site_names)
            
            # 4. 转换为 OrcaLink 格式
            for site_name, site_data in site_dict.items():
                object_id = site_to_object_id.get(site_name, site_name)
                pos = site_data['xpos']
                quat = site_data['xquat']
                
                # 转换为 position data structure
                try:
                    from data_structures import RigidBodyPosition
                    position_data = RigidBodyPosition()
                    position_data.object_id = object_id
                    position_data.position = np.array(pos, dtype=np.float32)
                    position_data.rotation = np.array(quat, dtype=np.float32)
                except ImportError:
                    # Fallback: create dict-like object
                    position_data = type('RigidBodyPosition', (), {
                        'object_id': object_id,
                        'position': np.array(pos, dtype=np.float32),
                        'rotation': np.array(quat, dtype=np.float32)
                    })()
                
                positions.append(position_data)
                logger.debug(f"_collect_site_positions - Collected site '{site_name}' -> object_id '{object_id}', pos={pos}")
            
            logger.debug(f"_collect_site_positions - Collected {len(positions)} positions")
            return positions
            
        except Exception as e:
            logger.error(f"Error collecting site positions: {e}", exc_info=True)
            return []

