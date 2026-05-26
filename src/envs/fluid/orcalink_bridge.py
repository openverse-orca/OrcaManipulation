"""
SPH-MuJoCo 多点软连接包装器

该模块封装 OrcaLinkClient 通信逻辑，管理 MuJoCo 与 SPH 之间的多点软连接数据交换。
包括读取刚体 site 位置、发送给 SPH、接收目标位置、更新 mocap body 等功能。
"""

import json
import numpy as np
import logging
import uuid
import re
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Tuple
import sys
from pathlib import Path

# 设置日志
logger = logging.getLogger(__name__)


@dataclass
class ConnectionPoint:
    """单个连接点配置"""
    point_id: str              # OrcaLink 通信标识
    site_name: str             # MuJoCo site 名称 (用于读取位置)
    mocap_name: str            # MuJoCo mocap body 名称 (用于设置目标位置)


@dataclass
class RigidBodyConfig:
    """刚体配置"""
    object_id: str                          # OrcaLink object ID
    mujoco_body: str                        # MuJoCo body 名称
    connection_points: List[ConnectionPoint]
    spring_stiffness: float = 5000.0
    spring_damping: float = 100.0


class OrcaLinkBridge:
    """SPH-MuJoCo 多点软连接包装器（OrcaLink 适配器）"""
    
    def __init__(self, env, config: dict):
        """
        初始化 OrcaLinkBridge（非阻塞）- 只加载配置，不连接
        
        Args:
            env: OrcaGymLocalEnv 实例
            config: 配置字典（从 fluid_config.json 传入）
        """
        import sys
        import traceback
        print("[PRINT-DEBUG] OrcaLinkBridge.__init__() - START", file=sys.stderr, flush=True)
        print("[PRINT-DEBUG] OrcaLinkBridge.__init__() - Call stack:", file=sys.stderr, flush=True)
        for line in traceback.format_stack()[:-1]:
            print(line.strip(), file=sys.stderr, flush=True)
        self.env = env
        self.rigid_bodies: Dict[str, RigidBodyConfig] = {}
        self.orcalink_client = None
        self.loop = None  # 持久事件循环（延迟创建）
        
        # 连接状态管理
        self._connection_state = "pending"  # pending, connecting, connected, failed
        
        # 多点通信映射字典
        # 依赖命名约定：xxx_SPH_SITE_000 对应 xxx_SPH_MOCAP_000
        self.site_index_to_mocap: Dict[Tuple[str, int], str] = {}  # (body_name, site_index) -> mocap_name
        
        # 多点力模式支持
        self.coupling_mode = None  # 延迟从配置读取
        self.force_channel_config = None  # 延迟从配置读取
        self.position_channel_config = None  # 延迟从配置读取
        
        # 当前耦合模式实例（NEW: Strategy Pattern）
        self.current_mode = None
        
        # 生成完整配置（从配置字典）
        self.config = self._prepare_orcalink_config(config)
        self.fluid_config = config  # 保存原始配置字典供后续使用
        
        logger.info(f"OrcaLinkBridge initializing with generated config")
        
        # 解析配置
        self._parse_rigid_bodies()
        
        # 不在这里连接，延迟到 connect() 方法
        logger.info(f"OrcaLinkBridge initialized with {len(self.rigid_bodies)} rigid bodies (connection deferred)")
        print("[PRINT-DEBUG] OrcaLinkBridge.__init__() - END", file=sys.stderr, flush=True)
    
    def _prepare_orcalink_config(self, fluid_config: dict) -> dict:
        """
        从 fluid_config 准备 OrcaLink 配置
        
        Args:
            fluid_config: 完整的 fluid_config.json 内容
            
        Returns:
            dict: OrcaLink 客户端配置
        """
        orcalink_cfg = fluid_config['orcalink']
        
        # 构建 orcalink_client 配置
        orcalink_config = {
            "orcalink_client": {
                "enabled": orcalink_cfg.get('enabled', True),
                "server_address": f"{orcalink_cfg.get('host', 'localhost')}:{orcalink_cfg.get('port', 50351)}",
                **orcalink_cfg.get('client', {})
            },
            "orcalink_bridge": orcalink_cfg.get('bridge', {}),
            "simulation": fluid_config.get('simulation', {}),
            "rigid_bodies": [],  # 动态生成
            "debug": fluid_config.get('debug', {})
        }
        
        # 确保 server_address 正确（覆盖 client 中的值）
        orcalink_config['orcalink_client']['server_address'] = f"{orcalink_cfg.get('host', 'localhost')}:{orcalink_cfg.get('port', 50351)}"
        orcalink_config['orcalink_client']['enabled'] = orcalink_cfg.get('enabled', True)
        
        # 使用 ConfigGenerator 生成 rigid_bodies
        try:
            import sys
            print("[PRINT-DEBUG] _prepare_orcalink_config - About to create ConfigGenerator", file=sys.stderr, flush=True)
            from .utils.config_generator import ConfigGenerator
            
            generator = ConfigGenerator(self.env)
            print("[PRINT-DEBUG] _prepare_orcalink_config - ConfigGenerator created, calling generate_rigid_bodies()", file=sys.stderr, flush=True)
            rigid_bodies = generator.generate_rigid_bodies()
            print(f"[PRINT-DEBUG] _prepare_orcalink_config - generate_rigid_bodies() returned {len(rigid_bodies)} bodies", file=sys.stderr, flush=True)
            orcalink_config['rigid_bodies'] = rigid_bodies
            logger.info(f"Generated {len(rigid_bodies)} rigid bodies from MuJoCo model")
        except Exception as e:
            logger.error(f"Error generating rigid_bodies: {e}", exc_info=True)
            orcalink_config['rigid_bodies'] = []
        
        return orcalink_config
    
    def _parse_rigid_bodies(self):
        """解析刚体和连接点配置"""
        rigid_bodies_cfg = self.config.get('rigid_bodies', [])
        
        for rb_cfg in rigid_bodies_cfg:
            obj_id = rb_cfg['object_id']
            
            # 解析连接点
            connection_points = []
            for pt_cfg in rb_cfg.get('connection_points', []):
                connection_points.append(ConnectionPoint(
                    point_id=pt_cfg['point_id'],
                    site_name=pt_cfg['site_name'],
                    mocap_name=pt_cfg['mocap_name']
                ))
            
            # 获取弹簧参数
            spring_params = rb_cfg.get('spring_params', {})
            
            # 创建刚体配置对象
            self.rigid_bodies[obj_id] = RigidBodyConfig(
                object_id=obj_id,
                mujoco_body=rb_cfg['mujoco_body'],
                connection_points=connection_points,
                spring_stiffness=spring_params.get('stiffness', 5000.0),
                spring_damping=spring_params.get('damping', 100.0)
            )
            
            logger.info(f"  Parsed rigid body '{obj_id}': {len(connection_points)} connection points")
        
        # 构建映射字典
        self._build_mocap_id_mapping()
    
    def _build_mocap_id_mapping(self):
        """构建索引映射：(body_name, site_index) -> mocap_name
        
        注意：不需要获取 mocap_id，因为 set_mocap_pos_and_quat 接受 mocap body 名称
        依赖命名约定：xxx_SPH_SITE_000 对应 xxx_SPH_MOCAP_000
        """
        # 构建索引映射，用于接收时查找
        # site_index_to_mocap: (body_name, index) -> mocap_name
        for rb_config in self.rigid_bodies.values():
            body_name = rb_config.mujoco_body
            for conn_pt in rb_config.connection_points:
                # 从 site_name 提取索引
                # "toys_usda_box_body_SPH_SITE_000" -> 0
                match = re.search(r'SPH_SITE_(\d+)$', conn_pt.site_name)
                if match:
                    index = int(match.group(1))
                    self.site_index_to_mocap[(body_name, index)] = conn_pt.mocap_name
                    logger.debug(f"Mapped (body='{body_name}', index={index}) -> mocap='{conn_pt.mocap_name}'")
                else:
                    logger.warning(f"Cannot extract index from site_name: '{conn_pt.site_name}'")

    @property
    def sph_coupling_mocap_names(self) -> FrozenSet[str]:
        """由 OrcaLink 多点耦合更新的 mocap body 名称集合（轨迹录制时排除）。"""
        names: set[str] = set()
        for rb in self.rigid_bodies.values():
            for cp in rb.connection_points:
                names.add(cp.mocap_name)
        return frozenset(names)
    
    def connect(self) -> bool:
        """
        显式连接到 OrcaLink 服务器
        
        Returns:
            bool: 连接成功返回 True，失败返回 False
        """
        if self._connection_state == "connected":
            return True
        if self._connection_state == "failed":
            logger.warning("Previous connection failed, not retrying")
            return False
        if self._connection_state == "connecting":
            logger.warning("Connection already in progress")
            return False
        
        self._connection_state = "connecting"
        logger.info("Connecting to OrcaLink server...")
        
        try:
            logger.debug("[DEBUG] connect() - Calling _init_orcalink()...")
            self._init_orcalink()
            logger.debug("[DEBUG] connect() - _init_orcalink() completed")
            logger.debug("[DEBUG] connect() - Setting connection state to 'connected'")
            self._connection_state = "connected"
            logger.debug("[DEBUG] connect() - Connection state set")
            logger.info("✅ Connected to OrcaLink successfully")
            logger.debug("[DEBUG] connect() - About to return True")
            import sys
            sys.stdout.flush()
            sys.stderr.flush()
            logger.debug("[DEBUG] connect() - Streams flushed, returning True")
            return True
        except Exception as e:
            self._connection_state = "failed"
            logger.error(f"❌ Failed to connect to OrcaLink: {e}")
            logger.warning("⚠️  SPH integration disabled, continuing without SPH")
            return False
    
    def is_connected(self) -> bool:
        """
        检查是否已连接到 OrcaLink 服务器
        
        Returns:
            bool: 已连接返回 True，否则返回 False
        """
        return self._connection_state == "connected"
    
    def _init_orcalink(self):
        """初始化 OrcaLink 客户端"""
        logger.debug("[DEBUG] _init_orcalink - START")
        try:
            # 导入 OrcaLinkClient（通过 pip 安装的包）
            logger.debug("[DEBUG] _init_orcalink - Importing OrcaLinkClient...")
            from orcalink_client import OrcaLinkClient
            from orcalink_client.data_structures import OrcaLinkConfig
            from orcalink_client.config_loader import _build_orcalink_config_from_dict
            logger.debug("[DEBUG] _init_orcalink - Imports successful")
            
            # 从配置字典构建 OrcaLinkConfig 对象
            logger.info("Building OrcaLink configuration from config dict")
            config = _build_orcalink_config_from_dict(self.config)
            
            # 验证关键配置
            if not config.enabled:
                logger.warning("OrcaLink is disabled in configuration. Connection will be skipped.")
                return
            
            logger.info(f"OrcaLink configuration loaded:")
            logger.info(f"  - enabled: {config.enabled}")
            logger.info(f"  - server_address: {config.server_address}")
            logger.info(f"  - session_id: {config.session_id}")
            logger.info(f"  - client_name: {config.client_name}")
            logger.info(f"  - coupling_mode: {config.coupling_mode}")
            logger.info(f"  - update_rate_hz: {config.update_rate_hz}")
            logger.info(f"  - expected_clients: {config.session.expected_clients}")
            logger.info(f"  - position channel: publish={config.position_channel.publish}, subscribe={config.position_channel.subscribe}")
            logger.info(f"  - force channel: publish={config.force_channel.publish}, subscribe={config.force_channel.subscribe}")
            
            # 创建客户端
            logger.debug("[DEBUG] _init_orcalink - Creating OrcaLinkClient instance...")
            self.orcalink_client = OrcaLinkClient(config)
            logger.debug("[DEBUG] _init_orcalink - OrcaLinkClient instance created")
            
            # 创建持久事件循环
            logger.debug("[DEBUG] _init_orcalink - Creating event loop...")
            import asyncio
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            logger.debug("[DEBUG] _init_orcalink - Event loop created")
            
            # 连接到服务器
            logger.info(f"Connecting to OrcaLink server at {config.server_address}...")
            logger.debug("[DEBUG] _init_orcalink - Calling orcalink_client.initialize()...")
            success = self.loop.run_until_complete(self.orcalink_client.initialize())
            logger.debug(f"[DEBUG] _init_orcalink - orcalink_client.initialize() returned: {success}")
            
            if not success:
                raise RuntimeError(f"Failed to connect to OrcaLink server at {config.server_address}")
            
            logger.info(f"✅ OrcaLinkClient successfully connected to {config.server_address}")
            
            logger.debug("[DEBUG] _init_orcalink - Reading coupling mode config...")
            # 读取耦合模式和通道配置
            self.coupling_mode = config.coupling_mode
            self.force_channel_config = config.force_channel
            self.position_channel_config = config.position_channel
            logger.info(f"Coupling mode: {self.coupling_mode}")
            logger.info(f"Force channel: publish={self.force_channel_config.publish}, subscribe={self.force_channel_config.subscribe}")
            logger.info(f"Position channel: publish={self.position_channel_config.publish}, subscribe={self.position_channel_config.subscribe}")
            
            # Create coupling mode instance (NEW: Strategy Pattern)
            logger.debug("[DEBUG] _init_orcalink - Creating coupling mode instance...")
            self.current_mode = self._create_mode(self.coupling_mode, self.config)
            logger.debug(f"[DEBUG] _init_orcalink - Mode instance created: {type(self.current_mode).__name__ if self.current_mode else 'None'}")
            if self.current_mode:
                import sys
                print(f"[PRINT-DEBUG] _init_orcalink - current_mode is not None: {type(self.current_mode).__name__}", file=sys.stderr, flush=True)
                logger.debug("[DEBUG] _init_orcalink - Initializing coupling mode...")
                # 获取模式特定配置
                mode_config = self.config.get('orcalink_bridge', {}).get(self.coupling_mode, {})
                logger.debug(f"[DEBUG] _init_orcalink - Mode config keys: {mode_config.keys()}")
                
                # 将顶层的 rigid_bodies 配置传入 mode_config
                mode_config['rigid_bodies'] = self.config.get('rigid_bodies', [])
                logger.debug(f"[DEBUG] _init_orcalink - Added {len(mode_config['rigid_bodies'])} rigid bodies to mode_config")
                
                logger.debug(f"[DEBUG] _init_orcalink - Mode config: {mode_config}")
                
                print(f"[PRINT-DEBUG] _init_orcalink - About to call initialize()", file=sys.stderr, flush=True)
                init_result = self.current_mode.initialize(mode_config, self.env, self.orcalink_client, self.loop)
                print(f"[PRINT-DEBUG] _init_orcalink - initialize() returned: {init_result}", file=sys.stderr, flush=True)
                logger.debug(f"[DEBUG] _init_orcalink - Mode initialize() returned: {init_result}")
                
                if init_result:
                    print(f"[PRINT-DEBUG] _init_orcalink - init_result is True, registering channels", file=sys.stderr, flush=True)
                    logger.debug("[DEBUG] _init_orcalink - Registering channels...")
                    self.current_mode.register_channels()
                    print(f"[PRINT-DEBUG] _init_orcalink - register_channels() completed", file=sys.stderr, flush=True)
                    logger.debug(f"[DEBUG] _init_orcalink - Channels registered")
                    logger.info(f"Coupling mode '{self.coupling_mode}' initialized and registered")
                else:
                    logger.error(f"Failed to initialize coupling mode '{self.coupling_mode}'")
                    self.current_mode = None
            else:
                logger.warning("[DEBUG] _init_orcalink - No current_mode created")
            
            logger.debug("[DEBUG] _init_orcalink - COMPLETED SUCCESSFULLY")
            logger.debug("[DEBUG] _init_orcalink - About to exit function")
            import sys
            sys.stdout.flush()
            sys.stderr.flush()
            logger.debug("[DEBUG] _init_orcalink - Exiting now")
            
        except Exception as e:
            logger.error(f"[DEBUG] _init_orcalink - EXCEPTION: {e}")
            logger.error(f"Failed to initialize OrcaLinkClient: {e}", exc_info=True)
            raise

    # 老方案，先保留不删除。
    # def send_positions_to_sph(self):
    #     """
    #     发送 MuJoCo SITE 位置到 SPH
    #     每个 SITE 作为独立的虚拟刚体发送
    #     """
    #     try:
    #         self.env.mj_forward()
    #         from data_structures import RigidBodyPosition
    #         positions_data = []
            
    #         for rb_config in self.rigid_bodies.values():
    #             site_names = [pt.site_name for pt in rb_config.connection_points]
    #             site_dict = self.env.query_site_pos_and_quat(site_names)
                
    #             if not site_dict:
    #                 continue
                
    #             for pt in rb_config.connection_points:
    #                 if pt.site_name not in site_dict:
    #                     continue
                        
    #                 xpos = site_dict[pt.site_name].get('xpos')
    #                 if xpos is None:
    #                     continue
                    
    #                 # 关键：使用 Python 自己的 SITE 名称作为 object_id
    #                 position_obj = RigidBodyPosition(
    #                     object_id=pt.site_name,  # 例如 "toys_usda_box_body_SPH_SITE_000"
    #                     position=np.array(xpos, dtype=np.float32),
    #                     rotation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    #                 )
    #                 positions_data.append(position_obj)
    #                 logger.debug(f"Send SITE: '{pt.site_name}' pos={xpos}")
            
    #         if positions_data:
    #             self.loop.run_until_complete(self.orcalink_client.publish_positions(positions_data))
    #             logger.debug(f"Sent {len(positions_data)} SITE positions")
    #     except Exception as e:
    #         logger.error(f"Error in send_positions_to_sph: {e}", exc_info=True)
    
    # def receive_and_apply_sph_targets(self):
    #     """
    #     接收 SPH 发送的 SITE 位置，映射到本地 MOCAP
    #     对方的 SITE_000 → 本地的 MOCAP_000
    #     """
    #     try:
    #         sph_data = self.loop.run_until_complete(
    #             self.orcalink_client.subscribe_positions(max_count=100, enable_sync_window=True)
    #         )
            
    #         if not sph_data:
    #             return
            
    #         mocap_updates = {}
            
    #         for obj_data in sph_data:
    #             # obj_data.object_id 是对方的 SITE 名称，例如 "toys_usda_box_body_SPH_SITE_000"
    #             remote_site_name = obj_data.object_id
                
    #             # 从 SITE 名称提取 body_name 和索引
    #             # "toys_usda_box_body_SPH_SITE_000" -> body_name="toys_usda_box_body", index=0
    #             match = re.search(r'(.+)_SPH_SITE_(\d+)$', remote_site_name)
    #             if not match:
    #                 logger.warning(f"Cannot parse SITE name: '{remote_site_name}'")
    #                 continue
                
    #             body_name = match.group(1)
    #             site_index = int(match.group(2))
                
    #             # 映射到本地 MOCAP
    #             key = (body_name, site_index)
    #             if key not in self.site_index_to_mocap:
    #                 logger.warning(f"No local MOCAP for remote SITE '{remote_site_name}'")
    #                 continue
                
    #             mocap_name = self.site_index_to_mocap[key]
                
    #             # 不需要检查 mocap_name_to_id，直接使用 mocap_name
    #             # set_mocap_pos_and_quat 接受 mocap body 名称
    #             target_pos = obj_data.position
    #             target_quat = np.array([1.0, 0.0, 0.0, 0.0])
                
    #             # 调试打印：使用 DEBUG 级别
    #             logger.debug(f"[Python←OrcaLink] RECV '{remote_site_name}': "
    #                        f"pos=[{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}] "
    #                        f"(MuJoCo Z-up, from C++) → MOCAP '{mocap_name}'")
                
    #             mocap_updates[mocap_name] = {
    #                 'pos': target_pos,
    #                 'quat': target_quat
    #             }
                
    #             logger.debug(f"Receive remote SITE '{remote_site_name}' -> local MOCAP '{mocap_name}'")
            
    #         if mocap_updates:
    #             self.env.set_mocap_pos_and_quat(mocap_updates)
    #             logger.debug(f"Updated {len(mocap_updates)} mocap positions")
    #     except Exception as e:
    #         logger.error(f"Error in receive_and_apply_sph_targets: {e}", exc_info=True)
    
    # def subscribe_and_apply_forces(self):
    #     """
    #     订阅多点力并应用到 MuJoCo SITE 点
        
    #     重要：必须在 mj_forward() 之后，mj_step() 之前调用
    #     """
    #     import numpy as np
        
    #     try:
    #         # 订阅力数据
    #         forces = self.loop.run_until_complete(
    #             self.orcalink_client.subscribe_forces()
    #         )
            
    #         logger.info(f"[DEBUG] subscribe_forces returned: {len(forces) if forces else 0} force data units")
            
    #         if not forces:
    #             logger.info("[DEBUG] No forces received from SPH")
    #             return
            
    #         # 应用每个力到对应的 SITE 点
    #         for i, force_data in enumerate(forces):
    #             site_name = force_data.object_id  # SITE 点 ID
                
    #             # 打印原始力数据（SPH Y-up坐标系）
    #             force_sph = force_data.force
    #             logger.info(f"[DEBUG] Force {i}: SITE='{site_name}', SPH_force=[{force_sph[0]:.3f}, {force_sph[1]:.3f}, {force_sph[2]:.3f}]")
                
    #             # 坐标转换: SPH 发送 [fx, -fz, fy] (MuJoCo Z-up)
    #             # MuJoCo 需要 [fx, fy, fz]
    #             force_mujoco = np.array([
    #                 force_data.force[0],   # fx
    #                 force_data.force[2],   # fy
    #                 -force_data.force[1]   # fz
    #             ], dtype=np.float64)
                
    #             logger.info(f"[DEBUG] Force {i}: MuJoCo_force=[{force_mujoco[0]:.3f}, {force_mujoco[1]:.3f}, {force_mujoco[2]:.3f}]")
                
    #             # 零力矩（当前场景不需要施加力矩）
    #             torque_mujoco = np.zeros(3, dtype=np.float64)
                
    #             # 使用封装的方法在 SITE 点施加力
    #             self.env.mj_apply_force_at_site(site_name, force_mujoco, torque_mujoco)
                
    #             logger.debug(f"Applied force to SITE '{site_name}': {force_mujoco}")
        
    #     except Exception as e:
    #         logger.error(f"Error applying forces: {e}", exc_info=True)
        
    def step(self) -> bool:
        """
        SPH 同步单步（参考 C++ SimulatorBase::timeStep 实现）
        
        流程：
        1. 检查连接和会话就绪
        2. 委托给当前模式处理数据交换
        3. 模式内部处理流控检查
        
        Returns:
            bool: 是否允许执行 MuJoCo step
                  - True: 同步正常，允许 MuJoCo 执行下一步
                  - False: 需要暂停 MuJoCo（流控暂停/未就绪/未连接）
        """
        # 1. 检查连接状态
        if not self.is_connected():
            logger.debug("[DEBUG] step() - not connected, returning False")
            return False
        
        # 2. 检查会话是否就绪（等待所有客户端加入）
        if not self.orcalink_client.is_session_ready_status():
            logger.debug("[DEBUG] step() - Waiting for all clients to join session...")
            return False
        
        logger.debug("[DEBUG] step() - Session ready, delegating to mode")
        try:
            # Delegate to current mode (Strategy Pattern)
            if self.current_mode:
                logger.debug(f"[DEBUG] step() - Calling current_mode.step() [{type(self.current_mode).__name__}]")
                result = self.current_mode.step()
                logger.debug(f"[DEBUG] step() - current_mode.step() returned: {result}")
                return result
            else:
                # No coupling mode configured
                logger.error("No coupling mode set, cannot proceed")
                return False
            
        except Exception as e:
            logger.error(f"OrcaLinkBridge.step error: {e}", exc_info=True)
            return False
    
    def _create_mode(self, mode_name: str, config: dict):
        """Factory method to create coupling mode instance"""
        import sys
        print(f"[PRINT-DEBUG] _create_mode() - START with mode_name={mode_name}", file=sys.stderr, flush=True)
        logger.debug(f"[DEBUG] _create_mode() - Creating mode: {mode_name}")
        
        from .coupling_modes import ForcePositionMode, SpringConstraintMode, MultiPointForceMode
        print(f"[PRINT-DEBUG] _create_mode() - Imports completed", file=sys.stderr, flush=True)
        
        if mode_name == 'force_position':
            print(f"[PRINT-DEBUG] _create_mode() - Creating ForcePositionMode", file=sys.stderr, flush=True)
            return ForcePositionMode()
        elif mode_name == 'spring_constraint':
            print(f"[PRINT-DEBUG] _create_mode() - Creating SpringConstraintMode", file=sys.stderr, flush=True)
            return SpringConstraintMode()
        elif mode_name == 'multi_point_force':
            print(f"[PRINT-DEBUG] _create_mode() - Creating MultiPointForceMode", file=sys.stderr, flush=True)
            mode_instance = MultiPointForceMode()
            print(f"[PRINT-DEBUG] _create_mode() - MultiPointForceMode created: {mode_instance}", file=sys.stderr, flush=True)
            return mode_instance
        else:
            logger.error(f"Unknown coupling mode: {mode_name}")
            return None
    
    def generate_sph_scene(self, output_path: str = "generated_scene.json", 
                          scene_config_path: str = None) -> str:
        """
        生成 SPH scene.json 文件
        
        从当前 MuJoCo model 自动生成完整的 SPH scene.json 文件，
        包括 Configuration、Materials、RigidBodies、FluidBlocks 部分，
        支持坐标系转换和几何映射推断。
        
        Args:
            output_path: 输出文件路径 (默认: "generated_scene.json")
            scene_config_path: 场景生成配置文件路径 (如未指定，使用默认配置)
            
        Returns:
            str: 生成的文件路径
            
        Raises:
            Exception: 如果生成失败
        """
        try:
            from .utils.scene_generator import SceneGenerator
            
            # 加载或使用默认配置
            if scene_config_path is None:
                # 尝试从同级目录加载 scene_config.json
                default_config_path = Path(__file__).parent / "scene_config.json"
                if default_config_path.exists():
                    scene_config_path = str(default_config_path)
                    logger.info(f"Using default scene config: {scene_config_path}")
            
            # 创建生成器
            generator = SceneGenerator(self.env, config_path=scene_config_path)
            
            # 生成完整的 scene.json（而不是只生成 RigidBodies）
            scene_data = generator.generate_complete_scene(
                output_path=output_path,
                include_fluid_blocks=True,
                include_wall=True
            )
            
            logger.info(f"Generated complete SPH scene.json at: {output_path}")
            logger.info(f"  - RigidBodies: {len(scene_data.get('RigidBodies', []))} bodies")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating SPH scene.json: {e}", exc_info=True)
            raise
    
    def close(self):
        """清理资源"""
        try:
            if self.orcalink_client and self.loop:
                self.loop.run_until_complete(self.orcalink_client.shutdown())
                self.loop.close()
                logger.info("OrcaLinkClient disconnected")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}", exc_info=True)

