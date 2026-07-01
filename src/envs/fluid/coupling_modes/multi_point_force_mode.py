"""
Multi-Point Force coupling mode implementation
"""

import logging
from typing import Optional, Dict, Any
from .base import ICouplingMode
from ..modules.force_application import ForceApplicationModule
from ..modules.position_publish import PositionPublishModule

# 配置模块 logger
logger = logging.getLogger(__name__)


class MultiPointForceMode(ICouplingMode):
    """Multi-Point Force coupling mode
    
    This mode implements multi-point force coupling:
    - MuJoCo sends SITE point positions to SPH
    - SPH decomposes fluid forces to tetrahedron anchor points
    - SPH sends decomposed forces to MuJoCo SITE points
    """
    
    def __init__(self):
        import sys
        print("[PRINT-DEBUG] MultiPointForceMode.__init__() - START", file=sys.stderr, flush=True)
        self.force_application_module: Optional[ForceApplicationModule] = None
        self.position_publish_module: Optional[PositionPublishModule] = None
        self.env = None
        self.orcalink_client = None
        self.loop = None
        self.config = {}
        print("[PRINT-DEBUG] MultiPointForceMode.__init__() - END", file=sys.stderr, flush=True)
    
    def initialize(self, config: Dict[str, Any], env, orcalink_client, loop) -> bool:
        """Initialize the mode"""
        import sys
        
        print("[PRINT-DEBUG] MultiPointForceMode.initialize() - START", file=sys.stderr, flush=True)
        logger.debug("[DEBUG] MultiPointForceMode.initialize() - Start")
        self.env = env
        self.orcalink_client = orcalink_client
        self.loop = loop  # 直接使用传入的 loop，不再从 orcalink_client 获取
        self.config = config
        logger.debug(f"[DEBUG] MultiPointForceMode.initialize() - Loop: {self.loop is not None}")
        
        # Initialize modules
        logger.debug("[DEBUG] MultiPointForceMode.initialize() - Creating ForceApplicationModule...")
        print("[PRINT-DEBUG] MultiPointForceMode.initialize() - Creating ForceApplicationModule", file=sys.stderr, flush=True)
        self.force_application_module = ForceApplicationModule(env, orcalink_client, self.loop)
        print("[PRINT-DEBUG] MultiPointForceMode.initialize() - ForceApplicationModule created", file=sys.stderr, flush=True)
        logger.debug("[DEBUG] MultiPointForceMode.initialize() - ForceApplicationModule created")
        
        logger.debug("[DEBUG] MultiPointForceMode.initialize() - Creating PositionPublishModule...")
        print("[PRINT-DEBUG] MultiPointForceMode.initialize() - Creating PositionPublishModule", file=sys.stderr, flush=True)
        self.position_publish_module = PositionPublishModule(
            env, orcalink_client, self.loop, config.get('rigid_bodies', []))
        print("[PRINT-DEBUG] MultiPointForceMode.initialize() - PositionPublishModule created", file=sys.stderr, flush=True)
        logger.debug("[DEBUG] MultiPointForceMode.initialize() - PositionPublishModule created")
        
        logger.debug("[DEBUG] MultiPointForceMode.initialize() - Returning True")
        print("[PRINT-DEBUG] MultiPointForceMode.initialize() - Returning True", file=sys.stderr, flush=True)
        return True
    
    def register_channels(self):
        """Register channels: subscribe FORCE (multi-point), publish POSITION (SITE)"""
        # Channel registration is handled by OrcaLinkClient during JoinSession
        pass
    
    def step(self) -> bool:
        """Execute one step"""
        logger.debug("[DEBUG] MultiPointForceMode.step() - Start")
        
        # 1. Subscribe to multi-point forces and apply to SITE points
        if self.force_application_module:
            logger.debug("[DEBUG] MultiPointForceMode.step() - Calling subscribe_and_apply_site_forces()...")
            self.force_application_module.subscribe_and_apply_site_forces()
            logger.debug("[DEBUG] MultiPointForceMode.step() - subscribe_and_apply_site_forces() completed")
        
        # 2. Check flow control
        if self.orcalink_client and hasattr(self.orcalink_client, 'should_pause_this_cycle'):
            logger.debug("[DEBUG] MultiPointForceMode.step() - Checking flow control...")
            if self.orcalink_client.should_pause_this_cycle():
                logger.debug("[DEBUG] MultiPointForceMode.step() - Flow control says pause, returning False")
                return False  # Pause MuJoCo step
        
        # 3. Publish SITE positions
        if self.position_publish_module:
            logger.debug("[DEBUG] MultiPointForceMode.step() - Publishing SITE positions...")
            self.position_publish_module.publish_site_positions()
            logger.debug("[DEBUG] MultiPointForceMode.step() - publish_site_positions() completed")
        
        logger.debug("[DEBUG] MultiPointForceMode.step() - Returning True")
        return True
    
    def shutdown(self):
        """Shutdown the mode"""
        self.force_application_module = None
        self.position_publish_module = None

