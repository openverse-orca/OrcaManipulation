"""
Force-Position coupling mode implementation
"""

from typing import Optional, Dict, Any
from .base import ICouplingMode
from ..modules.force_application import ForceApplicationModule
from ..modules.position_publish import PositionPublishModule


class ForcePositionMode(ICouplingMode):
    """Force-Position coupling mode
    
    This mode implements the traditional force-position coupling:
    - MuJoCo sends forces to SPH
    - SPH sends rigid body positions to MuJoCo
    """
    
    def __init__(self):
        self.force_application_module: Optional[ForceApplicationModule] = None
        self.position_publish_module: Optional[PositionPublishModule] = None
        self.env = None
        self.orcalink_client = None
        self.loop = None
    
    def initialize(self, config: Dict[str, Any], env, orcalink_client, loop) -> bool:
        """Initialize the mode"""
        self.env = env
        self.orcalink_client = orcalink_client
        self.loop = loop  # 直接使用传入的 loop，不再从 orcalink_client 获取
        
        # Initialize modules
        self.force_application_module = ForceApplicationModule(env, orcalink_client, self.loop)
        self.position_publish_module = PositionPublishModule(
            env, orcalink_client, self.loop, config.get('rigid_bodies', []))
        
        return True
    
    def register_channels(self):
        """Register channels: subscribe FORCE, publish POSITION"""
        # Channel registration is handled by OrcaLinkClient during JoinSession
        # This method can be used for additional setup if needed
        pass
    
    def step(self) -> bool:
        """Execute one step"""
        # 1. Subscribe to forces and apply to MuJoCo
        if self.force_application_module:
            self.force_application_module.subscribe_and_apply_forces()
        
        # 2. Check flow control
        if self.orcalink_client and hasattr(self.orcalink_client, 'should_pause_this_cycle'):
            if self.orcalink_client.should_pause_this_cycle():
                return False  # Pause MuJoCo step
        
        # 3. Publish positions
        if self.position_publish_module:
            self.position_publish_module.publish_positions()
        
        return True
    
    def shutdown(self):
        """Shutdown the mode"""
        self.force_application_module = None
        self.position_publish_module = None

