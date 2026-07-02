"""
Base interface for coupling modes
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class ICouplingMode(ABC):
    """Abstract base class for coupling mode implementations"""
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any], env, orcalink_client, loop) -> bool:
        """Initialize the coupling mode with configuration
        
        Args:
            config: Configuration dictionary for this mode
            env: OrcaGym environment instance
            orcalink_client: OrcaLinkClient instance
            loop: asyncio event loop for async operations
            
        Returns:
            bool: True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    def register_channels(self):
        """Register channels with OrcaLink client
        
        Each mode registers the channels it needs (force/position)
        with the appropriate publish/subscribe flags.
        """
        pass
    
    @abstractmethod
    def step(self) -> bool:
        """Execute one step (receive data, send data, flow control check)
        
        Returns:
            bool: True if MuJoCo step is allowed, False if should pause
        """
        pass
    
    @abstractmethod
    def shutdown(self):
        """Shutdown and cleanup resources"""
        pass

