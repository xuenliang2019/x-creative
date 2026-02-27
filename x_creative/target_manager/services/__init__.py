"""Services for target manager."""

from x_creative.target_manager.services.target_generator import TargetGeneratorService
from x_creative.target_manager.services.target_yaml_manager import TargetYAMLManager

__all__ = [
    "TargetGeneratorService",
    "TargetYAMLManager",
]
