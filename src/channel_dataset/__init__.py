"""Cross-medium physical-layer channel modeling framework."""

from channel_dataset.core.models import ChannelModel
from channel_dataset.core.realization import ChannelRealization, RealizationMetadata

__version__ = "0.1.0"

__all__ = ["ChannelModel", "ChannelRealization", "RealizationMetadata", "__version__"]
