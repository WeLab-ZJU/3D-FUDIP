from basicsr.archs.fudipn_2stage import FUDIP_Net as _FUDIPn2Stage
from basicsr.archs.fudipn_3stage import FUDIP_Net as _FUDIPn3Stage
from basicsr.utils.registry import ARCH_REGISTRY



@ARCH_REGISTRY.register()
class FUDIPn2Stage(_FUDIPn2Stage):
    """BasicSR registry wrapper for the upgraded 2-stage 3D-FUDIPn network."""


@ARCH_REGISTRY.register()
class FUDIPn3Stage(_FUDIPn3Stage):
    """BasicSR registry wrapper for the upgraded 3-stage 3D-FUDIPn network."""

