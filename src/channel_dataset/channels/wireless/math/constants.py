"""Physical constants used by wireless channel calculations."""

# Exact by definition in the International System of Units (SI).
SPEED_OF_LIGHT_MPS: float = 299_792_458.0

# Exact SI value after the 2019 redefinition.
BOLTZMANN_CONSTANT_J_PER_K: float = 1.380_649e-23

__all__ = ["BOLTZMANN_CONSTANT_J_PER_K", "SPEED_OF_LIGHT_MPS"]
