from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class VehicleDynamicsProfile:
    vehicle_model_hash: int | None = None

    wheelbase_eff_m: float = 2.6
    steer_max_rad: float = 0.55
    steer_rate_max_radps: float = 2.0

    accel_max_mps2: float = 2.5
    decel_comfort_mps2: float = 3.0
    decel_hard_mps2: float = 6.0
    jerk_max_mps3: float = 8.0

    a_lat_comfort_mps2: float = 3.0
    a_lat_hard_mps2: float = 5.0

    rain_lat_multiplier: float = 0.75
    caution_lat_multiplier: float = 0.75
    safety_margin: float = 0.80


def generic_gta_car_conservative() -> VehicleDynamicsProfile:
    return VehicleDynamicsProfile()


def build_vehicle_profile(name: str) -> VehicleDynamicsProfile:
    if name != "generic_gta_car_conservative":
        raise ValueError(f"Unsupported Atlas-HA vehicle profile {name!r}.")
    return generic_gta_car_conservative()
