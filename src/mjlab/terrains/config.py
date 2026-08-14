"""Terrain configuration presets and named terrain sets.

Each terrain type has a preset function with sensible defaults. Override any
parameter at the call site. Named configs (ROUGH, STAIRS, ALL) compose presets.

To add a new terrain, define a function decorated with @terrain_preset::

    @terrain_preset
    def my_new_terrain(**overrides: Any) -> terrain_gen.SomeTerrainCfg:
        defaults: dict[str, Any] = dict(...)
        defaults.update(overrides)
        return terrain_gen.SomeTerrainCfg(**defaults)

It will be auto-included in ALL_TERRAIN_PRESETS and ALL_TERRAINS_CFG.
"""

from collections.abc import Callable
from typing import Any, TypeVar

import mjlab.terrains as terrain_gen
from mjlab.terrains.terrain_entity import TerrainEntity, TerrainEntityCfg
from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeneratorCfg

# Preset registry.

ALL_TERRAIN_PRESETS: dict[str, Callable[..., SubTerrainCfg]] = {}

_F = TypeVar("_F", bound=Callable[..., SubTerrainCfg])


def terrain_preset(fn: _F) -> _F:
  """Register a terrain preset into ALL_TERRAIN_PRESETS."""
  ALL_TERRAIN_PRESETS[fn.__name__] = fn
  return fn


# Terrain presets.


@terrain_preset
def flat(**overrides: Any) -> terrain_gen.BoxFlatTerrainCfg:
  return terrain_gen.BoxFlatTerrainCfg(**overrides)


@terrain_preset
def pyramid_stairs(**overrides: Any) -> terrain_gen.BoxPyramidStairsTerrainCfg:
  defaults: dict[str, Any] = dict(
    step_height_range=(0.0, 0.2),
    step_width=0.3,
    platform_width=3.0,
    border_width=1.0,
  )
  defaults.update(overrides)
  return terrain_gen.BoxPyramidStairsTerrainCfg(**defaults)


@terrain_preset
def pyramid_stairs_inv(
  **overrides: Any,
) -> terrain_gen.BoxInvertedPyramidStairsTerrainCfg:
  defaults: dict[str, Any] = dict(
    step_height_range=(0.0, 0.2),
    step_width=0.3,
    platform_width=3.0,
    border_width=1.0,
  )
  defaults.update(overrides)
  return terrain_gen.BoxInvertedPyramidStairsTerrainCfg(**defaults)


@terrain_preset
def hf_pyramid_slope(
  **overrides: Any,
) -> terrain_gen.HfPyramidSlopedTerrainCfg:
  defaults: dict[str, Any] = dict(
    slope_range=(0.0, 0.7),
    platform_width=2.0,
    border_width=0.25,
  )
  defaults.update(overrides)
  return terrain_gen.HfPyramidSlopedTerrainCfg(**defaults)


@terrain_preset
def hf_pyramid_slope_inv(
  **overrides: Any,
) -> terrain_gen.HfPyramidSlopedTerrainCfg:
  defaults: dict[str, Any] = dict(
    slope_range=(0.0, 0.7),
    platform_width=2.0,
    border_width=0.25,
    inverted=True,
  )
  defaults.update(overrides)
  return terrain_gen.HfPyramidSlopedTerrainCfg(**defaults)


@terrain_preset
def random_rough(
  **overrides: Any,
) -> terrain_gen.HfRandomUniformTerrainCfg:
  defaults: dict[str, Any] = dict(
    noise_range=(0.02, 0.10),
    noise_step=0.02,
    border_width=0.25,
  )
  defaults.update(overrides)
  return terrain_gen.HfRandomUniformTerrainCfg(**defaults)


@terrain_preset
def wave_terrain(**overrides: Any) -> terrain_gen.HfWaveTerrainCfg:
  defaults: dict[str, Any] = dict(
    amplitude_range=(0.0, 0.2),
    num_waves=4,
    border_width=0.25,
  )
  defaults.update(overrides)
  return terrain_gen.HfWaveTerrainCfg(**defaults)


@terrain_preset
def discrete_obstacles(
  **overrides: Any,
) -> terrain_gen.HfDiscreteObstaclesTerrainCfg:
  defaults: dict[str, Any] = dict(
    obstacle_width_range=(0.3, 1.0),
    obstacle_height_range=(0.05, 0.3),
    num_obstacles=40,
    border_width=0.25,
  )
  defaults.update(overrides)
  return terrain_gen.HfDiscreteObstaclesTerrainCfg(**defaults)


@terrain_preset
def perlin_noise(
  **overrides: Any,
) -> terrain_gen.HfPerlinNoiseTerrainCfg:
  defaults: dict[str, Any] = dict(
    height_range=(0.0, 1.0),
    octaves=4,
    persistence=0.3,
    lacunarity=2.0,
    scale=10.0,
    horizontal_scale=0.1,
    border_width=0.50,
  )
  defaults.update(overrides)
  return terrain_gen.HfPerlinNoiseTerrainCfg(**defaults)


@terrain_preset
def box_random_grid(
  **overrides: Any,
) -> terrain_gen.BoxRandomGridTerrainCfg:
  defaults: dict[str, Any] = dict(
    grid_width=0.4,
    grid_height_range=(0.0, 0.3),
    platform_width=1.0,
  )
  defaults.update(overrides)
  return terrain_gen.BoxRandomGridTerrainCfg(**defaults)


@terrain_preset
def random_spread_boxes(
  **overrides: Any,
) -> terrain_gen.BoxRandomSpreadTerrainCfg:
  defaults: dict[str, Any] = dict(
    num_boxes=80,
    box_width_range=(0.1, 1.0),
    box_length_range=(0.1, 2.0),
    box_height_range=(0.05, 0.3),
    platform_width=1.0,
    border_width=0.25,
  )
  defaults.update(overrides)
  return terrain_gen.BoxRandomSpreadTerrainCfg(**defaults)


@terrain_preset
def open_stairs(
  **overrides: Any,
) -> terrain_gen.BoxOpenStairsTerrainCfg:
  defaults: dict[str, Any] = dict(
    step_height_range=(0.1, 0.2),
    step_width_range=(0.4, 0.8),
    platform_width=1.0,
    border_width=0.25,
    inverted=False,
  )
  defaults.update(overrides)
  return terrain_gen.BoxOpenStairsTerrainCfg(**defaults)


@terrain_preset
def random_stairs(
  **overrides: Any,
) -> terrain_gen.BoxRandomStairsTerrainCfg:
  defaults: dict[str, Any] = dict(
    step_width=0.8,
    step_height_range=(0.1, 0.3),
    platform_width=1.0,
    border_width=0.25,
  )
  defaults.update(overrides)
  return terrain_gen.BoxRandomStairsTerrainCfg(**defaults)


@terrain_preset
def stepping_stones(
  **overrides: Any,
) -> terrain_gen.BoxSteppingStonesTerrainCfg:
  defaults: dict[str, Any] = dict(
    stone_size_range=(0.4, 0.8),
    stone_distance_range=(0.2, 0.5),
    stone_height=0.2,
    stone_height_variation=0.1,
    stone_size_variation=0.2,
    displacement_range=0.1,
    floor_depth=2.0,
    platform_width=1.0,
    border_width=0.25,
  )
  defaults.update(overrides)
  return terrain_gen.BoxSteppingStonesTerrainCfg(**defaults)


@terrain_preset
def narrow_beams(
  **overrides: Any,
) -> terrain_gen.BoxNarrowBeamsTerrainCfg:
  defaults: dict[str, Any] = dict(
    num_beams=12,
    beam_width_range=(0.2, 0.8),
    beam_height=0.2,
    spacing=0.8,
    platform_width=1.0,
    border_width=0.25,
    floor_depth=2.0,
  )
  defaults.update(overrides)
  return terrain_gen.BoxNarrowBeamsTerrainCfg(**defaults)


@terrain_preset
def nested_rings(
  **overrides: Any,
) -> terrain_gen.BoxNestedRingsTerrainCfg:
  defaults: dict[str, Any] = dict(
    num_rings=8,
    ring_width_range=(0.3, 0.6),
    gap_range=(0.1, 0.4),
    height_range=(0.1, 0.4),
    platform_width=1.0,
    border_width=0.25,
    floor_depth=2.0,
  )
  defaults.update(overrides)
  return terrain_gen.BoxNestedRingsTerrainCfg(**defaults)


@terrain_preset
def tilted_grid(
  **overrides: Any,
) -> terrain_gen.BoxTiltedGridTerrainCfg:
  defaults: dict[str, Any] = dict(
    grid_width=1.0,
    tilt_range_deg=20.0,
    height_range=0.3,
    platform_width=1.0,
    border_width=0.25,
    floor_depth=2.0,
  )
  defaults.update(overrides)
  return terrain_gen.BoxTiltedGridTerrainCfg(**defaults)


# Named terrain sets.

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
  seed=0,
  size=(8.0, 8.0),
  border_width=20.0,
  num_rows=20,
  num_cols=20,
  sub_terrains={
    "flat": flat(proportion=0.1),
    "pyramid_stairs": pyramid_stairs(
      proportion=0.25,
      step_height_range=(0.0, 0.18),
      step_width_range=(0.25, 0.35),
      platform_width=2.0,
    ),
    "pyramid_stairs_inv": pyramid_stairs_inv(
      proportion=0.25,
      step_height_range=(0.0, 0.18),
      step_width_range=(0.25, 0.35),
      platform_width=2.0,
    ),
    "hf_pyramid_slope": hf_pyramid_slope(proportion=0.1, slope_range=(0.0, 1.0)),
    "hf_pyramid_slope_inv": hf_pyramid_slope_inv(
      proportion=0.1, slope_range=(0.0, 1.0)
    ),
    "random_rough": random_rough(proportion=0.1),
    "wave_terrain": wave_terrain(proportion=0.1),
  },
  add_lights=True,
)

STAIRS_TERRAINS_CFG = TerrainGeneratorCfg(
  size=(8.0, 8.0),
  border_width=20.0,
  num_rows=10,
  num_cols=4,
  curriculum=True,
  sub_terrains={
    "flat": flat(proportion=0.25),
    "easy_stairs": pyramid_stairs(
      proportion=0.35,
      step_height_range=(0.02, 0.05),
      step_width=0.40,
    ),
    "moderate_stairs": pyramid_stairs(
      proportion=0.25,
      step_height_range=(0.05, 0.08),
      step_width=0.35,
      platform_width=2.5,
      border_width=0.8,
    ),
    "challenging_stairs": pyramid_stairs(
      proportion=0.15,
      step_height_range=(0.08, 0.10),
      step_width=0.30,
      platform_width=2.0,
      border_width=0.5,
    ),
  },
  add_lights=True,
)

# Go2 deployment-oriented terrain. Goals: (1) robust outdoor locomotion and
# (2) reliable traversal of real staircases with risers around 20 cm.
#
# Stair geometry is split into two contiguous bands:
#  - Ramp stairs use 8-16 cm risers and generous 28-35 cm treads. This removes
#    nearly-flat stair patches while retaining an easier skill-formation band.
#  - Deployment stairs use 16-22 cm risers and tight 24-28 cm treads. The band
#    includes the 20 cm deployment target while retaining overlap with the ramp
#    boundary at 16 cm.
# Both bands have equal normal and inverted proportions for descending-first
# and ascending-first starts. curriculum=False samples every difficulty from
# the beginning. random_rough scales with difficulty so its recorded difficulty
# corresponds to actual geometric severity.

_STAIRS_RAMP = dict(
  step_height_range=(0.08, 0.16),
  step_width_range=(0.28, 0.35),
  platform_width=2.0,
)
_STAIRS_DEPLOY = dict(
  step_height_range=(0.16, 0.22),
  step_width_range=(0.24, 0.28),
  platform_width=2.0,
)

ROUGH_TERRAINS_V2_CFG = TerrainGeneratorCfg(
  seed=0,
  size=(8.0, 8.0),
  border_width=20.0,
  num_rows=20,
  num_cols=20,
  sub_terrains={
    "flat": flat(proportion=0.18),
    "pyramid_stairs_ramp": pyramid_stairs(proportion=0.07, **_STAIRS_RAMP),
    "pyramid_stairs_deploy": pyramid_stairs(proportion=0.15, **_STAIRS_DEPLOY),
    "pyramid_stairs_inv_ramp": pyramid_stairs_inv(proportion=0.07, **_STAIRS_RAMP),
    "pyramid_stairs_inv_deploy": pyramid_stairs_inv(proportion=0.15, **_STAIRS_DEPLOY),
    "hf_pyramid_slope": hf_pyramid_slope(proportion=0.08, slope_range=(0.0, 1.0)),
    "hf_pyramid_slope_inv": hf_pyramid_slope_inv(
      proportion=0.08, slope_range=(0.0, 1.0)
    ),
    "random_rough": random_rough(proportion=0.12, scale_with_difficulty=True),
    "wave_terrain": wave_terrain(proportion=0.10),
  },
  add_lights=True,
)

# V3 allocates 42% to stairs, 16% to box obstacles, and 36% to slopes, rough
# ground, discrete obstacles, and Perlin ground. Explicit flat and wave terrain
# together occupy 6%. The proportions sum to 1.0; seed 0 determines the exact
# spatial realization of the 20x20 random grid.
ROUGH_TERRAINS_V3_CFG = TerrainGeneratorCfg(
  seed=0,
  size=(8.0, 8.0),
  border_width=20.0,
  num_rows=20,
  num_cols=20,
  sub_terrains={
    "flat": flat(proportion=0.04),
    "pyramid_stairs_ramp": pyramid_stairs(proportion=0.06, **_STAIRS_RAMP),
    "pyramid_stairs_deploy": pyramid_stairs(proportion=0.15, **_STAIRS_DEPLOY),
    "pyramid_stairs_inv_ramp": pyramid_stairs_inv(proportion=0.06, **_STAIRS_RAMP),
    "pyramid_stairs_inv_deploy": pyramid_stairs_inv(proportion=0.15, **_STAIRS_DEPLOY),
    "hf_pyramid_slope": hf_pyramid_slope(proportion=0.05, slope_range=(0.0, 0.7)),
    "hf_pyramid_slope_inv": hf_pyramid_slope_inv(
      proportion=0.05, slope_range=(0.0, 0.7)
    ),
    "random_rough": random_rough(proportion=0.08, scale_with_difficulty=True),
    "wave_terrain": wave_terrain(proportion=0.02),
    "discrete_obstacles": discrete_obstacles(proportion=0.08),
    # min_box_height: MJX box collision needs >= 0.1 m thickness and the
    # playground exporter rejects thinner terrain (588 offending boxes at
    # 1.5 cm on the first v3 attempt).
    "random_spread_boxes": random_spread_boxes(proportion=0.10, min_box_height=0.10),
    # resolution 0.1 (not the 0.05 preset default): at 0.05 an 8 m tile
    # becomes a 160x160 heightfield whose 5 cm cells put more than MuJoCo's
    # 50-contacts-per-geom-pair hfield cap under a single robot geom, so
    # contacts are silently DROPPED - run 18B logged 19.9M overflow warnings
    # across its whole training. 0.1 matches every other hfield type (80x80)
    # and, since effective feature scale = scale * resolution/horizontal_scale,
    # also stretches features from ~0.25 m to ~1 m wavelength: rolling natural
    # ground rather than fine moguls (which random_rough already covers).
    "perlin_noise": perlin_noise(proportion=0.10, resolution=0.1),
    "box_random_grid": box_random_grid(
      proportion=0.06,
      grid_width=0.6,
      grid_height_range=(0.04, 0.12),
      platform_width=1.0,
      merge_similar_heights=True,
    ),
  },
  add_lights=True,
)

ALL_TERRAINS_CFG = TerrainGeneratorCfg(
  size=(8.0, 8.0),
  border_width=20.0,
  num_rows=10,
  num_cols=len(ALL_TERRAIN_PRESETS),
  sub_terrains={name: fn(proportion=1.0) for name, fn in ALL_TERRAIN_PRESETS.items()},
  add_lights=True,
)

if __name__ == "__main__":
  import mujoco.viewer
  import torch

  device = "cuda" if torch.cuda.is_available() else "cpu"

  terrain_cfg = TerrainEntityCfg(
    terrain_type="generator",
    terrain_generator=ROUGH_TERRAINS_CFG,
  )
  terrain = TerrainEntity(terrain_cfg, device=device)
  mujoco.viewer.launch(terrain.spec.compile())
