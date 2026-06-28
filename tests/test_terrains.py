"""Tests for terrain generation."""

import mujoco
import numpy as np
import pytest

from mjlab.terrains.config import ALL_TERRAIN_PRESETS
from mjlab.terrains.heightfield_terrains import (
  _MIN_HFIELD_BASE_THICKNESS,
  HfDiscreteObstaclesTerrainCfg,
  HfPerlinNoiseTerrainCfg,
  HfPyramidSlopedTerrainCfg,
  HfRandomUniformTerrainCfg,
  HfWaveTerrainCfg,
)
from mjlab.terrains.primitive_terrains import (
  _MIN_BORDER_HEIGHT,
  BoxFlatTerrainCfg,
  BoxInvertedPyramidStairsTerrainCfg,
  BoxPyramidStairsTerrainCfg,
  BoxSteppingStonesTerrainCfg,
)
from mjlab.terrains.terrain_generator import TerrainGenerator, TerrainGeneratorCfg

_CFG = BoxSteppingStonesTerrainCfg(
  proportion=1.0,
  size=(8.0, 8.0),
  stone_size_range=(0.2, 0.6),
  stone_distance_range=(0.05, 0.25),
  stone_height=0.2,
  stone_height_variation=0.05,
  stone_size_variation=0.05,
  displacement_range=0.1,
  floor_depth=2.0,
  platform_width=1.5,
  border_width=0.25,
)


def _generate_stones(
  cfg: BoxSteppingStonesTerrainCfg,
  difficulty: float,
  rng: np.random.Generator,
) -> list[tuple[float, float, float, float]]:
  """Generate terrain and return stone (cx, cy, half_x, half_y) tuples."""
  spec = mujoco.MjSpec()
  spec.worldbody.add_body(name="terrain")
  output = cfg.function(difficulty=difficulty, spec=spec, rng=rng)

  center = cfg.size[0] / 2
  stones = []
  for geom_info in output.geometries:
    geom = geom_info.geom
    if geom is None:
      continue
    pos, size = geom.pos, geom.size
    # Skip platform, floor, and border geoms. The platform is the geom centered
    # exactly at the patch center (its size is grid-snapped, not the configured
    # width, so it is identified by position alone).
    is_platform = np.isclose(pos[0], center) and np.isclose(pos[1], center)
    is_full_span = np.isclose(size[0], cfg.size[0] / 2) or np.isclose(
      size[1], cfg.size[1] / 2
    )
    if is_platform or is_full_span:
      continue
    stones.append((pos[0], pos[1], size[0], size[1]))
  return stones


def test_no_stone_centers_inside_platform():
  """No stone center should fall inside the platform."""
  center = _CFG.size[0] / 2
  p_half = _CFG.platform_width / 2
  p_min, p_max = center - p_half, center + p_half

  for difficulty in [0.0, 0.5, 1.0]:
    stones = _generate_stones(_CFG, difficulty, np.random.default_rng(42))
    for cx, cy, _, _ in stones:
      assert not (p_min <= cx <= p_max and p_min <= cy <= p_max), (
        f"Stone at ({cx:.3f}, {cy:.3f}) inside platform at difficulty={difficulty}"
      )


def test_stone_size_decreases_with_difficulty():
  """Average stone size should be smaller at higher difficulty."""
  sizes = {}
  for difficulty in [0.0, 1.0]:
    stones = _generate_stones(_CFG, difficulty, np.random.default_rng(42))
    sizes[difficulty] = np.mean([hx + hy for _, _, hx, hy in stones])

  assert sizes[0.0] > sizes[1.0]


@pytest.mark.parametrize(
  "cfg_cls", [BoxPyramidStairsTerrainCfg, BoxInvertedPyramidStairsTerrainCfg]
)
def test_pyramid_stairs_border_present_at_zero_difficulty(cfg_cls):
  """At difficulty 0 the step height collapses to 0, but the flat border frame
  must still be generated as solid, non-degenerate geometry (regression for the
  empty-boundary bug, issue #1033)."""
  cfg = cfg_cls(
    size=(8.0, 8.0),
    step_height_range=(0.0, 0.2),
    step_width=0.3,
    platform_width=3.0,
    border_width=1.0,
  )
  spec = mujoco.MjSpec()
  spec.worldbody.add_body(name="terrain")
  output = cfg.function(difficulty=0.0, spec=spec, rng=np.random.default_rng(0))

  # The border frame is generated first and sits below z=0 with its top flush
  # at ground level.
  border_geoms = [g.geom for g in output.geometries[:4] if g.geom is not None]
  assert len(border_geoms) == 4, "Expected four border frame boxes."
  for geom in border_geoms:
    # Each frame box must be solid, not a degenerate zero-height geom, and its
    # top must be flush with the ground plane at z=0.
    assert geom.size[2] >= _MIN_BORDER_HEIGHT / 2 - 1e-9
    assert np.isclose(geom.pos[2] + geom.size[2], 0.0, atol=1e-6)


@pytest.mark.parametrize(
  "cfg_cls,inverted",
  [
    (BoxPyramidStairsTerrainCfg, False),
    (BoxInvertedPyramidStairsTerrainCfg, True),
  ],
)
def test_pyramid_stairs_min_collision_thickness_preserves_top_surfaces(
  cfg_cls, inverted
):
  cfg = cfg_cls(
    size=(8.0, 8.0),
    step_height_range=(0.0, 0.2),
    step_width=0.3,
    platform_width=3.0,
    border_width=1.0,
  )
  difficulty = 0.01
  step_height = cfg.step_height_range[0] + difficulty * (
    cfg.step_height_range[1] - cfg.step_height_range[0]
  )
  num_steps = int(
    min(
      (cfg.size[0] - 2 * cfg.border_width - cfg.platform_width)
      / (2 * cfg.step_width),
      (cfg.size[1] - 2 * cfg.border_width - cfg.platform_width)
      / (2 * cfg.step_width),
    )
  )

  spec = mujoco.MjSpec()
  spec.worldbody.add_body(name="terrain")
  output = cfg.function(difficulty=difficulty, spec=spec, rng=np.random.default_rng(0))

  geoms = [g.geom for g in output.geometries if g.geom is not None]
  assert all(
    2 * geom.size[2] >= cfg.min_collision_thickness - 1e-9 for geom in geoms
  )

  actual_tops = sorted(geom.pos[2] + geom.size[2] for geom in geoms)
  sign = -1.0 if inverted else 1.0
  expected_tops = [0.0] * 4
  expected_tops.extend(
    sign * (k + 1) * step_height for k in range(num_steps) for _ in range(4)
  )
  expected_tops.append(sign * (num_steps + 1) * step_height)
  np.testing.assert_allclose(actual_tops, sorted(expected_tops), atol=1e-9)


@pytest.mark.parametrize(
  "cfg_cls", [BoxPyramidStairsTerrainCfg, BoxInvertedPyramidStairsTerrainCfg]
)
def test_pyramid_stairs_random_step_width_sampled_once_per_patch(cfg_cls):
  seed = 123
  step_width_range = (0.25, 0.45)
  step_width = np.random.default_rng(seed).uniform(*step_width_range)
  cfg = cfg_cls(
    size=(8.0, 8.0),
    step_height_range=(0.16, 0.2),
    step_width=0.3,
    step_width_range=step_width_range,
    platform_width=3.0,
    border_width=1.0,
  )
  terrain_size = cfg.size[0] - 2 * cfg.border_width
  num_steps = int((terrain_size - cfg.platform_width) / (2 * step_width))

  spec = mujoco.MjSpec()
  spec.worldbody.add_body(name="terrain")
  output = cfg.function(
    difficulty=0.5, spec=spec, rng=np.random.default_rng(seed)
  )

  geoms = [g.geom for g in output.geometries if g.geom is not None]
  stair_geoms = geoms[4:-1]
  assert len(stair_geoms) == 4 * num_steps

  for k in range(num_steps):
    top, bottom, right, left = stair_geoms[4 * k : 4 * k + 4]
    np.testing.assert_allclose([top.size[1], bottom.size[1]], step_width / 2)
    np.testing.assert_allclose([right.size[0], left.size[0]], step_width / 2)

  platform = geoms[-1]
  platform_width = terrain_size - 2 * num_steps * step_width
  np.testing.assert_allclose(platform.size[:2], platform_width / 2)


def test_terrain_generator_grid_metadata_records_type_center_and_difficulty():
  cfg = TerrainGeneratorCfg(
    size=(2.0, 3.0),
    num_rows=2,
    num_cols=3,
    seed=0,
    sub_terrains={"flat": BoxFlatTerrainCfg()},
  )
  generator = TerrainGenerator(cfg)
  spec = mujoco.MjSpec()
  generator.compile(spec)

  metadata = generator.grid_metadata()
  assert metadata["num_rows"] == 2
  assert metadata["num_cols"] == 3
  assert metadata["tile_size"] == [2.0, 3.0]
  assert len(metadata["tiles"]) == 6

  for tile in metadata["tiles"]:
    row = tile["row"]
    col = tile["col"]
    assert tile["type"] == "flat"
    np.testing.assert_allclose(tile["center"], generator.terrain_origins[row, col])
    assert tile["difficulty"] == generator.terrain_difficulties[row, col]
    assert 0.0 <= tile["difficulty_score"] <= 1.0
    assert tile["difficulty_level"] in ("easy", "medium", "hard", "extreme")


def test_terrain_generator_curriculum_metadata_records_row_difficulty():
  cfg = TerrainGeneratorCfg(
    curriculum=True,
    size=(2.0, 2.0),
    num_rows=3,
    num_cols=99,
    difficulty_range=(0.2, 0.8),
    seed=0,
    sub_terrains={"flat": BoxFlatTerrainCfg()},
  )
  generator = TerrainGenerator(cfg)
  spec = mujoco.MjSpec()
  generator.compile(spec)

  tiles_by_row = {tile["row"]: tile for tile in generator.grid_metadata()["tiles"]}

  assert tiles_by_row[0]["difficulty"] == 0.2
  assert tiles_by_row[0]["difficulty_score"] == 0.0
  assert tiles_by_row[0]["difficulty_level"] == "easy"
  assert tiles_by_row[1]["difficulty"] == 0.5
  assert tiles_by_row[1]["difficulty_score"] == pytest.approx(0.5)
  assert tiles_by_row[1]["difficulty_level"] == "hard"
  assert tiles_by_row[2]["difficulty"] == 0.8
  assert tiles_by_row[2]["difficulty_score"] == 1.0
  assert tiles_by_row[2]["difficulty_level"] == "extreme"


@pytest.mark.parametrize(
  "cfg",
  [
    HfPyramidSlopedTerrainCfg(size=(8.0, 8.0), slope_range=(0.0, 1.0)),
    HfRandomUniformTerrainCfg(
      size=(8.0, 8.0), noise_range=(0.02, 0.10), noise_step=0.02
    ),
    HfWaveTerrainCfg(size=(8.0, 8.0), amplitude_range=(0.0, 0.2), num_waves=4),
    HfDiscreteObstaclesTerrainCfg(
      size=(8.0, 8.0),
      obstacle_width_range=(0.3, 1.0),
      obstacle_height_range=(0.05, 0.3),
      num_obstacles=4,
    ),
    HfPerlinNoiseTerrainCfg(size=(8.0, 8.0), height_range=(0.0, 1.0)),
  ],
)
def test_hfields_have_min_base_thickness(cfg):
  spec = mujoco.MjSpec()
  spec.worldbody.add_body(name="terrain")
  output = cfg.function(difficulty=0.0, spec=spec, rng=np.random.default_rng(0))

  hfields = [g.hfield for g in output.geometries if g.hfield is not None]
  assert len(hfields) == 1
  assert hfields[0].size[3] >= _MIN_HFIELD_BASE_THICKNESS - 1e-9


@pytest.mark.parametrize("preset_name", sorted(ALL_TERRAIN_PRESETS))
@pytest.mark.parametrize("difficulty", [0.0, 1.0])
def test_preset_compiles_across_difficulty(preset_name, difficulty):
  """Every terrain preset must generate compilable MuJoCo geometry across the
  full difficulty range. Difficulty 0 is exercised explicitly because curriculum
  row 0 lands there deterministically, which previously produced degenerate
  geometry (zero-height hfields, NaN colors, missing borders)."""
  cfg = ALL_TERRAIN_PRESETS[preset_name](size=(8.0, 8.0))
  spec = mujoco.MjSpec()
  spec.worldbody.add_body(name="terrain")
  cfg.function(difficulty=difficulty, spec=spec, rng=np.random.default_rng(0))
  # Compiling validates geom/hfield sizes and rgba values (catches NaNs and
  # non-positive sizes that MuJoCo rejects).
  spec.compile()
