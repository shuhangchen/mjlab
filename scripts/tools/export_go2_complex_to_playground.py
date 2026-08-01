"""Generate an MJLab terrain and install it as Playground's Go2 complex scene.

Run from the MJLab repo:
  uv run python scripts/tools/export_go2_complex_to_playground.py --dry-run
  uv run python scripts/tools/export_go2_complex_to_playground.py

The script keeps the Playground Go2 wrapper XML intact, including the robot,
sensors, keyframes, MuJoCo options, hidden floor, and visual settings. It only
replaces the generated terrain body/assets.
"""

from __future__ import annotations

import copy
import importlib
import json
import shutil
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import mujoco
import tyro

import mjlab
from mjlab.terrains.terrain_generator import TerrainGenerator, TerrainGeneratorCfg
from mjlab.utils.spec import export_spec

DEFAULT_TERRAIN_CFG = "mjlab.terrains.config:ROUGH_TERRAINS_CFG"
GO2_XML_DIR = Path("mujoco_playground/_src/locomotion/go2/xmls")
COMPLEX_XML = "scene_mjx_fullcollisions_complex_terrain.xml"
COMPLEX_GRID_METADATA = "scene_mjx_fullcollisions_complex_terrain_grid.json"
STAIR_XML = "scene_mjx_fullcollisions_stair_terrain.xml"
GO2_MESH_DIR = Path(
  "mujoco_playground/external_deps/mujoco_menagerie/unitree_go2/assets"
)
GENERATED_TEXTURE_PREFIX = "hf_texture_"
GENERATED_MATERIAL_PREFIX = "hf_material_"


@dataclass
class ExportGo2ComplexCfg:
  playground_root: Path = Path("../mujoco_playground")
  """Path to the local mujoco_playground repo."""

  terrain_cfg: str = DEFAULT_TERRAIN_CFG
  """Import path for a TerrainGeneratorCfg object or zero-arg factory."""

  seed: int | None = None
  """Override the terrain config seed. None preserves the config value."""

  stage_dir: Path | None = None
  """Intermediate export directory. Defaults to a timestamped directory in /tmp."""

  dry_run: bool = False
  """Write a complete candidate scene under stage_dir instead of Playground."""

  backup: bool = True
  """Back up the current Playground complex scene XML and generated textures."""

  backup_dir: Path | None = None
  """Backup directory. Defaults to a timestamped directory in /tmp."""

  validate: bool = True
  """Load and validate the generated Playground XML with MuJoCo."""

  compare_sensors_to_stair: bool = True
  """Require the generated scene to expose the same sensors as the stair scene."""

  keep_stage: bool = False
  """Keep stage_dir after a successful non-dry-run update."""

  min_box_height: float = 0.10
  """Minimum full z-height, in meters, for terrain box geoms."""

  min_hfield_base: float = 0.10
  """Minimum MuJoCo hfield base thickness, in meters."""

  expected_nsensor: int = 102
  """Expected number of sensors in the final Go2 scene. Set 0 to skip."""

  expected_nmesh: int = 16
  """Expected number of Go2 mesh assets in the final model. Set 0 to skip."""

  terrain_contype: int = 0
  """MuJoCo contype assigned to generated terrain geoms."""

  terrain_conaffinity: int = 1
  """MuJoCo conaffinity assigned to generated terrain geoms."""

  terrain_condim: int = 3
  """MuJoCo condim assigned to generated terrain geoms."""

  terrain_priority: int = 0
  """MuJoCo priority assigned to generated terrain geoms.

  0 = mjlab-native contact semantics: the robot's FEET (priority 1) own the
  foot-terrain contact params (mu 0.6, foot solimp), and body-terrain
  contacts stay frictional via the equal-priority max-condim rule. The
  previous value (2) made terrain outrank the feet, silently pinning every
  foot contact at the terrain's default mu=1.0 and making foot-side
  friction randomization physics-dead (discovered 2026-08-01, playground
  run 18I). Do not raise above the feet's priority.
  """

  asset_name_prefix: str = "go2_complex"
  """Stable suffix prefix for normalized generated hfield assets."""


def _timestamp() -> str:
  return datetime.now().strftime("%Y%m%d_%H%M%S")


def _fresh_dir(path: Path) -> None:
  if path.exists():
    shutil.rmtree(path)
  path.mkdir(parents=True, exist_ok=True)


def _resolve(path: Path) -> Path:
  return path.expanduser().resolve()


def _import_object(ref: str) -> Any:
  if ":" not in ref:
    raise ValueError(f"Expected import path 'module:attribute', got {ref!r}")
  module_name, attr_name = ref.split(":", 1)
  module = importlib.import_module(module_name)
  return getattr(module, attr_name)


def _load_terrain_cfg(ref: str, seed: int | None) -> TerrainGeneratorCfg:
  obj = _import_object(ref)
  if callable(obj):
    obj = obj()
  cfg = copy.deepcopy(obj)
  if not isinstance(cfg, TerrainGeneratorCfg):
    raise TypeError(
      f"{ref!r} resolved to {type(cfg).__name__}, expected TerrainGeneratorCfg"
    )
  if seed is not None:
    cfg.seed = seed
  return cfg


def _generate_terrain_xml(
  cfg: TerrainGeneratorCfg, output_dir: Path
) -> tuple[Path, dict[str, object]]:
  _fresh_dir(output_dir)
  spec = mujoco.MjSpec()
  generator = TerrainGenerator(cfg)
  generator.compile(spec)
  export_spec(spec, output_dir)
  terrain_xml = output_dir / "scene.xml"
  if not terrain_xml.exists():
    raise FileNotFoundError(f"Terrain export did not produce {terrain_xml}")
  return terrain_xml, generator.grid_metadata()


def _write_grid_metadata(path: Path, metadata: dict[str, object]) -> None:
  path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def _normalize_generated_hfield_assets(terrain_xml: Path, prefix: str) -> None:
  tree = ET.parse(terrain_xml)
  root = tree.getroot()
  asset = root.find("asset")
  if asset is None:
    return

  texture_name_map: dict[str, str] = {}
  material_name_map: dict[str, str] = {}
  hfield_name_map: dict[str, str] = {}
  file_moves: list[tuple[Path, Path]] = []

  for idx, texture in enumerate(
    elem
    for elem in asset.findall("texture")
    if elem.get("name", "").startswith(GENERATED_TEXTURE_PREFIX)
  ):
    old_name = texture.get("name")
    old_file = texture.get("file")
    if old_name is None or old_file is None:
      continue
    new_name = f"{GENERATED_TEXTURE_PREFIX}{prefix}_{idx:03d}"
    new_file = f"assets/{new_name}.png"
    texture_name_map[old_name] = new_name
    texture.set("name", new_name)
    texture.set("file", new_file)
    file_moves.append((terrain_xml.parent / old_file, terrain_xml.parent / new_file))

  for material in asset.findall("material"):
    old_name = material.get("name")
    old_texture = material.get("texture")
    if (
      old_name is None
      or old_texture is None
      or not old_name.startswith(GENERATED_MATERIAL_PREFIX)
    ):
      continue
    new_texture = texture_name_map.get(old_texture)
    if new_texture is None:
      continue
    suffix = new_texture.removeprefix(GENERATED_TEXTURE_PREFIX)
    new_name = f"{GENERATED_MATERIAL_PREFIX}{suffix}"
    material_name_map[old_name] = new_name
    material.set("name", new_name)
    material.set("texture", new_texture)

  for idx, hfield in enumerate(asset.findall("hfield")):
    old_name = hfield.get("name")
    if old_name is None:
      continue
    new_name = f"hfield_{prefix}_{idx:03d}"
    hfield_name_map[old_name] = new_name
    hfield.set("name", new_name)

  for elem in root.iter("geom"):
    material = elem.get("material")
    if material in material_name_map:
      elem.set("material", material_name_map[material])
    hfield = elem.get("hfield")
    if hfield in hfield_name_map:
      elem.set("hfield", hfield_name_map[hfield])
    hfieldname = elem.get("hfieldname")
    if hfieldname in hfield_name_map:
      elem.set("hfieldname", hfield_name_map[hfieldname])

  for src, dst in file_moves:
    if not src.exists():
      raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst != src:
      dst.unlink()
    src.rename(dst)

  ET.indent(root, space="  ")
  terrain_xml.write_text(
    "<?xml version='1.0' encoding='utf-8'?>\n"
    + ET.tostring(root, encoding="unicode")
  )


def _copy_support_xmls(go2_xml_dir: Path, candidate_dir: Path) -> None:
  for support_xml in go2_xml_dir.glob("*.xml"):
    if support_xml.name == COMPLEX_XML:
      continue
    shutil.copy2(support_xml, candidate_dir / support_xml.name)


def _patch_robot_meshdirs(candidate_dir: Path, mesh_dir: Path) -> None:
  for robot_xml_name in (
    "go2_mjx_fullcollisions.xml",
    "go2_mjx.xml",
    "go2_mjx_feetonly.xml",
  ):
    robot_xml = candidate_dir / robot_xml_name
    if not robot_xml.exists():
      continue
    tree = ET.parse(robot_xml)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is not None:
      compiler.set("meshdir", str(mesh_dir))
      ET.indent(root, space="  ")
      robot_xml.write_text(ET.tostring(root, encoding="unicode"))


def _copy_generated_assets(terrain_export_dir: Path, assets_dir: Path) -> list[Path]:
  assets_dir.mkdir(parents=True, exist_ok=True)
  copied: list[Path] = []
  terrain_assets_dir = terrain_export_dir / "assets"
  if not terrain_assets_dir.exists():
    return copied
  for src in sorted(terrain_assets_dir.iterdir()):
    if src.is_file():
      dst = assets_dir / src.name
      shutil.copy2(src, dst)
      copied.append(dst)
  return copied


def _remove_stale_generated_textures(assets_dir: Path) -> list[Path]:
  removed: list[Path] = []
  if not assets_dir.exists():
    return removed
  for path in sorted(assets_dir.glob(f"{GENERATED_TEXTURE_PREFIX}*.png")):
    path.unlink()
    removed.append(path)
  return removed


def _backup_current_scene(target_xml: Path, assets_dir: Path, backup_dir: Path) -> Path:
  backup_dir.mkdir(parents=True, exist_ok=False)
  backup_assets_dir = backup_dir / "assets"
  backup_assets_dir.mkdir()

  shutil.copy2(target_xml, backup_dir / target_xml.name)
  target_metadata = target_xml.with_name(COMPLEX_GRID_METADATA)
  if target_metadata.exists():
    shutil.copy2(target_metadata, backup_dir / target_metadata.name)
  for texture in sorted(assets_dir.glob(f"{GENERATED_TEXTURE_PREFIX}*.png")):
    shutil.copy2(texture, backup_assets_dir / texture.name)

  manifest = sorted(str(path.relative_to(backup_dir)) for path in backup_dir.rglob("*"))
  (backup_dir / "MANIFEST.txt").write_text("\n".join(manifest) + "\n")
  return backup_dir


def _replace_generated_asset_nodes(base_asset: ET.Element, terrain_asset: ET.Element) -> None:
  for elem in list(base_asset):
    name = elem.get("name", "")
    if elem.tag == "hfield" or name.startswith(
      (GENERATED_TEXTURE_PREFIX, GENERATED_MATERIAL_PREFIX)
    ):
      base_asset.remove(elem)

  for elem in list(terrain_asset):
    if elem.tag in {"texture", "material", "hfield"}:
      base_asset.append(copy.deepcopy(elem))


def _replace_terrain_body(
  base_worldbody: ET.Element,
  terrain_worldbody: ET.Element,
  cfg: ExportGo2ComplexCfg,
) -> None:
  old_body = base_worldbody.find("./body[@name='terrain']")
  new_body = terrain_worldbody.find("./body[@name='terrain']")
  if old_body is None:
    raise RuntimeError("Base Go2 XML is missing worldbody/body[@name='terrain']")
  if new_body is None:
    raise RuntimeError("MJLab terrain XML is missing worldbody/body[@name='terrain']")

  new_body = copy.deepcopy(new_body)
  for geom in new_body.iter("geom"):
    geom.set("contype", str(cfg.terrain_contype))
    geom.set("conaffinity", str(cfg.terrain_conaffinity))
    geom.set("condim", str(cfg.terrain_condim))
    geom.set("priority", str(cfg.terrain_priority))

  children = list(base_worldbody)
  insert_at = children.index(old_body)
  base_worldbody.remove(old_body)
  base_worldbody.insert(insert_at, new_body)


def _compose_playground_xml(
  base_xml: Path,
  terrain_xml: Path,
  output_xml: Path,
  cfg: ExportGo2ComplexCfg,
) -> None:
  base_tree = ET.parse(base_xml)
  terrain_tree = ET.parse(terrain_xml)
  base_root = base_tree.getroot()
  terrain_root = terrain_tree.getroot()
  base_root.set("model", "go2 fullcollisions complex terrain scene")

  base_asset = base_root.find("asset")
  terrain_asset = terrain_root.find("asset")
  if base_asset is None:
    raise RuntimeError("Base Go2 XML is missing asset section")
  if terrain_asset is None:
    raise RuntimeError("MJLab terrain XML is missing asset section")
  _replace_generated_asset_nodes(base_asset, terrain_asset)

  base_worldbody = base_root.find("worldbody")
  terrain_worldbody = terrain_root.find("worldbody")
  if base_worldbody is None:
    raise RuntimeError("Base Go2 XML is missing worldbody")
  if terrain_worldbody is None:
    raise RuntimeError("MJLab terrain XML is missing worldbody")
  _replace_terrain_body(base_worldbody, terrain_worldbody, cfg)

  ET.indent(base_root, space="  ")
  output_xml.parent.mkdir(parents=True, exist_ok=True)
  output_xml.write_text(
    "<?xml version='1.0' encoding='utf-8'?>\n"
    + ET.tostring(base_root, encoding="unicode")
  )


def _geom_name(model: mujoco.MjModel, geom_id: int) -> str:
  return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""


def _sensor_name(model: mujoco.MjModel, sensor_id: int) -> str:
  return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_id) or ""


def _referenced_files(path: Path) -> dict[str, list[str]]:
  root = ET.parse(path).getroot()
  refs = sorted({elem.get("file") for elem in root.iter() if elem.get("file")})
  missing = [ref for ref in refs if not (path.parent / ref).exists()]
  return {"referenced": refs, "missing": missing}


def _model_stats(path: Path) -> dict[str, Any]:
  model = mujoco.MjModel.from_xml_path(str(path))
  geom_counts = Counter(
    mujoco.mjtGeom(model.geom_type[i]).name for i in range(model.ngeom)
  )
  terrain_indices = [
    i
    for i in range(model.ngeom)
    if _geom_name(model, i).startswith(("terrain_", "pit_"))
  ]
  terrain_mask_counts = Counter(
    (
      int(model.geom_contype[i]),
      int(model.geom_conaffinity[i]),
      int(model.geom_condim[i]),
      int(model.geom_priority[i]),
    )
    for i in terrain_indices
  )
  box_full_z = [
    float(model.geom_size[i, 2] * 2)
    for i in terrain_indices
    if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX
  ]
  hfield_base = [float(model.hfield_size[i, 3]) for i in range(model.nhfield)]
  terrain_types = Counter(
    mujoco.mjtGeom(model.geom_type[i]).name for i in terrain_indices
  )

  return {
    "path": str(path),
    "nbody": int(model.nbody),
    "ngeom": int(model.ngeom),
    "nhfield": int(model.nhfield),
    "nmesh": int(model.nmesh),
    "ntex": int(model.ntex),
    "nmat": int(model.nmat),
    "nsensor": int(model.nsensor),
    "geom_counts": dict(sorted(geom_counts.items())),
    "terrain_types": dict(sorted(terrain_types.items())),
    "terrain_mask_counts": {
      str(k): v for k, v in sorted(terrain_mask_counts.items())
    },
    "terrain_box_full_z_min": min(box_full_z) if box_full_z else None,
    "terrain_box_full_z_under_min": 0,
    "hfield_base_min": min(hfield_base) if hfield_base else None,
    "hfield_base_under_min": 0,
    "sensor_names": [_sensor_name(model, i) for i in range(model.nsensor)],
  }


def _validate_scene(
  scene_xml: Path,
  stair_xml: Path,
  cfg: ExportGo2ComplexCfg,
) -> dict[str, Any]:
  refs = _referenced_files(scene_xml)
  stats = _model_stats(scene_xml)
  errors: list[str] = []

  if refs["missing"]:
    errors.append(f"missing referenced files: {refs['missing']}")

  if cfg.expected_nsensor and stats["nsensor"] != cfg.expected_nsensor:
    errors.append(f"expected {cfg.expected_nsensor} sensors, got {stats['nsensor']}")
  if cfg.expected_nmesh and stats["nmesh"] != cfg.expected_nmesh:
    errors.append(f"expected {cfg.expected_nmesh} meshes, got {stats['nmesh']}")

  model = mujoco.MjModel.from_xml_path(str(scene_xml))
  terrain_indices = [
    i
    for i in range(model.ngeom)
    if _geom_name(model, i).startswith(("terrain_", "pit_"))
  ]
  if not terrain_indices:
    errors.append("no generated terrain geoms found")

  bad_masks = [
    (
      _geom_name(model, i),
      int(model.geom_contype[i]),
      int(model.geom_conaffinity[i]),
      int(model.geom_condim[i]),
      int(model.geom_priority[i]),
    )
    for i in terrain_indices
    if (
      int(model.geom_contype[i]),
      int(model.geom_conaffinity[i]),
      int(model.geom_condim[i]),
      int(model.geom_priority[i]),
    )
    != (
      cfg.terrain_contype,
      cfg.terrain_conaffinity,
      cfg.terrain_condim,
      cfg.terrain_priority,
    )
  ]
  if bad_masks:
    errors.append(f"{len(bad_masks)} terrain geoms have unexpected contact masks")

  box_full_z = [
    float(model.geom_size[i, 2] * 2)
    for i in terrain_indices
    if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX
  ]
  hfield_base = [float(model.hfield_size[i, 3]) for i in range(model.nhfield)]
  stats["terrain_box_full_z_under_min"] = sum(
    z < cfg.min_box_height - 1e-9 for z in box_full_z
  )
  stats["hfield_base_under_min"] = sum(
    z < cfg.min_hfield_base - 1e-9 for z in hfield_base
  )

  if stats["terrain_box_full_z_under_min"]:
    errors.append(
      f"{stats['terrain_box_full_z_under_min']} terrain boxes are thinner than "
      f"{cfg.min_box_height:.3f} m"
    )
  if stats["hfield_base_under_min"]:
    errors.append(
      f"{stats['hfield_base_under_min']} hfields have base thinner than "
      f"{cfg.min_hfield_base:.3f} m"
    )

  sensor_diff = None
  if cfg.compare_sensors_to_stair:
    stair_stats = _model_stats(stair_xml)
    sensor_diff = {
      "generated_minus_stair": sorted(
        set(stats["sensor_names"]) - set(stair_stats["sensor_names"])
      ),
      "stair_minus_generated": sorted(
        set(stair_stats["sensor_names"]) - set(stats["sensor_names"])
      ),
    }
    if sensor_diff["generated_minus_stair"] or sensor_diff["stair_minus_generated"]:
      errors.append(f"sensor set differs from stair scene: {sensor_diff}")

  summary = {
    "stats": {k: v for k, v in stats.items() if k != "sensor_names"},
    "refs": refs,
    "sensor_diff_vs_stair": sensor_diff,
    "errors": errors,
  }
  if errors:
    raise RuntimeError(json.dumps(summary, indent=2, sort_keys=True))
  return summary


def export_go2_complex_to_playground(cfg: ExportGo2ComplexCfg) -> dict[str, Any]:
  playground_root = _resolve(cfg.playground_root)
  go2_xml_dir = playground_root / GO2_XML_DIR
  target_xml = go2_xml_dir / COMPLEX_XML
  target_metadata = go2_xml_dir / COMPLEX_GRID_METADATA
  stair_xml = go2_xml_dir / STAIR_XML
  assets_dir = go2_xml_dir / "assets"
  mesh_dir = playground_root / GO2_MESH_DIR

  for required in (go2_xml_dir, target_xml, stair_xml, assets_dir, mesh_dir):
    if not required.exists():
      raise FileNotFoundError(required)

  stage_dir = _resolve(
    cfg.stage_dir
    if cfg.stage_dir is not None
    else Path(tempfile.gettempdir()) / f"mjlab_go2_complex_export_{_timestamp()}"
  )
  terrain_dir = stage_dir / "terrain"
  terrain_xml, grid_metadata = _generate_terrain_xml(
    _load_terrain_cfg(cfg.terrain_cfg, cfg.seed), terrain_dir
  )
  _normalize_generated_hfield_assets(terrain_xml, cfg.asset_name_prefix)

  candidate_dir = stage_dir / "playground"
  candidate_assets_dir = candidate_dir / "assets"
  candidate_xml = candidate_dir / COMPLEX_XML
  candidate_metadata = candidate_dir / COMPLEX_GRID_METADATA
  _fresh_dir(candidate_dir)
  shutil.copytree(assets_dir, candidate_assets_dir)
  _remove_stale_generated_textures(candidate_assets_dir)
  _copy_support_xmls(go2_xml_dir, candidate_dir)
  _patch_robot_meshdirs(candidate_dir, mesh_dir)
  _compose_playground_xml(target_xml, terrain_xml, candidate_xml, cfg)
  _write_grid_metadata(candidate_metadata, grid_metadata)
  copied_assets = _copy_generated_assets(terrain_dir, candidate_assets_dir)

  validation = None
  if cfg.validate:
    validation_stair_xml = candidate_dir / STAIR_XML
    validation = _validate_scene(candidate_xml, validation_stair_xml, cfg)

  backup_dir = None
  applied_asset_count = 0
  output_xml = candidate_xml
  output_metadata = candidate_metadata
  if not cfg.dry_run:
    if cfg.backup:
      backup_dir = _resolve(
        cfg.backup_dir
        if cfg.backup_dir is not None
        else Path(tempfile.gettempdir()) / f"go2_complex_scene_backup_{_timestamp()}"
      )
      _backup_current_scene(target_xml, assets_dir, backup_dir)

    _remove_stale_generated_textures(assets_dir)
    shutil.copy2(candidate_xml, target_xml)
    shutil.copy2(candidate_metadata, target_metadata)
    applied_asset_count = len(_copy_generated_assets(terrain_dir, assets_dir))
    output_xml = target_xml
    output_metadata = target_metadata

    if cfg.validate:
      validation_stair_xml = stair_xml
      validation = _validate_scene(output_xml, validation_stair_xml, cfg)

  stage_dir_result = str(stage_dir)
  terrain_xml_result = str(terrain_xml)

  if not cfg.dry_run and not cfg.keep_stage and cfg.stage_dir is None:
    shutil.rmtree(stage_dir)

  return {
    "playground_root": str(playground_root),
    "target_xml": str(target_xml),
    "output_xml": str(output_xml),
    "output_grid_metadata": str(output_metadata),
    "candidate_xml": str(candidate_xml) if candidate_xml.exists() else None,
    "candidate_grid_metadata": (
      str(candidate_metadata) if candidate_metadata.exists() else None
    ),
    "stage_dir": stage_dir_result if stage_dir.exists() else None,
    "terrain_xml": terrain_xml_result if terrain_xml.exists() else None,
    "copied_asset_count": len(copied_assets),
    "applied_asset_count": applied_asset_count,
    "backup_dir": str(backup_dir) if backup_dir is not None else None,
    "dry_run": cfg.dry_run,
    "validation": validation,
  }


def main() -> None:
  cfg = tyro.cli(ExportGo2ComplexCfg, config=mjlab.TYRO_FLAGS)
  result = export_go2_complex_to_playground(cfg)
  print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
