from __future__ import annotations

import copy
import json
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import mujoco

from mjlab.terrains.config import ROUGH_TERRAINS_CFG
from mjlab.terrains.terrain_generator import TerrainGenerator
from mjlab.utils.spec import export_spec

ROOT = Path("/home/shc/workspace/rl/mujoco_playground")
GO2_XML_DIR = ROOT / "mujoco_playground/_src/locomotion/go2/xmls"
GO2_MESH_DIR = (
  ROOT / "mujoco_playground/external_deps/mujoco_menagerie/unitree_go2/assets"
)
BASE_XML = GO2_XML_DIR / "scene_mjx_fullcollisions_complex_terrain.xml"
STAIR_XML = GO2_XML_DIR / "scene_mjx_fullcollisions_stair_terrain.xml"
OUT = Path("/tmp/mjlab_playground_complex_candidate_validation")
TERRAIN_OUT = OUT / "terrain"
CANDIDATE_OUT = OUT / "playground"
CANDIDATE_XML = CANDIDATE_OUT / "scene_mjx_fullcollisions_complex_terrain.xml"


def _fresh_dir(path: Path) -> None:
  if path.exists():
    shutil.rmtree(path)
  path.mkdir(parents=True, exist_ok=True)


def generate_terrain() -> None:
  _fresh_dir(TERRAIN_OUT)
  spec = mujoco.MjSpec()
  TerrainGenerator(ROUGH_TERRAINS_CFG).compile(spec)
  export_spec(spec, TERRAIN_OUT)


def compose_playground_xml() -> None:
  _fresh_dir(CANDIDATE_OUT)
  shutil.copytree(GO2_XML_DIR / "assets", CANDIDATE_OUT / "assets")
  for stale_hf_texture in (CANDIDATE_OUT / "assets").glob("hf_texture_*.png"):
    stale_hf_texture.unlink()
  for support_xml in GO2_XML_DIR.glob("*.xml"):
    if support_xml.name != BASE_XML.name:
      shutil.copy2(support_xml, CANDIDATE_OUT / support_xml.name)
  for robot_xml_name in ("go2_mjx_fullcollisions.xml", "go2_mjx.xml", "go2_mjx_feetonly.xml"):
    robot_xml = CANDIDATE_OUT / robot_xml_name
    if not robot_xml.exists():
      continue
    robot_tree = ET.parse(robot_xml)
    compiler = robot_tree.getroot().find("compiler")
    if compiler is not None:
      compiler.set("meshdir", str(GO2_MESH_DIR))
      ET.indent(robot_tree.getroot(), space="  ")
      robot_xml.write_text(ET.tostring(robot_tree.getroot(), encoding="unicode"))

  base_tree = ET.parse(BASE_XML)
  terrain_tree = ET.parse(TERRAIN_OUT / "scene.xml")
  base_root = base_tree.getroot()
  terrain_root = terrain_tree.getroot()
  base_root.set("model", "go2 fullcollisions complex terrain scene")

  base_asset = base_root.find("asset")
  terrain_asset = terrain_root.find("asset")
  if base_asset is None or terrain_asset is None:
    raise RuntimeError("missing asset section")

  # Drop old generated terrain assets from the base wrapper; keep skybox and
  # non-terrain assets from the Go2 scene.
  for elem in list(base_asset):
    name = elem.get("name", "")
    if elem.tag == "hfield" or name.startswith(("hf_texture_", "hf_material_")):
      base_asset.remove(elem)

  for elem in list(terrain_asset):
    if elem.tag in {"texture", "material", "hfield"}:
      base_asset.append(copy.deepcopy(elem))

  base_worldbody = base_root.find("worldbody")
  terrain_worldbody = terrain_root.find("worldbody")
  if base_worldbody is None or terrain_worldbody is None:
    raise RuntimeError("missing worldbody")

  old_body = base_worldbody.find("./body[@name='terrain']")
  new_body = terrain_worldbody.find("./body[@name='terrain']")
  if old_body is None or new_body is None:
    raise RuntimeError("missing terrain body")

  # Match existing Playground terrain contact policy: robot geoms generate
  # contacts against terrain, but terrain pieces do not collide with each other.
  new_body = copy.deepcopy(new_body)
  for geom in new_body.iter("geom"):
    geom.set("contype", "0")
    geom.set("conaffinity", "1")
    geom.set("condim", "3")
    geom.set("priority", geom.get("priority", "2"))

  children = list(base_worldbody)
  insert_at = children.index(old_body)
  base_worldbody.remove(old_body)
  base_worldbody.insert(insert_at, new_body)

  ET.indent(base_root, space="  ")
  CANDIDATE_XML.write_text(
    "<?xml version='1.0' encoding='utf-8'?>\n"
    + ET.tostring(base_root, encoding="unicode")
  )

  for file in (TERRAIN_OUT / "assets").glob("*"):
    shutil.copy2(file, CANDIDATE_OUT / "assets" / file.name)


def load_model(path: Path) -> mujoco.MjModel:
  return mujoco.MjModel.from_xml_path(str(path))


def _geom_name(model: mujoco.MjModel, i: int) -> str:
  return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""


def _sensor_name(model: mujoco.MjModel, i: int) -> str:
  return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i) or ""


def model_stats(path: Path) -> dict:
  model = load_model(path)
  geom_counts = Counter(mujoco.mjtGeom(model.geom_type[i]).name for i in range(model.ngeom))
  terrain_indices = [
    i
    for i in range(model.ngeom)
    if _geom_name(model, i).startswith("terrain_")
    or _geom_name(model, i).startswith("pit_")
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
  robot_mask_counts = Counter(
    (
      int(model.geom_contype[i]),
      int(model.geom_conaffinity[i]),
      int(model.geom_condim[i]),
      int(model.geom_priority[i]),
    )
    for i in range(model.ngeom)
    if i not in terrain_indices and _geom_name(model, i) != "floor"
  )
  box_full_z = [
    float(model.geom_size[i, 2] * 2)
    for i in terrain_indices
    if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX
  ]
  hfield_base = [
    float(model.hfield_size[i, 3])
    for i in range(model.nhfield)
  ]
  terrain_types = Counter(
    mujoco.mjtGeom(model.geom_type[i]).name for i in terrain_indices
  )
  sensor_names = [_sensor_name(model, i) for i in range(model.nsensor)]

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
    "terrain_mask_counts": {str(k): v for k, v in sorted(terrain_mask_counts.items())},
    "robot_mask_counts": {str(k): v for k, v in sorted(robot_mask_counts.items())},
    "terrain_box_full_z_min": min(box_full_z) if box_full_z else None,
    "terrain_box_full_z_under_0p10": sum(z < 0.1 - 1e-9 for z in box_full_z),
    "hfield_base_min": min(hfield_base) if hfield_base else None,
    "hfield_base_under_0p10": sum(z < 0.1 - 1e-9 for z in hfield_base),
    "sensor_names": sensor_names,
  }


def referenced_files(path: Path) -> dict:
  root = ET.parse(path).getroot()
  refs = sorted({elem.get("file") for elem in root.iter() if elem.get("file")})
  missing = [ref for ref in refs if not (path.parent / ref).exists()]
  return {"referenced": refs, "missing": missing}


def xml_counts(path: Path) -> dict:
  root = ET.parse(path).getroot()
  counts = Counter()
  terrain_names = Counter()
  for elem in root.iter():
    counts[elem.tag] += 1
    if elem.tag == "geom":
      name = elem.get("name", "")
      if name.startswith("terrain_"):
        terrain_names[name.split("_")[0]] += 1
  return {"tags": dict(sorted(counts.items())), "terrain_prefixes": dict(terrain_names)}


def main() -> None:
  generate_terrain()
  compose_playground_xml()

  current = model_stats(BASE_XML)
  stair = model_stats(STAIR_XML)
  candidate = model_stats(CANDIDATE_XML)

  sensor_diff_vs_stair = {
    "candidate_minus_stair": sorted(
      set(candidate["sensor_names"]) - set(stair["sensor_names"])
    ),
    "stair_minus_candidate": sorted(
      set(stair["sensor_names"]) - set(candidate["sensor_names"])
    ),
  }
  sensor_diff_vs_current = {
    "candidate_minus_current": sorted(
      set(candidate["sensor_names"]) - set(current["sensor_names"])
    ),
    "current_minus_candidate": sorted(
      set(current["sensor_names"]) - set(candidate["sensor_names"])
    ),
  }

  output = {
    "output_dir": str(OUT),
    "terrain_xml": str(TERRAIN_OUT / "scene.xml"),
    "candidate_xml": str(CANDIDATE_XML),
    "current": {k: v for k, v in current.items() if k != "sensor_names"},
    "stair": {k: v for k, v in stair.items() if k != "sensor_names"},
    "candidate": {k: v for k, v in candidate.items() if k != "sensor_names"},
    "sensor_diff_vs_stair": sensor_diff_vs_stair,
    "sensor_diff_vs_current": sensor_diff_vs_current,
    "candidate_refs": referenced_files(CANDIDATE_XML),
    "candidate_xml_counts": xml_counts(CANDIDATE_XML),
  }
  print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
