#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Download and convert TACO Object_Models OBJ files to USD for RoboLab.

Downloads Object_Models.zip from gs://foundational-research/hoi-dataset/TACO_Dataset_Overall/,
extracts 206 OBJ meshes, and converts each to USD with physics APIs matching
the existing RoboLab asset format (PhysicsRigidBodyAPI, CollisionAPI, MassAPI,
physics_material prim, and a default gray OmniPBR material).

TACO meshes are in centimeters; this script scales them to meters (×0.01).
Mass is estimated from mesh volume × density (default 1000 kg/m³).

After running this script, register objects in the catalog:
    uv run python generate_catalog.py --objects ../../assets/objects/taco

Usage:
    python convert_taco_to_usd.py                      # full pipeline (download + convert all)
    python convert_taco_to_usd.py --skip-download      # reuse cached zip in --zip-dir
    python convert_taco_to_usd.py --ids 004 025 131    # convert specific IDs only
    python convert_taco_to_usd.py --dry-run            # list what would be converted
    python convert_taco_to_usd.py --overwrite          # re-convert already-existing USDs
    python convert_taco_to_usd.py --density 800        # wood-like objects (kg/m³)
    python convert_taco_to_usd.py --static 5.0 --dynamic 5.0 --restitution 0.1
"""

import glob
import os
import subprocess
import sys
from pathlib import Path


# ── USD environment bootstrap ────────────────────────────────────────────────
# pxr requires Isaac Sim's shared libraries on LD_LIBRARY_PATH.
# If not already set up, this block discovers them and re-execs the script.

def _bootstrap_usd_env() -> None:
    """Re-exec with Isaac Sim USD libs on LD_LIBRARY_PATH if pxr isn't importable."""
    if os.environ.get("_TACO_USD_ENV") == "1":
        return  # already bootstrapped

    try:
        from pxr import Usd  # noqa: F401
        return  # already importable
    except ImportError:
        pass

    repo_root = Path(__file__).resolve().parents[3]
    venv_site = repo_root / ".venv/lib/python3.11/site-packages"

    # Isaac Sim USD shared libraries
    usd_bin_dirs = sorted(glob.glob(str(venv_site / "isaacsim/extscache/omni.usd.libs-*/bin")))
    if not usd_bin_dirs:
        sys.exit("ERROR: Cannot find Isaac Sim USD libraries. Is the .venv set up correctly?")
    usd_bin = usd_bin_dirs[-1]
    usd_pkg = str(Path(usd_bin).parent)

    # libpython — prefer Isaac Sim's bundled copy (omni/kernel/plugins/) to avoid ABI mismatch
    py_lib_candidates = sorted(glob.glob(str(venv_site / "omni/kernel/plugins")))
    if not py_lib_candidates:
        # Fall back to uv-managed Python lib dir (resolve through venv symlink)
        real_py = Path(sys.executable).resolve()
        py_lib_candidates = [str(real_py.parent.parent / "lib")]
    py_lib = py_lib_candidates[0]

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{usd_bin}:{py_lib}:{env.get('LD_LIBRARY_PATH', '')}"
    env["PYTHONPATH"] = f"{usd_pkg}:{env.get('PYTHONPATH', '')}"
    env["_TACO_USD_ENV"] = "1"
    result = subprocess.run([sys.executable] + sys.argv, env=env)
    sys.exit(result.returncode)


_bootstrap_usd_env()

# ── Imports (pxr is now available) ───────────────────────────────────────────

import argparse
import json
import tempfile
import zipfile
from typing import Optional

import numpy as np
import trimesh
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt


# ── Constants ─────────────────────────────────────────────────────────────────

GCS_ZIP_URL = "gs://foundational-research/hoi-dataset/TACO_Dataset_Overall/Object_Models.zip"
OBJ_SUBDIR = "object_models_released"
CM_TO_M = 0.01  # TACO OBJ files use centimeters; Isaac Sim uses meters

REPO_ROOT = Path(__file__).resolve().parents[3]
TACO_DIR = REPO_ROOT / "assets/objects/taco"


# ── USD construction ──────────────────────────────────────────────────────────

def _set_xform_ops(prim: Usd.Prim) -> None:
    """Add standard translate/rotateXYZ/scale xform ops matching existing objects."""
    xform = UsdGeom.Xform(prim)
    xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))
    xform.AddRotateXYZOp().Set(Gf.Vec3f(0, 0, 0))
    xform.AddScaleOp().Set(Gf.Vec3f(1, 1, 1))


def _apply_rigid_body(prim: Usd.Prim, mass: float) -> None:
    """Apply PhysicsRigidBodyAPI and PhysicsMassAPI to root prim."""
    rb = UsdPhysics.RigidBodyAPI.Apply(prim)
    rb.GetRigidBodyEnabledAttr().Set(True)
    rb.GetKinematicEnabledAttr().Set(False)
    rb.GetStartsAsleepAttr().Set(False)
    rb.GetVelocityAttr().Set(Gf.Vec3f(0, 0, 0))
    rb.GetAngularVelocityAttr().Set(Gf.Vec3f(0, 0, 0))
    prim.SetCustomDataByKey("physxRigidBody:enableCCD", True)

    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.GetMassAttr().Set(float(mass))
    mass_api.GetDensityAttr().Set(0.0)
    mass_api.GetCenterOfMassAttr().Set(Gf.Vec3f(-float("inf"), -float("inf"), -float("inf")))
    mass_api.GetDiagonalInertiaAttr().Set(Gf.Vec3f(0, 0, 0))
    mass_api.GetPrincipalAxesAttr().Set(Gf.Quatf(0, 0, 0, 0))


def _apply_collision(mesh_prim: Usd.Prim) -> None:
    """Apply PhysicsCollisionAPI + convex decomposition to mesh prim."""
    col_api = UsdPhysics.CollisionAPI.Apply(mesh_prim)
    col_api.GetCollisionEnabledAttr().Set(True)
    mesh_col = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
    mesh_col.GetApproximationAttr().Set("convexDecomposition")

    # convex decomposition params (matching banana.usd)
    mesh_prim.CreateAttribute("physxConvexDecompositionCollision:errorPercentage", Sdf.ValueTypeNames.Float).Set(1.0)
    mesh_prim.CreateAttribute("physxConvexDecompositionCollision:maxConvexHulls", Sdf.ValueTypeNames.Int).Set(256)
    mesh_prim.CreateAttribute("physxConvexDecompositionCollision:shrinkWrap", Sdf.ValueTypeNames.Bool).Set(True)

    mass_api = UsdPhysics.MassAPI.Apply(mesh_prim)
    mass_api.GetMassAttr().Set(0.0)
    mass_api.GetDensityAttr().Set(0.0)
    mass_api.GetCenterOfMassAttr().Set(Gf.Vec3f(-float("inf"), -float("inf"), -float("inf")))
    mass_api.GetDiagonalInertiaAttr().Set(Gf.Vec3f(0, 0, 0))
    mass_api.GetPrincipalAxesAttr().Set(Gf.Quatf(0, 0, 0, 0))


def _add_physics_material(
    stage: Usd.Stage,
    parent_path: str,
    static_friction: float,
    dynamic_friction: float,
    restitution: float,
) -> None:
    """Add physics_material prim matching existing object format."""
    mat_path = f"{parent_path}/physics_material"
    mat_prim = stage.DefinePrim(mat_path, "Material")
    phys_mat = UsdPhysics.MaterialAPI.Apply(mat_prim)
    phys_mat.GetStaticFrictionAttr().Set(static_friction)
    phys_mat.GetDynamicFrictionAttr().Set(dynamic_friction)
    phys_mat.GetRestitutionAttr().Set(restitution)
    phys_mat.GetDensityAttr().Set(0.0)


def _add_default_material(stage: Usd.Stage, parent_path: str) -> None:
    """Add a simple gray OmniPBR material (no texture — TACO meshes have none)."""
    looks_path = f"{parent_path}/Looks"
    stage.DefinePrim(looks_path, "Scope")
    mat_path = f"{looks_path}/Material"
    mat_prim = stage.DefinePrim(mat_path, "Material")

    shader_path = f"{mat_path}/OmniPBR"
    shader_prim = stage.DefinePrim(shader_path, "Shader")
    shader = UsdShade.Shader(shader_prim)
    shader.SetShaderId("OmniPBR")
    shader_prim.CreateAttribute("info:implementationSource", Sdf.ValueTypeNames.Token).Set("sourceAsset")
    shader_prim.CreateAttribute("info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("OmniPBR.mdl"))
    shader_prim.CreateAttribute("info:mdl:sourceAsset:subIdentifier", Sdf.ValueTypeNames.Token).Set("OmniPBR")
    shader_prim.CreateAttribute("inputs:diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.5, 0.5, 0.5)
    )
    shader_prim.CreateAttribute("inputs:metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader_prim.CreateAttribute("inputs:roughness", Sdf.ValueTypeNames.Float).Set(0.5)


def _add_semantic_attributes(
    prim: Usd.Prim, name: str, obj_class: str, description: str
) -> None:
    """Add string metadata attributes and semantic labels matching existing objects."""
    prim.CreateAttribute("name", Sdf.ValueTypeNames.String).Set(name)
    prim.CreateAttribute("class", Sdf.ValueTypeNames.String).Set(obj_class)
    prim.CreateAttribute("dataset", Sdf.ValueTypeNames.String).Set("taco")
    prim.CreateAttribute("description", Sdf.ValueTypeNames.String).Set(description)

    # Semantic schema (name + class labels)
    sem0 = prim.CreateAttribute("semantic:Semantics0:params:semanticData", Sdf.ValueTypeNames.String)
    sem0.Set(name)
    prim.CreateAttribute("semantic:Semantics0:params:semanticType", Sdf.ValueTypeNames.String).Set("name")
    prim.CreateAttribute("semantic:Semantics1:params:semanticData", Sdf.ValueTypeNames.String).Set(obj_class)
    prim.CreateAttribute("semantic:Semantics1:params:semanticType", Sdf.ValueTypeNames.String).Set("class")


def _estimate_mass(mesh_m: trimesh.Trimesh, density_kg_m3: float) -> float:
    """Estimate mass from a meter-scale mesh volume."""
    if mesh_m.is_watertight:
        volume_m3 = abs(mesh_m.volume)
    else:
        # Bounding-box fallback with 0.4 fill factor for non-watertight scans
        volume_m3 = abs(mesh_m.bounding_box.volume) * 0.4

    mass = volume_m3 * density_kg_m3
    # Clamp to a sane range for tool-sized objects
    return float(max(0.05, min(5.0, mass)))


def obj_to_usd(
    obj_path: Path,
    usd_path: Path,
    density: float = 1000.0,
    static_friction: float = 2.0,
    dynamic_friction: float = 2.0,
    restitution: float = 0.1,
) -> bool:
    """
    Convert a single TACO OBJ file to a RoboLab-compatible USD.

    Args:
        obj_path: Path to the .obj file (centimeter-scale, no MTL/textures).
        usd_path: Output path for the .usd file.
        density: Object density in kg/m³ for mass estimation.
        static_friction: Static friction coefficient.
        dynamic_friction: Dynamic friction coefficient.
        restitution: Restitution (bounciness) coefficient.

    Returns:
        True on success, False on failure.
    """
    obj_id = obj_path.stem.replace("_cm", "")  # "004_cm" → "004"
    prim_name = f"taco_{obj_id}"

    # Load mesh
    try:
        mesh = trimesh.load(str(obj_path), force="mesh", process=False)
    except Exception as e:
        print(f"  ERROR loading mesh: {e}")
        return False

    if not isinstance(mesh, trimesh.Trimesh):
        print(f"  ERROR: expected Trimesh, got {type(mesh)}")
        return False

    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        print(f"  ERROR: empty mesh")
        return False

    # Scale cm → m (single load, single scale)
    mesh.apply_scale(CM_TO_M)
    mass = _estimate_mass(mesh, density)
    dims = mesh.bounding_box.extents.tolist()  # [w, d, h] in meters

    # Build USD stage
    usd_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(usd_path))
    stage.SetMetadata("defaultPrim", prim_name)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    # Root Xform prim
    root = stage.DefinePrim(f"/{prim_name}", "Xform")
    _set_xform_ops(root)
    _apply_rigid_body(root, mass)
    _add_semantic_attributes(
        root,
        name=prim_name,
        obj_class="object",
        description=f"TACO dataset object {obj_id} (tool-action-object interaction dataset). "
                    f"Dims: {dims[0]:.3f}x{dims[1]:.3f}x{dims[2]:.3f} m.",
    )
    # physxRigidBody:enableCCD is a PhysX extension attribute (not standard USD)
    root.CreateAttribute("physxRigidBody:enableCCD", Sdf.ValueTypeNames.Bool).Set(True)

    # Mesh prim
    mesh_prim_path = f"/{prim_name}/{prim_name}_Mesh"
    mesh_prim = stage.DefinePrim(mesh_prim_path, "Mesh")
    usd_mesh = UsdGeom.Mesh(mesh_prim)

    # Geometry: scale mesh already applied
    vertices = mesh.vertices.astype(float).tolist()
    usd_mesh.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*v) for v in vertices]))

    face_counts = [3] * len(mesh.faces)
    usd_mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray(face_counts))

    face_indices = mesh.faces.flatten().tolist()
    usd_mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(face_indices))

    # Normals
    if mesh.vertex_normals is not None and len(mesh.vertex_normals) == len(mesh.vertices):
        normals = mesh.vertex_normals.astype(float).tolist()
        usd_mesh.GetNormalsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*n) for n in normals]))
        UsdGeom.Primvar(mesh_prim.GetAttribute("normals")).SetInterpolation("vertex")

    usd_mesh.GetOrientationAttr().Set(UsdGeom.Tokens.rightHanded)
    usd_mesh.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)  # triangulated OBJ, no subdivision

    _apply_collision(mesh_prim)

    # Materials
    mat_scope_path = f"/{prim_name}/Looks"
    _add_default_material(stage, f"/{prim_name}")
    _add_physics_material(stage, f"/{prim_name}", static_friction, dynamic_friction, restitution)

    # Bind visual material to mesh prim (default purpose)
    mat_path = f"{mat_scope_path}/Material"
    mat_prim = stage.GetPrimAtPath(mat_path)
    mat = UsdShade.Material(mat_prim)
    binding_api = UsdShade.MaterialBindingAPI.Apply(mesh_prim)
    binding_api.Bind(mat)

    # Bind physics material to mesh prim (physics purpose) — required by get_friction()
    phys_mat_prim = stage.GetPrimAtPath(f"/{prim_name}/physics_material")
    phys_mat = UsdShade.Material(phys_mat_prim)
    binding_api.Bind(phys_mat, UsdShade.Tokens.strongerThanDescendants, "physics")

    stage.Save()
    return True


# ── Download and extraction ───────────────────────────────────────────────────

def download_zip(zip_dir: Path, skip_if_exists: bool = True) -> Path:
    """Download Object_Models.zip from GCS using gsutil."""
    zip_path = zip_dir / "Object_Models.zip"
    if skip_if_exists and zip_path.exists():
        print(f"Reusing cached zip: {zip_path}")
        return zip_path

    zip_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {GCS_ZIP_URL} → {zip_path} ...")
    result = subprocess.run(
        ["gsutil", "cp", GCS_ZIP_URL, str(zip_path)],
        check=True,
    )
    print("Download complete.")
    return zip_path


def extract_objs(zip_path: Path, extract_dir: Path) -> list[Path]:
    """Extract OBJ files from the zip; return sorted list of extracted paths."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        obj_names = [n for n in z.namelist() if n.endswith(".obj")]
        print(f"Extracting {len(obj_names)} OBJ files ...")
        z.extractall(extract_dir, members=obj_names)

    obj_paths = sorted(extract_dir.glob(f"{OBJ_SUBDIR}/*.obj"))
    print(f"Extracted {len(obj_paths)} OBJs to {extract_dir / OBJ_SUBDIR}")
    return obj_paths


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert TACO Object_Models OBJ files to RoboLab USD assets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--zip-dir",
        type=Path,
        default=Path("/tmp/taco_download"),
        help="Directory for the downloaded zip and extracted OBJs (default: /tmp/taco_download)",
    )
    parser.add_argument(
        "--obj-dir",
        type=Path,
        default=None,
        help="Use already-extracted OBJ directory instead of downloading",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=TACO_DIR,
        help=f"Output directory for USD files (default: {TACO_DIR})",
    )
    parser.add_argument(
        "--ids",
        nargs="+",
        default=None,
        metavar="ID",
        help="Convert only specific object IDs (e.g. --ids 004 025 131)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download; use cached zip in --zip-dir",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-convert objects even if USD already exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be converted without converting",
    )
    parser.add_argument(
        "--density",
        type=float,
        default=1000.0,
        help="Object density in kg/m³ for mass estimation (default: 1000 = plastic)",
    )
    parser.add_argument(
        "--static",
        type=float,
        default=2.0,
        dest="static_friction",
        help="Static friction coefficient (default: 2.0)",
    )
    parser.add_argument(
        "--dynamic",
        type=float,
        default=2.0,
        dest="dynamic_friction",
        help="Dynamic friction coefficient (default: 2.0)",
    )
    parser.add_argument(
        "--restitution",
        type=float,
        default=0.1,
        help="Restitution coefficient (default: 0.1)",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress per-file output",
    )
    args = parser.parse_args()

    # Determine OBJ source
    if args.obj_dir is not None:
        obj_paths = sorted(args.obj_dir.glob("*.obj"))
        if not obj_paths:
            print(f"ERROR: No OBJ files found in {args.obj_dir}")
            return 1
    else:
        zip_path = download_zip(args.zip_dir, skip_if_exists=args.skip_download)
        extract_dir = args.zip_dir
        obj_paths = extract_objs(zip_path, extract_dir)

    # Filter by requested IDs
    if args.ids:
        id_set = set(args.ids)
        obj_paths = [p for p in obj_paths if p.stem.replace("_cm", "") in id_set]
        if not obj_paths:
            print(f"ERROR: No OBJ files matched IDs: {args.ids}")
            return 1

    # Plan conversions
    to_convert = []
    for obj_path in obj_paths:
        obj_id = obj_path.stem.replace("_cm", "")
        usd_path = args.out_dir / f"taco_{obj_id}.usd"
        if usd_path.exists() and not args.overwrite:
            if not args.quiet:
                print(f"  Skip (exists): taco_{obj_id}.usd")
            continue
        to_convert.append((obj_path, usd_path))

    print(f"\n{len(to_convert)} objects to convert (out of {len(obj_paths)} total)")
    if args.dry_run:
        for obj_path, usd_path in to_convert:
            print(f"  {obj_path.name} → {usd_path.name}")
        return 0

    # Convert
    args.out_dir.mkdir(parents=True, exist_ok=True)
    success_count = 0
    error_count = 0

    for i, (obj_path, usd_path) in enumerate(to_convert, 1):
        if not args.quiet:
            print(f"[{i}/{len(to_convert)}] {obj_path.name} → {usd_path.name}")
        ok = obj_to_usd(
            obj_path,
            usd_path,
            density=args.density,
            static_friction=args.static_friction,
            dynamic_friction=args.dynamic_friction,
            restitution=args.restitution,
        )
        if ok:
            success_count += 1
            if not args.quiet:
                print(f"  OK")
        else:
            error_count += 1
            print(f"  FAILED: {obj_path.name}")

    print(f"\n{'='*50}")
    print(f"Converted: {success_count}")
    print(f"Errors:    {error_count}")
    if success_count > 0:
        print(f"\nNext step — register in catalog:")
        print(f"  cd {REPO_ROOT}/assets/objects/_utils")
        print(f"  uv run python generate_catalog.py --objects {args.out_dir}")

    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
