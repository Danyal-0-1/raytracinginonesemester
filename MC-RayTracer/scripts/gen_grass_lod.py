"""
gen_grass_lod.py — distance-based LOD grass geometry generator

Implements the core idea from the INRIA grass rendering paper: instead of
using one representation everywhere, the plane is split into concentric
distance zones from the camera and each zone gets a different representation:

    ┌─────────────────────────────────────┐
    │         FAR  (texture only)         │  ← just the ground plane texture
    ├─────────────────────────────────────┤
    │       MID  (billboard quads)        │  ← photo-textured upright quads
    ├─────────────────────────────────────┤
    │      CLOSE  (explicit blades)       │  ← individual tapered blade quads
    └──────────────  camera  ────────────-┘

This script samples the full plane uniformly, then assigns each candidate
position to a zone by its 2D ground-plane distance from the camera.

Outputs TWO OBJ files (loaded separately in the scene JSON so they can have
different materials/textures):
  - grass_lod_blades.obj     — explicit blade geometry (close zone)
  - grass_lod_billboards.obj — billboard quads (mid zone)

The far zone needs no extra geometry — the ground plane's diffuse texture
handles it.

Coordinate system: Z-up. Ground is XY plane (Z=0).

Usage:
    python scripts/gen_grass_lod.py

    # Custom zone radii and counts:
    python scripts/gen_grass_lod.py \\
        --cam-x 0 --cam-y -3 \\
        --r-blade 2.0 --r-billboard 4.5 \\
        --blade-count 2500 --billboard-count 300

Default camera matches all grass scene JSON files: position (0, -3, 1.5).
Default plane: 5×5m centred at origin (matches plane_5x5_uv.obj).
"""

import math
import random
import argparse
import os
import sys

# ── reuse the generation functions from the sibling scripts ──────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from gen_grass_blades      import generate_blades,      write_obj as write_blades_obj
from gen_grass_billboards  import generate_billboards,  write_obj as write_billboards_obj


def ground_dist(px, py, cam_x, cam_y):
    """2D Euclidean distance from a ground point to the camera's XY projection."""
    dx = px - cam_x
    dy = py - cam_y
    return math.sqrt(dx*dx + dy*dy)


def sample_lod_positions(total_samples, xmin, xmax, ymin, ymax,
                         cam_x, cam_y, r_blade, r_billboard, seed):
    """
    Sample `total_samples` positions uniformly over the plane and split them
    into three zones by distance from the camera:
      dist < r_blade          → close  (explicit blades)
      r_blade ≤ dist < r_bb  → mid    (billboard quads)
      dist ≥ r_billboard      → far    (discard — texture only)

    Returns (blade_positions, billboard_positions) as lists of (x, y).
    """
    rng = random.Random(seed)
    blade_pos = []
    bb_pos    = []

    for _ in range(total_samples):
        px = rng.uniform(xmin, xmax)
        py = rng.uniform(ymin, ymax)
        d  = ground_dist(px, py, cam_x, cam_y)

        if d < r_blade:
            blade_pos.append((px, py))
        elif d < r_billboard:
            bb_pos.append((px, py))
        # else: far zone — no geometry needed

    return blade_pos, bb_pos


def generate_blades_at_positions(positions, seed):
    """Run the blade generator for an explicit list of (x,y) positions."""
    import random as _rng_mod
    rng = _rng_mod.Random(seed + 1)

    base_width = 0.02
    tip_width  = 0.005
    height     = 0.15
    max_lean   = 0.03

    from gen_grass_blades import normalize, cross

    verts = []; normals = []; uvs = []; faces = []

    for (px, py) in positions:
        theta = rng.uniform(0, 2 * math.pi)
        fwd   = ( math.cos(theta),  math.sin(theta))
        perp  = (-math.sin(theta),  math.cos(theta))

        lean_scale = rng.uniform(0.5, 1.5)
        tip_lean_x = fwd[0] * max_lean * lean_scale
        tip_lean_y = fwd[1] * max_lean * lean_scale

        h  = height * rng.uniform(0.8, 1.2)
        hb = base_width / 2.0
        ht = tip_width  / 2.0

        bl = (px - perp[0]*hb,              py - perp[1]*hb,              0.0)
        br = (px + perp[0]*hb,              py + perp[1]*hb,              0.0)
        tr = (px + perp[0]*ht + tip_lean_x, py + perp[1]*ht + tip_lean_y, h)
        tl = (px - perp[0]*ht + tip_lean_x, py - perp[1]*ht + tip_lean_y, h)

        e1 = (br[0]-bl[0], br[1]-bl[1], br[2]-bl[2])
        e2 = (tl[0]-bl[0], tl[1]-bl[1], tl[2]-bl[2])
        n  = normalize(cross(e1, e2))

        vi = len(verts) + 1;  ni = len(normals) + 1;  ti = len(uvs) + 1
        verts   += [bl, br, tr, tl]
        normals += [n]
        uvs     += [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        faces.append((vi+0, vi+1, vi+2, ni, ti+0, ti+1, ti+2))
        faces.append((vi+0, vi+2, vi+3, ni, ti+0, ti+2, ti+3))

    return verts, normals, uvs, faces


def generate_billboards_at_positions(positions, seed,
                                     base_width=0.30, base_height=0.50):
    """Run the billboard generator for an explicit list of (x,y) positions."""
    import random as _rng_mod
    rng = _rng_mod.Random(seed + 2)

    from gen_grass_billboards import normalize, cross

    verts = []; normals = []; uvs = []; faces = []

    for (px, py) in positions:
        theta  = rng.uniform(0, 2 * math.pi)
        wx     = -math.sin(theta);  wy = math.cos(theta)
        tilt   = rng.uniform(-0.087, 0.087)  # ±5° — stay visually planted
        tip_dx = math.cos(theta) * math.sin(tilt)
        tip_dy = math.sin(theta) * math.sin(tilt)
        tip_dz = math.cos(tilt)
        scale  = rng.uniform(0.8, 1.2)
        w = base_width * scale;  h = base_height * scale;  hw = w / 2.0

        bl = (px - wx*hw,            py - wy*hw,            0.0)
        br = (px + wx*hw,            py + wy*hw,            0.0)
        tr = (px + wx*hw + tip_dx*h, py + wy*hw + tip_dy*h, tip_dz*h)
        tl = (px - wx*hw + tip_dx*h, py - wy*hw + tip_dy*h, tip_dz*h)

        e1 = (br[0]-bl[0], br[1]-bl[1], br[2]-bl[2])
        e2 = (tl[0]-bl[0], tl[1]-bl[1], tl[2]-bl[2])
        n  = normalize(cross(e1, e2))

        vi = len(verts) + 1;  ni = len(normals) + 1;  ti = len(uvs) + 1
        verts   += [bl, br, tr, tl]
        normals += [n]
        uvs     += [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        faces.append((vi+0, vi+1, vi+2, ni, ti+0, ti+1, ti+2))
        faces.append((vi+0, vi+2, vi+3, ni, ti+0, ti+2, ti+3))

    return verts, normals, uvs, faces


def main():
    MESH_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "assets", "meshes"))

    parser = argparse.ArgumentParser(
        description="Generate distance-LOD grass geometry (blades + billboards)")

    # Camera position (XY ground projection — Z is ignored for distance calc)
    parser.add_argument("--cam-x", type=float, default=0.0,
                        help="Camera X position on ground plane (default 0)")
    parser.add_argument("--cam-y", type=float, default=-3.0,
                        help="Camera Y position on ground plane (default -3)")

    # Zone radii (ground-plane distance from camera)
    parser.add_argument("--r-blade",     type=float, default=3.5,
                        help="Max distance for explicit blade zone (m)")
    parser.add_argument("--r-billboard", type=float, default=5.2,
                        help="Max distance for billboard zone (m). Beyond this = texture only.")

    # Density: total samples drawn; zone areas determine actual counts naturally
    parser.add_argument("--blade-density",     type=float, default=600.0,
                        help="Target blades per m² in the close zone")
    parser.add_argument("--billboard-density", type=float, default=30.0,
                        help="Target billboards per m² in the mid zone")

    # Plane bounds
    parser.add_argument("--xmin", type=float, default=-2.5)
    parser.add_argument("--xmax", type=float, default= 2.5)
    parser.add_argument("--ymin", type=float, default=-2.5)
    parser.add_argument("--ymax", type=float, default= 2.5)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--out-blades",     type=str,
                        default=os.path.join(MESH_DIR, "grass_lod_blades.obj"))
    parser.add_argument("--out-billboards", type=str,
                        default=os.path.join(MESH_DIR, "grass_lod_billboards.obj"))

    args = parser.parse_args()

    cam_x, cam_y     = args.cam_x, args.cam_y
    r_blade          = args.r_blade
    r_billboard      = args.r_billboard
    xmin, xmax       = args.xmin, args.xmax
    ymin, ymax       = args.ymin, args.ymax

    # ── Estimate zone areas (clip circles to the plane rectangle) ─────────
    # We approximate by Monte Carlo using a dense sample grid
    N_PROBE = 200_000
    rng = random.Random(args.seed - 1)
    plane_area   = (xmax - xmin) * (ymax - ymin)
    blade_hits   = 0
    bb_hits      = 0
    for _ in range(N_PROBE):
        px = rng.uniform(xmin, xmax)
        py = rng.uniform(ymin, ymax)
        d  = ground_dist(px, py, cam_x, cam_y)
        if d < r_blade:
            blade_hits += 1
        elif d < r_billboard:
            bb_hits += 1
    blade_area = plane_area * blade_hits / N_PROBE
    bb_area    = plane_area * bb_hits    / N_PROBE
    far_area   = plane_area - blade_area - bb_area

    blade_count = max(1, int(args.blade_density     * blade_area))
    bb_count    = max(1, int(args.billboard_density * bb_area))

    print(f"Camera ground projection: ({cam_x}, {cam_y})")
    print(f"Zone radii:  blade < {r_blade}m  |  billboard {r_blade}–{r_billboard}m  |  far > {r_billboard}m")
    print(f"Zone areas:  blade {blade_area:.2f} m²  |  billboard {bb_area:.2f} m²  |  far {far_area:.2f} m²")
    print(f"Blade count: {blade_count}  |  Billboard count: {bb_count}")

    # ── Sample positions per zone ─────────────────────────────────────────
    # Sample enough points to fill each zone to target density
    total_samples = (blade_count + bb_count) * 4   # oversample then filter
    blade_pos, bb_pos = sample_lod_positions(
        total_samples, xmin, xmax, ymin, ymax,
        cam_x, cam_y, r_blade, r_billboard, args.seed)

    # Trim to target counts (they may be slightly off due to randomness)
    blade_pos = blade_pos[:blade_count]
    bb_pos    = bb_pos[:bb_count]

    print(f"Actual positions:  blades {len(blade_pos)}  |  billboards {len(bb_pos)}")

    # ── Generate and write OBJs ────────────────────────────────────────────
    print("Generating blade geometry...")
    bv, bn, bu, bf = generate_blades_at_positions(blade_pos, args.seed)
    write_blades_obj(
        os.path.normpath(args.out_blades), bv, bn, bu, bf,
        header=(f"LOD blades: {len(blade_pos)} blades in close zone "
                f"(dist < {r_blade}m from camera ({cam_x},{cam_y}))"))
    print(f"  → {len(bf)} triangles written to {os.path.normpath(args.out_blades)}")

    print("Generating billboard geometry...")
    bbv, bbn, bbu, bbf = generate_billboards_at_positions(bb_pos, args.seed)
    write_billboards_obj(
        os.path.normpath(args.out_billboards), bbv, bbn, bbu, bbf,
        header=(f"LOD billboards: {len(bb_pos)} quads in mid zone "
                f"({r_blade}m–{r_billboard}m from camera ({cam_x},{cam_y}))"))
    print(f"  → {len(bbf)} triangles written to {os.path.normpath(args.out_billboards)}")

    print("\nDone. Load both OBJs in grass_scene4_lod.json:")
    print(f"  blades     → {os.path.basename(args.out_blades)}")
    print(f"  billboards → {os.path.basename(args.out_billboards)}")
    print("  far zone   → ground plane texture only (no extra geometry)")


if __name__ == "__main__":
    main()
