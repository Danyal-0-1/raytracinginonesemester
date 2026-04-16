"""
gen_grass_billboards.py — procedural grass billboard OBJ generator

Places N upright quads (2 triangles each) randomly over a rectangular
region of the XY ground plane (Z-up). Each quad represents a clump of grass;
all visual detail comes from the diffuse + alpha textures assigned in the
scene JSON — the geometry is just an upright rectangle.

Each billboard:
  - Random position inside the region
  - Random azimuth rotation (so clumps face different directions)
  - Slight random tilt ±15° for variety
  - Random scale variation ±20%
  - UVs mapped 0–1 across the full quad face

Coordinate system: Z-up. Ground is the XY plane (Z=0).

Usage:
    # Symmetric patch:
    python scripts/gen_grass_billboards.py --count 400 --patch 5.0

    # Explicit bounds (mid-distance zone only):
    python scripts/gen_grass_billboards.py --count 200 \\
        --xmin -2.5 --xmax 2.5 --ymin -0.5 --ymax 2.0

    Output: assets/meshes/grass_billboards.obj  (override with --output)
"""

import math
import random
import argparse
import os


def normalize(v):
    l = math.sqrt(sum(x * x for x in v))
    return tuple(x / l for x in v) if l > 1e-10 else v


def cross(a, b):
    return (a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0])


def generate_billboards(count, xmin, xmax, ymin, ymax, seed,
                        base_width=0.30, base_height=0.50):
    """
    Generate `count` billboard quads distributed uniformly in [xmin,xmax]×[ymin,ymax].

    Returns (verts, normals, uvs, faces) as flat lists suitable for OBJ output.
    All indices in `faces` are 1-based OBJ style.
    """
    rng = random.Random(seed)

    verts   = []
    normals = []
    uvs     = []
    faces   = []

    for _ in range(count):
        px = rng.uniform(xmin, xmax)
        py = rng.uniform(ymin, ymax)

        # Random azimuth: which horizontal direction this billboard faces
        theta = rng.uniform(0, 2 * math.pi)
        # Width axis is perpendicular to the facing direction in XY
        wx = -math.sin(theta)
        wy =  math.cos(theta)

        # Random tilt ±5° around the width axis — keep quads nearly vertical
        # so base stays visually planted on the ground
        tilt   = rng.uniform(-0.087, 0.087)       # radians (≈ ±5°)
        tip_dx = math.cos(theta) * math.sin(tilt)
        tip_dy = math.sin(theta) * math.sin(tilt)
        tip_dz = math.cos(tilt)

        # Scale variation ±20%
        scale = rng.uniform(0.8, 1.2)
        w  = base_width  * scale
        h  = base_height * scale
        hw = w / 2.0

        # 4 vertices of the billboard quad
        bl = (px - wx*hw,            py - wy*hw,            0.0)
        br = (px + wx*hw,            py + wy*hw,            0.0)
        tr = (px + wx*hw + tip_dx*h, py + wy*hw + tip_dy*h, tip_dz*h)
        tl = (px - wx*hw + tip_dx*h, py - wy*hw + tip_dy*h, tip_dz*h)

        e1 = (br[0]-bl[0], br[1]-bl[1], br[2]-bl[2])
        e2 = (tl[0]-bl[0], tl[1]-bl[1], tl[2]-bl[2])
        n  = normalize(cross(e1, e2))

        vi = len(verts)   + 1
        ni = len(normals) + 1
        ti = len(uvs)     + 1

        verts   += [bl, br, tr, tl]
        normals += [n]
        uvs     += [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]

        faces.append((vi+0, vi+1, vi+2, ni, ti+0, ti+1, ti+2))
        faces.append((vi+0, vi+2, vi+3, ni, ti+0, ti+2, ti+3))

    return verts, normals, uvs, faces


def write_obj(path, verts, normals, uvs, faces, header=""):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write(f"# {header}\n")
        f.write("# Coordinate system: Z-up. Ground at Z=0.\n")
        f.write("o grass_billboards\n\n")
        f.write("# Vertices\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        f.write("\n# Normals\n")
        for n in normals:
            f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
        f.write("\n# UVs\n")
        for uv in uvs:
            f.write(f"vt {uv[0]:.4f} {uv[1]:.4f}\n")
        f.write("\n# Faces  (v/vt/vn)\n")
        for v0, v1, v2, ni, t0, t1, t2 in faces:
            f.write(f"f {v0}/{t0}/{ni} {v1}/{t1}/{ni} {v2}/{t2}/{ni}\n")


def main():
    parser = argparse.ArgumentParser(description="Generate grass billboard OBJ")
    parser.add_argument("--count",  type=int,   default=250,
                        help="Number of billboard quads")
    # Convenient symmetric-patch shorthand
    parser.add_argument("--patch",  type=float, default=None,
                        help="Side length of a square patch centred at origin "
                             "(overrides --xmin/xmax/ymin/ymax)")
    # Explicit rectangular bounds
    parser.add_argument("--xmin",   type=float, default=-2.5)
    parser.add_argument("--xmax",   type=float, default= 2.5)
    parser.add_argument("--ymin",   type=float, default=-2.5)
    parser.add_argument("--ymax",   type=float, default= 2.5)
    parser.add_argument("--width",  type=float, default=0.30,
                        help="Base billboard width (m)")
    parser.add_argument("--height", type=float, default=0.50,
                        help="Base billboard height (m)")
    parser.add_argument("--seed",   type=int,   default=42)
    parser.add_argument("--output", type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             "..", "assets", "meshes",
                                             "grass_billboards.obj"))
    args = parser.parse_args()

    if args.patch is not None:
        h = args.patch / 2.0
        xmin, xmax, ymin, ymax = -h, h, -h, h
    else:
        xmin, xmax, ymin, ymax = args.xmin, args.xmax, args.ymin, args.ymax

    area = (xmax - xmin) * (ymax - ymin)
    print(f"Generating {args.count} billboards over "
          f"[{xmin},{xmax}]×[{ymin},{ymax}] ({area:.1f} m², seed={args.seed})...")

    verts, normals, uvs, faces = generate_billboards(
        args.count, xmin, xmax, ymin, ymax, args.seed,
        args.width, args.height)

    out_path = os.path.normpath(args.output)
    header = (f"{args.count} billboards, bounds [{xmin},{xmax}]×[{ymin},{ymax}], "
              f"{args.width}m×{args.height}m, seed {args.seed}")
    write_obj(out_path, verts, normals, uvs, faces, header)
    print(f"Written {len(verts)} vertices, {len(faces)} triangles → {out_path}")


if __name__ == "__main__":
    main()
    