"""
gen_grass_blades.py — procedural grass blade OBJ generator

Generates N tapered quad blades (2 triangles each) randomly distributed
over a rectangular region of the XY ground plane (Z-up).

Each blade:
  - Random position inside the region
  - Random azimuth (which horizontal direction it faces/leans)
  - Slight random forward lean at the tip
  - Random height variation ±20%
  - Tapered: base wider than tip

Coordinate system: Z-up. Ground is the XY plane (Z=0).

Usage:
    # Symmetric patch (e.g. 5×5m centred at origin):
    python scripts/gen_grass_blades.py --count 7500 --patch 5.0

    # Explicit bounds (e.g. only the near half of the plane):
    python scripts/gen_grass_blades.py --count 2000 \\
        --xmin -2.5 --xmax 2.5 --ymin -2.5 --ymax -0.5

    Output: assets/meshes/grass_blades.obj  (override with --output)
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


def generate_blades(count, xmin, xmax, ymin, ymax, seed,
                    base_width=0.02, tip_width=0.005,
                    height=0.15, max_lean=0.03):
    """
    Generate `count` grass blades distributed uniformly in [xmin,xmax]×[ymin,ymax].

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

        # Random azimuth: direction the blade leans toward
        theta = rng.uniform(0, 2 * math.pi)
        fwd  = ( math.cos(theta),  math.sin(theta))  # lean direction in XY
        perp = (-math.sin(theta),  math.cos(theta))  # width axis (perpendicular)

        # Tip lean: random 0.5–1.5× max lean
        lean_scale = rng.uniform(0.5, 1.5)
        tip_lean_x = fwd[0] * max_lean * lean_scale
        tip_lean_y = fwd[1] * max_lean * lean_scale

        # Height variation ±20%
        h  = height * rng.uniform(0.8, 1.2)
        hb = base_width / 2.0
        ht = tip_width  / 2.0

        # 4 vertices of the blade quad
        bl = (px - perp[0]*hb,            py - perp[1]*hb,            0.0)  # base-left
        br = (px + perp[0]*hb,            py + perp[1]*hb,            0.0)  # base-right
        tr = (px + perp[0]*ht + tip_lean_x, py + perp[1]*ht + tip_lean_y, h)  # tip-right
        tl = (px - perp[0]*ht + tip_lean_x, py - perp[1]*ht + tip_lean_y, h)  # tip-left

        # Blade normal: average of the two triangle face normals
        e1 = (br[0]-bl[0], br[1]-bl[1], br[2]-bl[2])
        e2 = (tl[0]-bl[0], tl[1]-bl[1], tl[2]-bl[2])
        n  = normalize(cross(e1, e2))

        vi = len(verts)   + 1   # 1-based OBJ vertex index
        ni = len(normals) + 1
        ti = len(uvs)     + 1

        verts   += [bl, br, tr, tl]
        normals += [n]
        uvs     += [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]

        # Triangle 1: bl, br, tr  |  Triangle 2: bl, tr, tl
        faces.append((vi+0, vi+1, vi+2, ni, ti+0, ti+1, ti+2))
        faces.append((vi+0, vi+2, vi+3, ni, ti+0, ti+2, ti+3))

    return verts, normals, uvs, faces


def write_obj(path, verts, normals, uvs, faces, header=""):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write(f"# {header}\n")
        f.write("# Coordinate system: Z-up. Ground at Z=0.\n")
        f.write("o grass_blades\n\n")
        f.write("# Vertices\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        f.write("\n# Normals\n")
        for n in normals:
            f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
        f.write("\n# UV coordinates\n")
        for uv in uvs:
            f.write(f"vt {uv[0]:.4f} {uv[1]:.4f}\n")
        f.write("\n# Faces  (v/vt/vn)\n")
        for v0, v1, v2, ni, t0, t1, t2 in faces:
            f.write(f"f {v0}/{t0}/{ni} {v1}/{t1}/{ni} {v2}/{t2}/{ni}\n")


def main():
    parser = argparse.ArgumentParser(description="Generate grass blade OBJ")
    parser.add_argument("--count",  type=int,   default=15000,
                        help="Number of blades to generate")
    # Convenient symmetric-patch shorthand
    parser.add_argument("--patch",  type=float, default=None,
                        help="Side length of a square patch centred at origin "
                             "(overrides --xmin/xmax/ymin/ymax)")
    # Explicit rectangular bounds
    parser.add_argument("--xmin",   type=float, default=-2.5)
    parser.add_argument("--xmax",   type=float, default= 2.5)
    parser.add_argument("--ymin",   type=float, default=-2.5)
    parser.add_argument("--ymax",   type=float, default= 2.5)
    parser.add_argument("--seed",   type=int,   default=42)
    parser.add_argument("--output", type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             "..", "assets", "meshes",
                                             "grass_blades.obj"))
    args = parser.parse_args()

    if args.patch is not None:
        h = args.patch / 2.0
        xmin, xmax, ymin, ymax = -h, h, -h, h
    else:
        xmin, xmax, ymin, ymax = args.xmin, args.xmax, args.ymin, args.ymax

    area = (xmax - xmin) * (ymax - ymin)
    print(f"Generating {args.count} grass blades over "
          f"[{xmin},{xmax}]×[{ymin},{ymax}] ({area:.1f} m², seed={args.seed})...")

    verts, normals, uvs, faces = generate_blades(
        args.count, xmin, xmax, ymin, ymax, args.seed)

    out_path = os.path.normpath(args.output)
    header = (f"{args.count} blades, bounds [{xmin},{xmax}]×[{ymin},{ymax}], "
              f"seed {args.seed}")
    write_obj(out_path, verts, normals, uvs, faces, header)
    print(f"Written {len(verts)} vertices, {len(faces)} triangles → {out_path}")


if __name__ == "__main__":
    main()
