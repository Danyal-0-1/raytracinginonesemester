"""
gen_grass_billboard_texture.py — generate diffuse + alpha textures for grass billboards

Creates a matched texture pair using the EXACT same blade parameters as
gen_grass_blades.py, so billboard quads look like a side-view of explicit
blade geometry when viewed from a distance.

The texture represents what you'd see looking at a cluster of blades from
the side: tapered blade shapes in green, everything else transparent.

Outputs (to assets/grass/):
  grass_billboard_diffuse.png  — RGBA green blades, transparent background
  grass_billboard_alpha.png    — greyscale alpha mask (white = opaque)

Usage:
    python scripts/gen_grass_billboard_texture.py
    python scripts/gen_grass_billboard_texture.py --blades 12 --res 512 --seed 7
"""

import math
import random
import argparse
import os

try:
    from PIL import Image, ImageDraw, ImageFilter
except ImportError:
    raise SystemExit("Pillow is required: pip install Pillow")


# ── Blade parameter defaults — derived from gen_grass_blades.py geometry ─────
# Actual blade: base_width=0.02m, tip_width=0.005m, height=0.15m (±20%), lean=0.03m (×0.5–1.5)
# Billboard quad: width=0.30m, height=0.50m  (matches gen_grass_billboards.py defaults)
# Fractions = blade_dim / quad_dim so texture looks identical to explicit geometry
BASE_WIDTH_FRAC = 0.067   # 0.02m / 0.30m
TIP_WIDTH_FRAC  = 0.017   # 0.005m / 0.30m
HEIGHT_FRAC_MIN = 0.24    # 0.15m * 0.8 / 0.50m
HEIGHT_FRAC_MAX = 0.36    # 0.15m * 1.2 / 0.50m
MAX_LEAN_FRAC   = 0.15    # 0.03m * 1.5 / 0.30m

# ── Blade colour — matches the explicit blade material albedo [0.25, 0.55, 0.15]
BLADE_COLOR_BASE = (45,  110, 30)   # darker at root (shadowed, near ground)
BLADE_COLOR_TIP  = (90,  175, 55)   # lighter at tip (sunlit, open air)


def lerp_color(c0, c1, t):
    return tuple(int(c0[i] + (c1[i] - c0[i]) * t) for i in range(3))


def blade_polygon(cx, tex_w, tex_h, rng):
    """
    Return a list of (x, y) pixel vertices for one tapered blade quad,
    plus the colour gradient tuple (base_col, tip_col).

    cx      — horizontal centre position in [0, 1]
    tex_w/h — texture dimensions in pixels
    """
    bw   = BASE_WIDTH_FRAC * tex_w * rng.uniform(0.8, 1.2)
    tw   = TIP_WIDTH_FRAC  * tex_w * rng.uniform(0.8, 1.2)
    h    = tex_h * rng.uniform(HEIGHT_FRAC_MIN, HEIGHT_FRAC_MAX)
    lean = MAX_LEAN_FRAC * tex_w * rng.uniform(-1.0, 1.0)

    # Blade sits on the bottom edge; grows upward (y decreases in image space)
    base_y = tex_h - 1
    tip_y  = int(tex_h - h)
    base_x = int(cx * tex_w)
    tip_x  = base_x + int(lean)

    # Four corners of the tapered quad
    bl = (base_x - int(bw / 2), base_y)
    br = (base_x + int(bw / 2), base_y)
    tr = (tip_x  + int(tw / 2), tip_y)
    tl = (tip_x  - int(tw / 2), tip_y)

    # Colour: slightly randomised per blade to break uniformity
    hue_shift = rng.randint(-12, 12)
    base_col = tuple(max(0, min(255, c + hue_shift)) for c in BLADE_COLOR_BASE)
    tip_col  = tuple(max(0, min(255, c + hue_shift)) for c in BLADE_COLOR_TIP)

    return [bl, br, tr, tl], base_col, tip_col


def draw_blade_gradient(draw_rgba, draw_alpha, poly, base_col, tip_col, tex_h):
    """
    Draw one blade onto the RGBA and alpha images using a vertical gradient.
    We approximate the gradient by drawing thin horizontal scanlines.
    """
    # Bounding box of the blade
    ys = [p[1] for p in poly]
    y_min, y_max = min(ys), max(ys)
    span = max(y_max - y_min, 1)

    # We draw the full polygon first on alpha, then overdraw the RGBA gradient
    draw_alpha.polygon(poly, fill=255)

    for y in range(y_min, y_max + 1):
        t = 1.0 - (y - y_min) / span   # 0 at base, 1 at tip
        col = lerp_color(base_col, tip_col, t)
        # Clip the scanline to the polygon by drawing a 1-pixel-tall polygon
        # (simpler: just paint the full polygon multiple times — fast enough)
        draw_rgba.polygon(poly, fill=col + (255,))
        # We overpaint per-scanline by clipping y-range.
        # Since ImageDraw doesn't support scanline clipping natively,
        # we instead draw the full blade in base colour then overlay a gradient
        # via horizontal stripes clipped by the polygon on a separate pass.

    # Proper approach: draw full polygon in base_col, then for each scanline
    # draw a horizontal line at the correct gradient colour clipped to polygon.
    # PIL doesn't clip lines to polygons, so we use a mask.
    pass


def generate_texture(num_blades, tex_w, tex_h, seed, soft_edge=True):
    """
    Generate the RGBA diffuse image and greyscale alpha image.
    Blades are distributed across the texture width with slight overlap.
    """
    rng = random.Random(seed)

    # RGBA image: green blades on transparent background
    rgba = Image.new("RGBA", (tex_w, tex_h), (0, 0, 0, 0))
    # Greyscale alpha mask
    alpha_img = Image.new("L", (tex_w, tex_h), 0)

    # Build blade list sorted back-to-front (left-to-right, slight depth sort)
    # so overlapping blades composite naturally
    positions = sorted([rng.uniform(0.05, 0.95) for _ in range(num_blades)])

    for cx in positions:
        poly, base_col, tip_col = blade_polygon(cx, tex_w, tex_h, rng)

        ys   = [p[1] for p in poly]
        y_min, y_max = min(ys), max(ys)
        span = max(y_max - y_min, 1)

        # Draw this blade on a temporary layer, then composite onto main
        layer = Image.new("RGBA", (tex_w, tex_h), (0, 0, 0, 0))
        layer_draw = ImageDraw.Draw(layer)

        # Draw the full polygon in base colour first
        layer_draw.polygon(poly, fill=base_col + (255,))

        # Overlay gradient: draw horizontal strips from base to tip colour
        # using a clipping approach via the polygon mask
        mask_layer = Image.new("L", (tex_w, tex_h), 0)
        mask_draw  = ImageDraw.Draw(mask_layer)
        mask_draw.polygon(poly, fill=255)

        for y in range(y_min, y_max + 1):
            t   = 1.0 - (y - y_min) / span
            col = lerp_color(base_col, tip_col, t)
            # Draw full-width scanline, then mask will clip it to blade shape
            grad_strip = Image.new("RGBA", (tex_w, 1), col + (255,))
            layer.paste(grad_strip, (0, y), mask=mask_layer.crop((0, y, tex_w, y + 1)))

        # Composite this blade onto the main image
        rgba = Image.alpha_composite(rgba, layer)

        # Update alpha mask
        alpha_draw = ImageDraw.Draw(alpha_img)
        alpha_draw.polygon(poly, fill=255)

    # Optional: very slight blur on edges to soften the pixel-perfect boundary
    if soft_edge:
        alpha_img = alpha_img.filter(ImageFilter.GaussianBlur(radius=0.8))

    return rgba, alpha_img


def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    OUT_DIR    = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "assets", "grass"))

    parser = argparse.ArgumentParser(
        description="Generate matched diffuse+alpha textures for grass billboards")
    parser.add_argument("--blades",   type=int, default=10,
                        help="Number of blade shapes per texture (default 10)")
    parser.add_argument("--res-w",    type=int, default=512,
                        help="Texture width in pixels (default 512)")
    parser.add_argument("--res-h",    type=int, default=512,
                        help="Texture height in pixels (default 512)")
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--no-blur",  action="store_true",
                        help="Disable soft edge blur on alpha")
    parser.add_argument("--out-dir",  type=str, default=OUT_DIR)
    args = parser.parse_args()

    print(f"Generating billboard texture: {args.blades} blades, "
          f"{args.res_w}×{args.res_h}px, seed={args.seed}")

    rgba, alpha = generate_texture(
        args.blades, args.res_w, args.res_h, args.seed,
        soft_edge=not args.no_blur)

    os.makedirs(args.out_dir, exist_ok=True)

    diff_path  = os.path.join(args.out_dir, "grass_billboard_diffuse.png")
    alpha_path = os.path.join(args.out_dir, "grass_billboard_alpha.png")

    rgba.save(diff_path)
    alpha.save(alpha_path)

    # Count opaque pixels to estimate alpha density (affects render cost)
    opaque = sum(1 for px in alpha.getdata() if px > 127)
    total  = args.res_w * args.res_h
    print(f"  Diffuse → {diff_path}")
    print(f"  Alpha   → {alpha_path}")
    print(f"  Coverage: {opaque}/{total} px opaque ({100*opaque/total:.1f}%)")
    print(f"  Tip: lower coverage = fewer alpha re-traces = faster renders")
    print(f"  Try --blades 6 for sparser look, --blades 14 for dense.")


if __name__ == "__main__":
    main()
