## Building and Running

1. Navigate to the build directory or make a new one:
   ```bash
   mkdir build && cd build
   ```
2. Configure and build:
   ```bash
   cmake ..
   cmake --build .
   ```
   This produces two executables: **renderer** (offline raytracer) and **StagePreview** (Polyscope viewer).

   To build only the **renderer** (no Polyscope fetch or StagePreview):
   ```bash
   cmake --build . --target renderer
   ```

3. Run from the HW1 directory (make sure paths in config files resolve correctly):
    - Renderer: `./build/renderer` or `./build/renderer config/frog.json`
    - StagePreview: `./build/StagePreview` or `./build/StagePreview config/frog.json`


### Config files

Scene configs (camera, settings, scene meshes) live in `config/`, e.g. `config/frog.json`. Paths inside JSON (e.g. `"./assets/meshes/frog.obj"`) are relative to the **HW1** directory.


### Polyscope (StagePreview)

**StagePreview** uses [Polyscope](https://polyscope.run/) to show the camera, pixel positions, rays, and meshes from a config file. Run it, then use the Polyscope UI to orbit the view and inspect structures. No extra setup needed.

### Tweaking the scene

To change camera or mesh setup, edit a JSON in `config/`. Materials are defined in `include/ray.h`.