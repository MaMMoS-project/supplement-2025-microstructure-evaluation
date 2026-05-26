# supplement-2025-microstructure-evaluation
Supplementary material for evaluation of the impact of microstructure on coercive fields.

The script `hard-magnet-grains.py` runs micromagnetic hysteresis simulations for
different grains and two different temperatures.

To run the script you need [pixi](https://pixi.sh). Then:

1. Install the required dependencies: `pixi install --frozen`
2. Run the pixi task `full`.
  - To run the simulation on cpu, run `pixi run full`.
  - For Cuda 12 support, run `pixi run -e cuda full`.

The explicitly required software packages are recorded the `pixi.toml` file.
Fixed versions of all dependencies are stored in `pixi.lock`.

**Note**: Only Linux is supported, because esys-escript on conda-forge is only available for Linux.

Mesh files in `.fly` format are provided via the [MaMMoS software suite](https://mammos-project.github.io/mammos/) and downloaded from Zenodo. More details are available [here](https://mammos-project.github.io/mammos/examples/mammos-mumag/meshes.html).
