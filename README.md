# supplement-2025-microstructure-evaluation
Supplementary material for evaluation of the impact of microstructure on coercive fields.

The script `hard-magnet-grains.py` runs micromagnetic hysteresis simulations for
different grains and two different temperatures.

To run the script you need [pixi](https://pixi.sh). Then:

1. Install the required dependencies: `pixi install --frozen`
2. Run the script in the new environment: `pixi run python3 hard-magnet-grains.py`

The explicitly required software packages are recorded the `pixi.toml` file.
Fixed versions of all dependencies are stored in `pixi.lock`.

**Note**: Only Linux is supported, because esys-escript on conda-forge is only available for Linux.

Mesh files in `.fly` format are managed with git lfs, see https://git-lfs.com/ for details on how 
to install and use it.
