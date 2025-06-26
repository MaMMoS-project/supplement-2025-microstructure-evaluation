# %config InlineBackend.figure_format = "retina"

import shutil
import sys

import math

import mammos_analysis
import mammos_dft
import mammos_entity as me
import mammos_mumag
import mammos_spindynamics
import mammos_units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps

# importing another expanded hysteresis.py
import hysteresis


def randomDisc(n):
    w = np.random.uniform(-np.pi, np.pi, n)
    r = np.sqrt(np.random.rand(n))
    x = r * np.cos(w)
    y = r * np.sin(w)
    return x, y


def randomCap(n, h):
    x, y = randomDisc(n)
    k = h * (x * x + y * y)
    s = np.sqrt(h * (2.0 - k))
    return s * x, s * y, 1.0 - k


def randomPointsCap(w, n):
    angle = w * np.pi / 180.0
    h = 1 - np.cos(angle)
    return randomCap(n, h)


def randomAngleCap(w, n):
    if n == 1:
        return [w * np.pi / 180.0]
    else:
        xx, yy, zz = randomPointsCap(w, n)
        theta = np.arccos(zz)
        phi = np.arctan2(yy, xx)
        return theta, phi


# what material from database do we want to invesigate
material_string = "Fe16N2"

# fixing the seed for comparability between sizes
# seed is used for random thetas and phis only
seed = 42
np.random.seed(seed=seed)

# Allow convenient conversions between A/m and T
u.set_enabled_equivalencies(u.magnetic_flux_field())
# Alex: change material to Fe16N2
results_dft = mammos_dft.db.get_micromagnetic_properties(material_string, print_info=True)

# (optional todo) find out how to change to Tesla vs. Kelvin (or °C)
results_spindynamics = mammos_spindynamics.db.get_spontaneous_magnetization(material_string)

results_kuzmin = mammos_analysis.kuzmin_properties(
    T=results_spindynamics.T,
    # Ms=results_spindynamics.Ms.to(u.T), # this works
    Ms=results_spindynamics.Ms,
    K1_0=results_dft.K1_0,
)

# results_kuzmin.plot() # TODO bug/feature request (!) plot function needs to take units from plotted quantity
# plt.show()

# -------------------------------------------------
# planned number of grains and their arrangement
# equi 2 x 2 x 2
# cube 40 [x], cube 80 [x], cube 160 [x]
# colu 8 x 8 x 1
# cube 40 [x], cube 80 [x], cube 160 [x]
# plat 2 x 2 x 4
# cube 40 [x], cube 80 [x], cube 160 [x]
# on the cluster nodes lanthanum or gadoliunium

# and for each shape the temperature dependence of hysteresis computed
# T = np.linspace(0, 1.1 * results_kuzmin.Tc, 7)
T = [300, 423]
coneangle = 15.0  # degree

CUBESIZES = [40, 80, 160]  # in nm
GRAINSIZES = [
    20e-9,
    40e-9,
    80e-9,
]  # this is only true for equi-axed and only used for h_c_krofae
SHAPES = ["colu", "equi", "3plat12"]  #
NUMGRAINS = [8, 8, 12]  # TODO should later be extracted automatically from tesselation

# decomposition temp is 539 K of Fe16N2 https://doi.org/10.1063/9.0000628
for size, gsize in zip(CUBESIZES, GRAINSIZES):
    for shape, numgrains in zip(SHAPES, NUMGRAINS):
        simulations = []
        thetas, phis = randomAngleCap(coneangle, numgrains)
        for temperature in T:
            print("-" * 10)
            print(f"sim {size}x{size}x{size} cube, filled with {shape} at {temperature} K")
            print("-" * 10)
            Js_T = results_kuzmin.Ms(temperature).to(u.T)
            K1_T = results_kuzmin.K1(temperature)
            A_T = results_kuzmin.A(temperature)
            delta_0 = np.sqrt(A_T / K1_T)
            lex = np.sqrt((A_T * 4 * np.pi * 1e-7) / (Js_T**2))
            h_ani = ((2 * K1_T) / Js_T).to(u.T)

            # arXiv:1603.08239v1 [cond-mat.mtrl-sci]
            # n = 0.27 ... forgot what this controls ...
            h_c_krofae = h_ani.value - 0.27 * np.log(gsize / (delta_0.value * np.pi)) * Js_T.value

            print(
                f"T = {temperature} K, Js = {Js_T.value:.3f} T, K1 = {K1_T.value * 1e-6:.3f} MJ/m³, A = {A_T.value * 1e12:.3f} pJ/m\ndelta0 = {delta_0.value * 1e9:.3f} nm, lex = {lex.value * 1e9:.3f} nm, Hani = {h_ani.value:.3f} T, N*Js = {Js_T / 3.0:.3f} T, Hc(approx.) = {h_c_krofae:.3f} T"
            )

            # 1.5 nm meshsize at 300 K and 423 K for Fe16N2
            mesh_filepath = f"prereq/{shape}/{shape}_{size}.fly"
            print(f"Running simulation for T={temperature:.0f}")
            # results_hysteresis = mammos_mumag.hysteresis.run(
            results_hysteresis = hysteresis.run(
                mesh_filepath=mesh_filepath,
                # Alex: changed Ms, A, K1 to be lists now
                Ms=[results_kuzmin.Ms(temperature)] * numgrains,  # XXX currently equal entries
                A=[results_kuzmin.A(temperature)] * numgrains,  # XXX currently equal entries
                K1=[results_kuzmin.K1(temperature)] * numgrains,  # XXX currently equal entries
                theta=thetas,
                phi=phis,
                # (Bonus Ontology Quest) try to specify anisotropy field Hani of the material below
                # if Hani is wrong expression, then let's say, theoretical maximum switching field.
                # hstart=(7 * u.T).to(u.A / u.m),
                # hfinal=(-7 * u.T).to(u.A / u.m),
                hstart=(2.0 * u.T).to(u.A / u.m),
                hfinal=(-2.0 * u.T).to(u.A / u.m),
                # mfinal=-2., # XXX if I try this it does not work ... maybe because of ontology
                # or because of something else, I changed it now inside hysteresis.py I think
                # similar behaviour with hsteps or hstep ...
                hnsteps=400,
                # hnsteps=30,
                # hnsteps=4000,
            )
            simulations.append(results_hysteresis)

        # -------------------------------------------------
        # to have a proper workflow we might have to think about three phases
        # (I) preparation: of directories for standalone simulation
        # (II) simulation: execution/submission to queue
        # (III) post-processing: plotting extracting from finished simulation folder(s)
        # -------------------------------------------------

        Hcs = []
        for res in simulations:
            cf = mammos_analysis.hysteresis.extract_coercive_field(H=res.H, M=res.M).value
            if np.isnan(cf):  # Above Tc
                cf = 0
            print("cf", cf)
            print("cf*np.pi*4e-7", cf * np.pi * 4e-7)
            # print('cf.to(u.T)',cf.to(u.T)) # TODO does not work it's already a float32
            Hcs.append(cf)

        plt.plot(T, Hcs, linestyle="-", marker="o")
        plt.xlabel(me.T().axis_label)
        plt.ylabel(me.Hc().axis_label)

        plt.savefig(f"{shape}-{size}-0.png")
        # plt.savefig(f'{shape}-{size}-0.svg')
        plt.close("all")

        colors = colormaps["plasma"].colors[:: math.ceil(256 / len(T))]

        fix, ax = plt.subplots()
        for temperature, sim, color in zip(T, simulations, colors, strict=False):
            if np.isnan(sim.M).all():  # no Ms above Tc
                continue
            sim.plot(
                ax=ax,
                label=f"{temperature:.0f}",
                color=color,
                duplicate_change_color=False,
            )
        ax.legend(loc="lower right")

        plt.savefig(f"{shape}-{size}-1.png")
        # plt.savefig(f'{shape}-{size}-1.svg')

        plt.close("all")

        shutil.move("hystloop/hystloop.dat", f"./{shape}-{size}.dat")

print("Done")
