# %config InlineBackend.figure_format = "retina"

import pathlib

import mammos_analysis
import mammos_dft
import mammos_entity as me
import mammos_mumag
import mammos_spindynamics
import mammos_units as u
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps


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

# output directory
outdir = pathlib.Path("out")

# fixing the seed for comparability between sizes
# seed is used for random thetas and phis only
seed = 42
np.random.seed(seed=seed)

# Allow convenient conversions between A/m and T
u.set_enabled_equivalencies(u.magnetic_flux_field())

# Load dft data
results_dft = mammos_dft.db.get_micromagnetic_properties(material_string, print_info=True)

# (optional todo) find out how to change to Tesla vs. Kelvin (or °C)
# Load spin dynamics data
results_spindynamics = mammos_spindynamics.db.get_spontaneous_magnetization(material_string)

# Use Kuz'min model to evaluate micromagnetic intrinsic properties
results_kuzmin = mammos_analysis.kuzmin_properties(
    T=results_spindynamics.T,
    Ms=results_spindynamics.Ms,
    K1_0=results_dft.Ku_0,
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
T = [300, 423] * u.K
coneangle = 15.0  # degree

CUBESIZES = [40, 80, 160] * u.nm
GRAINSIZES = [20, 40, 80] * u.nm
SHAPES = ["colu", "equi", "3plat"]
NUMGRAINS = [8, 8, 12]

# decomposition temp is 539 K of Fe16N2 https://doi.org/10.1063/9.0000628
for size, gsize in zip(CUBESIZES, GRAINSIZES):
    for shape, numgrains in zip(SHAPES, NUMGRAINS):
        simulations = []
        thetas, phis = randomAngleCap(coneangle, numgrains)
        mesh_name = f"cube{int(size.value)}_{shape}_grains{numgrains}_gsize{int(gsize.value)}"
        for temperature in T:
            print("-" * 10)
            print(f"sim {size} x {size} x {size} cube, filled with {shape} at {temperature}")
            print("-" * 10)
            Js_T = results_kuzmin.Ms(temperature).q.to(u.T)
            K1_T = results_kuzmin.K1(temperature).q
            A_T = results_kuzmin.A(temperature).q
            delta_0 = np.sqrt(A_T / K1_T)
            lex = np.sqrt((A_T * u.constants.mu0) / (Js_T**2))
            h_ani = 2 * K1_T / Js_T
            # arXiv:1603.08239v1 [cond-mat.mtrl-sci]
            # n = 0.27 ... forgot what this controls ...
            h_c_krofae = h_ani - 0.27 * np.log(gsize / (delta_0 * np.pi)) * Js_T

            print(
                f"T = {temperature}, Js = {Js_T:.3f}, K1 = {K1_T.to('MJ/m3'):.3f}, A = {A_T.to('pJ/m'):.3f}\ndelta0 = {delta_0.to('nm'):.3f}, lex = {lex.to('nm'):.3f}, Hani = {h_ani.to('T'):.3f}, N*Js = {Js_T / 3.0:.3f}, Hc(approx.) = {h_c_krofae.to('T'):.3f}"
            )

            # 1.5 nm meshsize at 300 K and 423 K for Fe16N2
            print(f"Running simulation for T={temperature:.0f}")
            results_hysteresis = mammos_mumag.hysteresis.run(
                mesh=mesh_name,
                # The zero value represents a zero parameters for the grain boundary phase.
                Ms=me.operations.concat_flat([results_kuzmin.Ms(temperature)] * numgrains, 0),
                A=me.operations.concat_flat([results_kuzmin.A(temperature)] * numgrains, 0),
                K1=me.operations.concat_flat([results_kuzmin.K1(temperature)] * numgrains, 0),
                theta=np.concatenate((thetas, [0.0])),
                phi=np.concatenate((phis, [0.0])),
                # (Bonus Ontology Quest) try to specify anisotropy field Hani of the material below
                # if Hani is wrong expression, then let's say, theoretical maximum switching field.
                h_start=(2.0 * u.T).to(u.A / u.m),
                h_final=(-2.0 * u.T).to(u.A / u.m),
                m_final=(-2.0 * u.T).to(u.A / u.m),
                h_n_steps=400,
                outdir=outdir / f"{shape}-{int(size.value)}-{temperature.value:.2f}",
            )
            simulations.append(results_hysteresis)

        # -------------------------------------------------
        # to have a proper workflow we might have to think about three phases
        # (I) preparation: of directories for standalone simulation
        # (II) simulation: execution/submission to queue
        # (III) post-processing: plotting extracting from finished simulation folder(s)
        # -------------------------------------------------

        T_Celsius = T.to("Celsius", equivalencies=u.temperature()).value
        Hcs = []
        for res in simulations:
            cf = mammos_analysis.hysteresis.extract_coercive_field(H=res.H, M=res.M).q.to("T")
            if np.isnan(cf):  # Above Tc
                cf = 0 * u.T
            Hcs.append(cf)
        Hcs = u.Quantity(Hcs)  # transform into array
        plt.plot(T_Celsius, Hcs, linestyle="-", marker="o")
        plt.xlabel("Temperature (degree Celsius)")
        plt.ylabel(r"$\mu_0 H_c$ (T)")
        plt.savefig(outdir / f"{shape}-{int(size.value)}-0.pdf")
        plt.close("all")

        colors = colormaps["plasma"].colors[:: np.ceil(256 / len(T)).astype("int")]
        fix, ax = plt.subplots()
        for temperature, sim, color in zip(T_Celsius, simulations, colors, strict=False):
            if np.isnan(sim.M.q).all():  # no Ms above Tc
                continue
            B = sim.H.q.to("T")
            J = sim.M.q.to("T")
            ax.plot(B, J, label=f"{temperature:.0f} °C", color=color)
            ax.plot(-B, -J, color=color)
            ax.set_title("Hysteresis Loop")
            ax.set_xlabel("B (T)")
            ax.set_ylabel("J (T)")
        ax.legend(loc="lower right")
        plt.savefig(outdir / f"{shape}-{int(size.value)}-1.pdf")
        plt.close("all")

print("Done")
