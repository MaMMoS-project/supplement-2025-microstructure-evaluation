### ATTENTION THIS IS AN ADAPTION OF ORIGINAL HYSTERESIS.PY ###
"""Functions for evaluating and processin the hysteresis loop."""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

import mammos_entity as me
import mammos_units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd
import pyvista as pv
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from mammos_mumag.materials import Materials
from mammos_mumag.parameters import Parameters
from mammos_mumag.simulation import Simulation

from mammos_mumag.hysteresis import Result

if TYPE_CHECKING:
    import matplotlib
    import pyvista


# Alex: changed Ms, A, K1 to lists here and added theta and phi as list of floats
def run(
    Ms: list[float] | u.Quantity | me.Entity,
    A: list[float] | u.Quantity | me.Entity,
    K1: list[float] | u.Quantity | me.Entity,
    theta: list[float],
    phi: list[float],
    mesh_filepath: pathlib.Path,
    hstart: float | u.Quantity,
    hfinal: float | u.Quantity,
    hstep: float | u.Quantity | None = None,
    hnsteps: int = 20,
    outdir: str | pathlib.Path = "hystloop",
) -> Result:
    r"""Run hysteresis loop.

    Args:
        Ms: Spontaneous magnetisation in :math:`\mathrm{A}/\mathrm{m}`.
        A: Exchange stiffness constant in :math:`\mathrm{J}/\mathrm{m}`.
        K1: First magnetocrystalline anisotropy constant in
            :math:`\mathrm{J}/\mathrm{m}^3`.
        mesh_filepath: TODO
        hstart: Initial strength of the external field.
        hfinal: Final strength of the external field.
        hstep: Step size.
        hnsteps: Number of steps in the field sweep.
        outdir: Directory where simulation results are written to.

    Returns:
       Result object.

    """
    if hstep is None:
        hstep = (hfinal - hstart) / hnsteps

    # TODO make belows ontology stuff work again
    # if not isinstance(A, u.Quantity) or A.unit != u.J / u.m:
    #    A = me.A(A, unit=u.J / u.m)
    # if not isinstance(K1, u.Quantity) or K1.unit != u.J / u.m**3:
    #    K1 = me.Ku(K1, unit=u.J / u.m**3)
    # if not isinstance(Ms, u.Quantity) or Ms.unit != u.A / u.m:
    #    Ms = me.Ms(Ms, unit=u.A / u.m)

    # before initializing Simulation() we need to prepare the domains
    # need to know how many grains ? it's in the lists of Ms, K1 and A
    domains = []
    # definition of each grains intrinsic material parameter
    for ms, k, a, t, p in zip(Ms, K1, A, theta, phi):
        d = {
            "theta": t,
            "phi": p,
            "K1": k,
            "K2": me.Ku(0),
            "Ms": ms,
            "A": a,
        }
        domains.append(d)

    # definition of grain boundary's intrinsic material parameter
    gb_params = {
        "theta": 0,
        "phi": 0.0,
        "K1": me.Ku(0),
        "K2": me.Ku(0),
        "Ms": me.Ms(0),
        "A": me.A(0),
    }
    domains.append(gb_params)

    # definition of inner air sphere's intrinsic material parameter
    airsphere_1_params = {
        "theta": 0.0,
        "phi": 0.0,
        "K1": me.Ku(0),
        "K2": me.Ku(0),
        "Ms": me.Ms(0),
        "A": me.A(0),
    }
    domains.append(airsphere_1_params)

    # definition of outer air sphere's intrinsic material parameter
    airsphere_2_params = {
        "theta": 0.0,
        "phi": 0.0,
        "K1": me.Ku(0),
        "K2": me.Ku(0),
        "Ms": me.Ms(0),
        "A": me.A(0),
    }
    domains.append(airsphere_2_params)

    sim = Simulation(
        mesh_filepath=mesh_filepath,
        materials=Materials(domains),
        parameters=Parameters(
            size=1.0e-9,
            scale=0,
            m_vect=[0, 0, 1],
            hstart=hstart.to(u.T, equivalencies=u.magnetic_flux_field()).value,
            hfinal=hfinal.to(u.T, equivalencies=u.magnetic_flux_field()).value,
            hstep=hstep.to(u.T, equivalencies=u.magnetic_flux_field()).value,
            h_vect=[0.01745, 0, 0.99984],
            mstep=0.4,
            mfinal=-2.0,  # Alex changed it: mfinal to -2
            tol_fun=1e-10,
            tol_hmag_factor=1,
            precond_iter=10,
        ),
    )
    sim.run_loop(outdir=outdir, name="hystloop")
    df = pd.read_csv(
        f"{outdir}/hystloop.dat",
        delimiter=" ",
        names=["configuration_type", "mu0_Hext", "polarisation", "energy_density"],
    )
    return Result(
        H=me.Entity(
            "ExternalMagneticField",
            # Alex change below
            # value=(df["mu0_Hext"].to_numpy() * u.T).to(
            #    u.A / u.m, equivalencies=u.magnetic_flux_field()
            # ),
            value=(df["mu0_Hext"].to_numpy() * u.T),  # want to have it in Tesla
            # ),
            # unit=u.A / u.m,
            unit=u.T,
        ),
        M=me.Ms(
            # Alex change below
            # (df["polarisation"].to_numpy() * u.T).to(
            #    u.A / u.m, equivalencies=u.magnetic_flux_field()
            # ),
            # unit=u.A / u.m,
            (df["polarisation"].to_numpy() * u.T),
            # ),
            unit=u.T,
        ),
        energy_density=me.Entity("EnergyDensity", value=df["energy_density"], unit=u.J / u.m**3),
        configurations={i + 1: fname for i, fname in enumerate(sorted(pathlib.Path(outdir).resolve().glob("*.vtu")))},
        configuration_type=df["configuration_type"].to_numpy(),
    )
