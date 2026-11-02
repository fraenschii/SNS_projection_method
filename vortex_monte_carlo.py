"""Monte Carlo velocity/vorticity snapshots using snsprojection_solver.py.

Five snapshots include t=0 and T. Figures show the first path, the ensemble
mean, and the sample standard deviation, separately for velocity/vorticity.

Mean and variance are accumulated with Welford's stable online algorithm.
Velocity SD means sqrt(Var(u1)+Var(u2)), NOT SD of the speed |u|.
Arrow colours for the mean show |mean(u)|, NOT mean(|u|).
Vorticity statistics are computed from each path's curl, not from velocity SD.
Statistics are evaluated at the FE nodes and interpolated for visualization.
Vorticity uses DG1 cell connectivity, preserving jumps across cell edges.

To regenerate figures from an existing folder:
    from vortex_monte_carlo import plot_statistics
    plot_statistics("/absolute/path/to/the/result/folder")

Use at least two samples
All paths start from the same deterministic initial condition.
Changing sigma with the same seed, T and time grid reuses Brownian paths.
"""
from datetime import datetime
from pathlib import Path
from time import perf_counter
import json

import numpy as np
from mpi4py import MPI
from dolfinx.mesh import create_unit_square
from dolfinx.plot import vtk_mesh
from snsprojection_solver import solve_projection

# 1. Experiment settings.
INITIAL_DATA = "same_sign"          # "same_sign" : Vortex pair; other choices: "vortex_sheet", "mixed_sign" (four vortices), "smooth_vortex"
N = 64                             # Subdivisions in each spatial direction
NUM_STEPS = 1024
T = 2.0
VISCOSITY = 0.001
SIGMA = 1.0                        # G(u) d beta = SIGMA * u * d beta
SAMPLES = 2                      


SEED = 2026                        # Sample s uses seed SEED+s (s starts at 0)
SNAPSHOT_COUNT = 5
PREDICTION_SOLVER = "iterative"     #  or "direct"
SHOW_PLOTS = True                   # False still saves all PDF/PNG figures
QUIVER_SCALE = 5.0                  # Larger values give shorter velocity arrows
RESUME_FOLDER = None


def update_moments(mean, m2, values, count):
    """Update an array-valued mean and sum of squared deviations, in place."""
    delta = values - mean
    mean += delta / count
    m2 += delta * (values - mean)


def save_statistics(folder, state):
    """Save statistics after each completed sample"""
    completed = state["completed_samples"]
    for field in ("velocity", "vorticity"):
        variance = (np.maximum(state[field+"_m2"] / (completed-1), 0) if completed > 1
                    else np.full_like(state[field+"_mean"], np.nan))
        state[field+"_std"] = np.sqrt(variance.sum(axis=-1) if field == "velocity" else variance)
    temporary = folder / "statistics.tmp"
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **state)
    temporary.replace(folder / "statistics.npz")
    np.savetxt(folder / "runtimes.csv",
               np.column_stack((np.arange(1, completed+1), state["wall_seconds"])),
               delimiter=",", header="sample,solver_seconds", comments="")


def run_experiment():
    """Solve independent paths."""

   # only initial data from the list currently works
    if INITIAL_DATA not in ("same_sign", "mixed_sign", "vortex_sheet", "smooth_vortex"):
        raise ValueError("Choose a named INITIAL_DATA profile listed above")
    parameters = dict(initial_data=INITIAL_DATA, n=N, num_steps=NUM_STEPS, T=T,
                      viscosity=VISCOSITY, sigma=SIGMA, seed=SEED,
                      snapshot_count=SNAPSHOT_COUNT, prediction_solver=PREDICTION_SOLVER)
    mesh = create_unit_square(MPI.COMM_WORLD, N, N)
    state = dict(parameters=json.dumps(parameters), completed_samples=0, wall_seconds=[])

    # create a folder or resume in specified one
    if RESUME_FOLDER is None:
        tag = (f"n{N}_N{NUM_STEPS}_T{T:g}_nu{VISCOSITY:g}_sigma{SIGMA:g}_"
               + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        folder = Path(__file__).resolve().parent / "results" / "vortex_monte_carlo" / INITIAL_DATA / tag
        folder.mkdir(parents=True, exist_ok=False)
        (folder / "parameters.json").write_text(json.dumps(parameters, indent=2))
    else:
        folder = Path(RESUME_FOLDER).expanduser().resolve()
        with np.load(folder / "statistics.npz", allow_pickle=False) as saved:
            state = {key: saved[key].copy() for key in saved.files
                     if key not in ("velocity_variance_components", "velocity_second_moment", "vorticity_second_moment")}
        if json.loads(str(state["parameters"])) != parameters:
            raise ValueError("Resume parameters differ from the checkpoint; restore them or start a new folder")
        if int(state["completed_samples"]) > SAMPLES:
            raise ValueError("SAMPLES is smaller than the completed checkpoint count")
        state["wall_seconds"] = state["wall_seconds"].tolist()
        print(f"Resuming after {int(state['completed_samples'])} completed samples", flush=True)

    print(f"Results: {folder}\nTarget: {SAMPLES} samples; initial data: {INITIAL_DATA}", flush=True)
    for sample in range(int(state["completed_samples"]), SAMPLES):
        print(f"Sample {sample+1}/{SAMPLES} (seed {SEED+sample}) ...", flush=True)
        rng = np.random.default_rng(SEED+sample)
        increments = rng.normal(0, np.sqrt(T/NUM_STEPS), NUM_STEPS)
        result = solve_projection(
            n=N, domain=mesh, num_steps=NUM_STEPS, T=T, viscosity=VISCOSITY,
            noise_amplitude=SIGMA, initial_data=INITIAL_DATA, increments=increments,
            snapshot_count=SNAPSHOT_COUNT, prediction_solver=PREDICTION_SOLVER,
            verbose=False)
        snapshots = result["snapshots"]
        values = dict(velocity=np.stack([s["velocity"] for s in snapshots]).reshape(len(snapshots), -1, 2),
                      vorticity=np.stack([s["vorticity"] for s in snapshots]))

        # the first sample for plotting snapshots etc.
        if sample == 0:
            topology, _, xyz_w = vtk_mesh(result["Vomega"])
            state.update(times=np.array([s["time"] for s in snapshots]),
                         velocity_xy=result["V"].tabulate_dof_coordinates()[:, :2].copy(),
                         vorticity_xy=xyz_w[:, :2].copy(),
                         vorticity_triangles=topology.reshape(-1, 4)[:, 1:].copy())
            for field, value in values.items():
                state[field+"_mean"] = np.zeros_like(value)
                state[field+"_m2"] = np.zeros_like(value)
                state[field+"_sample"] = value.copy()

        completed = sample+1
        state["completed_samples"] = completed
        for field, value in values.items():
            update_moments(state[field+"_mean"], state[field+"_m2"], value, completed)
        state["wall_seconds"].append(result["metrics"]["wall_seconds"])
        save_statistics(folder, state)
        # compute estimate of remaining time
        remaining_minutes = (SAMPLES-completed)*np.mean(state["wall_seconds"])/60
        print(f"  Solve: {state['wall_seconds'][-1]:.1f}s; completed {completed}; "
              f"estimated remaining solver time: {remaining_minutes:.1f} min", flush=True)
        del result

    plot_statistics(folder, show=SHOW_PLOTS)
    return folder


def plot_statistics(folder, show=True):
    """Replot a checkpoint without performing any PDE solves. Saves six figures."""
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from matplotlib.colors import Normalize

    folder = Path(folder).expanduser().resolve()
    with np.load(folder / "statistics.npz", allow_pickle=False) as saved:
        data = {name: saved[name].copy() for name in saved.files}
    count = int(data["completed_samples"])
    if count < 2:
        raise ValueError("At least two completed samples are needed for standard deviation plots")
    p = json.loads(str(data["parameters"]))
    times = data["times"]
    xy_u, xy_w = data["velocity_xy"], data["vorticity_xy"]
    triangles_w = mtri.Triangulation(xy_w[:, 0], xy_w[:, 1], data["vorticity_triangles"])
    triangles_u = mtri.Triangulation(xy_u[:, 0], xy_u[:, 1])
    rounded = np.round(xy_u, 12)
    xs, ys = np.unique(rounded[:, 0]), np.unique(rounded[:, 1])
    keep = (np.isin(rounded[:, 0], xs[::max(1, int(np.ceil(len(xs)/15)))])
            & np.isin(rounded[:, 1], ys[::max(1, int(np.ceil(len(ys)/15)))]))
    speed_limit = max(np.linalg.norm(data["velocity_"+s], axis=-1).max()
                      for s in ("sample", "mean"))
    # Only the rough sheet's signed initial vorticity gets a separate scale.
    split_initial = p["initial_data"] == "vortex_sheet"
    positive = np.flatnonzero(times > 0).tolist()
    initial = np.flatnonzero(times <= 0).tolist()
    signed_indices = positive if split_initial else list(range(len(times)))
    omega_limit = max(np.abs(data["vorticity_"+s][signed_indices]).max()
                      for s in ("sample", "mean"))
    initial_limit = max(np.abs(data["vorticity_"+s][initial]).max()
                        for s in ("sample", "mean"))
    figures = []
    started = perf_counter()
    settings = {"font.size": 8, "axes.titlesize": 9, "pdf.fonttype": 42}
    with plt.rc_context(settings):
        for field in ("velocity", "vorticity"):
            for statistic in ("sample", "mean", "std"):
                values = data[field+"_"+statistic]
                nonnegative = field == "velocity" or statistic == "std"
                limit = (values.max() if statistic == "std" else
                         speed_limit if field == "velocity" else omega_limit)
                limit = max(float(limit), 1e-14)
                fig, axes = plt.subplots(1, len(times), figsize=(2.6*len(times), 3.0),
                                         squeeze=False, layout="constrained")
                figures.append(fig)
                axes = axes[0]
                images = []
                for j, time in enumerate(times):
                    ax = axes[j]
                    panel_limit = (max(float(initial_limit), 1e-14)
                                   if field == "vorticity" and statistic != "std"
                                   and split_initial and j in initial else limit)
                    if field == "velocity" and statistic != "std":
                        speed = np.linalg.norm(values[j], axis=-1)
                        image = ax.quiver(xy_u[keep, 0], xy_u[keep, 1],
                                          values[j, keep, 0], values[j, keep, 1], speed[keep],
                                          norm=Normalize(0, limit), cmap="viridis", angles="xy",
                                          scale_units="xy", scale=QUIVER_SCALE, width=0.006)
                    else:
                        image = ax.tripcolor(
                            triangles_u if field == "velocity" else triangles_w, values[j],
                            shading="gouraud", cmap="magma" if nonnegative else "RdBu_r",
                            vmin=0 if nonnegative else -panel_limit, vmax=panel_limit,
                            rasterized=True)
                    images.append(image)
                    ax.set_title(f"$t={time:.2f}$")
                    ax.set_aspect("equal")
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    ax.set_xticks([0, .5, 1])
                    ax.set_yticks([0, .5, 1])
                    ax.set_xlabel("$x$")
                    if j == 0:
                        ax.set_ylabel("$y$")
                label = ("Velocity fluctuation SD" if field == "velocity" and statistic == "std"
                         else "Speed of mean velocity" if field == "velocity" and statistic == "mean"
                         else "Speed" if field == "velocity"
                         else "Vorticity SD" if statistic == "std" else "Vorticity")
                if field == "vorticity" and statistic != "std" and split_initial:
                    fig.colorbar(images[initial[0]], ax=[axes[j] for j in initial],
                                 label="Initial (separate scale)", shrink=.75, pad=.02)
                    fig.colorbar(images[positive[0]], ax=[axes[j] for j in positive],
                                 label=label, shrink=.75, pad=.02)
                else:
                    fig.colorbar(images[-1], ax=axes.tolist(), label=label, shrink=.75, pad=.02)
                heading = ("Sample 1" if statistic == "sample" else
                           f"Monte Carlo mean (M={count})" if statistic == "mean" else
                           f"Sample standard deviation (M={count}, denominator M-1)")
                fig.suptitle(f"{field.capitalize()}: {heading}\n"
                             f"{p['initial_data']}, n={p['n']}, N={p['num_steps']}, "
                             f"nu={p['viscosity']:g}, sigma={p['sigma']:g}", fontsize=10)
                fig.savefig(folder / f"{field}_{statistic}.pdf", dpi=300)
                fig.savefig(folder / f"{field}_{statistic}.png", dpi=300)
        print(f"Saved six PDF/PNG figures in {folder} ({perf_counter()-started:.1f}s plotting)", flush=True)
        if show:
            plt.show()
        for fig in figures:
            plt.close(fig)


if __name__ == "__main__":
    run_experiment()
