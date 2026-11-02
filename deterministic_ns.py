"""One deterministic solve and five velocity/vorticity snapshots.

Run: python deterministic_ns.py

"""
from datetime import datetime
from pathlib import Path
import json

import numpy as np

INITIAL_DATA = "vortex_sheet"      # Options: "vortex_sheet", "mixed_sign", "smooth_vortex", "same_sign"
N = 32                             # Spatial subdivisions per direction
NUM_STEPS = 512                   # number of time steps
T = 1.0                            # final time
VISCOSITY = 0.001                  # \nu
SNAPSHOT_COUNT = 5                  
# number of plots at different times (will be plotted at t=i/SNAPSHOT_COUNT, i=0,...,SNAPSHOT_COUNT)
PREDICTION_SOLVER = "iterative"     # Or "direct"
SHOW_PLOTS = True
QUIVER_SCALE = 5.0                  # Same arrow scale as the stochastic figures


def run_experiment():
    """Solve the deterministic system (G(u)=0), and make two figures with velocity and vorticity."""
    from dolfinx.plot import vtk_mesh
    from snsprojection_solver import solve_projection

    
    if not 2 <= SNAPSHOT_COUNT <= NUM_STEPS + 1:
        raise ValueError("Require 2 <= SNAPSHOT_COUNT <= NUM_STEPS+1")
    parameters = dict(initial_data=INITIAL_DATA, n=N, num_steps=NUM_STEPS,
                      T=T, viscosity=VISCOSITY, sigma=0.0,
                      snapshot_count=SNAPSHOT_COUNT,
                      prediction_solver=PREDICTION_SOLVER)
    tag = (f"n{N}_N{NUM_STEPS}_T{T:g}_nu{VISCOSITY:g}_sigma0_"
           + datetime.now().strftime("%Y%m%d_%H%M%S_%f")) # tag to name the folder where results will be saved
    folder = (Path(__file__).resolve().parent / "results" / "deterministic_ns"
              / INITIAL_DATA / tag)
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "parameters.json").write_text(json.dumps(parameters, indent=2))
    print(f"Results: {folder}", flush=True)

    # call projection method solver with given paramenters
    result = solve_projection(
        n=N, num_steps=NUM_STEPS, T=T, viscosity=VISCOSITY,
        noise_amplitude=0.0, increments=np.zeros(NUM_STEPS),
        initial_data=INITIAL_DATA, snapshot_count=SNAPSHOT_COUNT,
        prediction_solver=PREDICTION_SOLVER,
        verbose=False) 
    snapshots = result["snapshots"]
    topology, _, xyz_w = vtk_mesh(result["Vomega"])
    np.savez_compressed(
        folder / "solution.npz", parameters=json.dumps(parameters),
        times=np.array([s["time"] for s in snapshots]),
        velocity=np.stack([s["velocity"] for s in snapshots]).reshape(len(snapshots), -1, 2),
        vorticity=np.stack([s["vorticity"] for s in snapshots]),
        velocity_xy=result["V"].tabulate_dof_coordinates()[:, :2],
        vorticity_xy=xyz_w[:, :2],
        vorticity_triangles=topology.reshape(-1, 4)[:, 1:],
        wall_seconds=result["metrics"]["wall_seconds"])
    (folder / "metrics.json").write_text(json.dumps(result["metrics"], indent=2))
    print(f"Solve: {result['metrics']['wall_seconds']:.1f}s", flush=True)
    plot_solution(folder, show=SHOW_PLOTS)
    return folder


# to plot the approximations and save the figures
def plot_solution(folder, show=True): 
    """Load solution.npz and save velocity/vorticity figures as PDF and PNG."""
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from matplotlib.colors import Normalize

    folder = Path(folder).expanduser().resolve()
    with np.load(folder / "solution.npz", allow_pickle=False) as saved:
        data = {name: saved[name].copy() for name in saved.files}
    p = json.loads(str(data["parameters"]))
    times = data["times"]
    xy_u, xy_w = data["velocity_xy"], data["vorticity_xy"]
    triangles_w = mtri.Triangulation(xy_w[:, 0], xy_w[:, 1], data["vorticity_triangles"])
    rounded = np.round(xy_u, 12)
    xs, ys = np.unique(rounded[:, 0]), np.unique(rounded[:, 1])
    keep = (np.isin(rounded[:, 0], xs[::max(1, int(np.ceil(len(xs)/15)))])
            & np.isin(rounded[:, 1], ys[::max(1, int(np.ceil(len(ys)/15))) ]))
    split_initial = p["initial_data"] == "vortex_sheet"
    initial = np.flatnonzero(times <= 0).tolist()
    positive = np.flatnonzero(times > 0).tolist()
    signed_indices = positive if split_initial else list(range(len(times)))
    figures = []
    settings = {"font.size": 8, "axes.titlesize": 9, "pdf.fonttype": 42}
    with plt.rc_context(settings):
        for field in ("velocity", "vorticity"):
            values = data[field]
            limit = (np.linalg.norm(values, axis=-1).max() if field == "velocity"
                     else np.abs(values[signed_indices]).max()) 
            limit = max(float(limit), 1e-14)
            initial_limit = max(float(np.abs(values[initial]).max()), 1e-14)
            fig, axes = plt.subplots(1, len(times), figsize=(2.6*len(times), 3.0),
                                     squeeze=False, layout="constrained")
            figures.append(fig)
            axes, images = axes[0], []
            for j, time in enumerate(times):
                ax = axes[j]
                if field == "velocity":
                    speed = np.linalg.norm(values[j], axis=-1)
                    image = ax.quiver(
                        xy_u[keep, 0], xy_u[keep, 1],
                        values[j, keep, 0], values[j, keep, 1], speed[keep],
                        norm=Normalize(0, limit), cmap="viridis", angles="xy",
                        scale_units="xy", scale=QUIVER_SCALE, width=0.006)
                else:
                    panel_limit = initial_limit if split_initial and j in initial else limit
                    image = ax.tripcolor(
                        triangles_w, values[j], shading="gouraud", cmap="RdBu_r",
                        vmin=-panel_limit, vmax=panel_limit, rasterized=True)
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
            if field == "vorticity" and split_initial:
                fig.colorbar(images[initial[0]], ax=[axes[j] for j in initial],
                             label="Initial (separate scale)", shrink=.75, pad=.02)
                fig.colorbar(images[positive[0]], ax=[axes[j] for j in positive],
                             label="Vorticity", shrink=.75, pad=.02)
            else:
                fig.colorbar(images[-1], ax=axes.tolist(),
                             label="Speed" if field == "velocity" else "Vorticity",
                             shrink=.75, pad=.02)
            fig.suptitle(f"{field.capitalize()}: deterministic solution\n"
                         f"{p['initial_data']}, n={p['n']}, N={p['num_steps']}, "
                         f"nu={p['viscosity']:g}, sigma=0", fontsize=10)
            fig.savefig(folder / f"{field}.pdf", dpi=300)
            fig.savefig(folder / f"{field}.png", dpi=300)
        print(f"Saved velocity/vorticity PDF and PNG figures in {folder}", flush=True)
        if show:
            plt.show()
        for fig in figures:
            plt.close(fig)


if __name__ == "__main__":
    run_experiment()
