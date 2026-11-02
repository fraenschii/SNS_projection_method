"""Space refinement with smooth initial data and scalar noise sigma*u*dW.

All meshes use the SAME time grid and the SAME noise samples.
The plotted strong error is sqrt(mean(||u_h(T)-u_ref(T)||_L2**2)).
Reference subdivisions must be divisible by every coarse n, so these meshes
are nested. A coarse P2 field can then be represented in the reference P2
space before integrating the error; coefficient arrays cannot be subtracted
directly across different meshes.
"""
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from dolfinx import fem
from dolfinx.mesh import create_unit_square
from ufl import inner, dx
from snsprojection_solver import solve_projection, smooth_initial_velocity

# 1. Choose initial data and experiment settings.
# u0 = (d_y psi, -d_x psi), psi = sin(pi*x)^2 sin(pi*y)^2.
INITIAL_DATA = smooth_initial_velocity
T = 0.5
VISCOSITY = 0.01
SIGMA = 0.3
MESH_COUNTS = [4, 8, 16, 32]            # Subdivisions in each spatial direction
REFERENCE_N = 4 * max(MESH_COUNTS)
TIME_STEPS = 256                   # Fixed across ALL meshes
SAMPLES = 100
SEED = 2026
REFERENCE_RATE = 3.0                # Smooth P2 velocity L2 approximation benchmark


reference_mesh = create_unit_square(MPI.COMM_WORLD, REFERENCE_N, REFERENCE_N)
stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
folder = Path(__file__).resolve().parent / "results" / "spatial_refinement" / stamp
folder.mkdir(parents=True)
print(f"Results: {folder}", flush=True)

# Generate independent samples once. Reuse them at EVERY refinement.
rng = np.random.default_rng(SEED)
increments = rng.normal(0, np.sqrt(T / TIME_STEPS), size=(SAMPLES, TIME_STEPS))

# 2. Compute one reference solution for each noise sample.
references = []
reference_seconds = np.zeros(SAMPLES)
for sample in range(SAMPLES):
    print(f"Reference sample {sample + 1}/{SAMPLES}, n={REFERENCE_N}", flush=True)
    result = solve_projection(
        n=REFERENCE_N, domain=reference_mesh, num_steps=TIME_STEPS, T=T,
        viscosity=VISCOSITY, noise_amplitude=SIGMA, initial_data=INITIAL_DATA,
        increments=increments[sample], snapshot_count=0, verbose=False)
    references.append(result["velocity"])
    reference_seconds[sample] = result["metrics"]["wall_seconds"]
    print(f"Reference sample {sample + 1}: "f"{reference_seconds[sample]:.1f} seconds",
          flush=True,)

# 3. Outer loop: refinement. Inner loop: independent noise samples.
squared_errors = np.full((len(MESH_COUNTS), SAMPLES), np.nan)
run_seconds = np.full_like(squared_errors, np.nan)
rms_errors = np.full(len(MESH_COUNTS), np.nan)
rates = np.full(len(MESH_COUNTS), np.nan)
table = np.full((len(MESH_COUNTS), 6), np.nan)
h_values = np.sqrt(2) / np.asarray(MESH_COUNTS)

for j, n in enumerate(MESH_COUNTS):
    mesh = create_unit_square(MPI.COMM_WORLD, n, n)
    for sample in range(SAMPLES):
        print(f"n={n}, sample {sample + 1}/{SAMPLES}", flush=True)
        result = solve_projection(
            n=n, domain=mesh, num_steps=TIME_STEPS, T=T,
            viscosity=VISCOSITY, noise_amplitude=SIGMA, initial_data=INITIAL_DATA,
            increments=increments[sample], snapshot_count=0, verbose=False)

        # Represent the coarse FE function on the nested reference mesh.
        reference = references[sample]
        V_ref = reference.function_space
        cells = np.arange(reference_mesh.topology.index_map(2).size_local, dtype=np.int32)
        interpolation_data = fem.create_interpolation_data(V_ref, result["velocity"].function_space, cells)
        coarse_on_reference = fem.Function(V_ref)
        coarse_on_reference.interpolate_nonmatching(result["velocity"], cells, interpolation_data)
        difference = coarse_on_reference - reference
        error = fem.form(inner(difference, difference) * dx(domain=reference_mesh))
        squared_errors[j, sample] = max(0.0, float(np.real(
            fem.assemble_scalar(error))))
        run_seconds[j, sample] = result["metrics"]["wall_seconds"]
        print(f"n={n}, sample {sample + 1}: "f"{run_seconds[j, sample]:.1f} seconds",
              flush=True,)

    # Monte Carlo estimate of E[||error||_L2^2], then its square root.
    mean_squared_error = squared_errors[j].mean()
    rms_errors[j] = np.sqrt(mean_squared_error)
    if j > 0 and rms_errors[j] > 0 and rms_errors[j-1] > 0:
        rates[j] = np.log(rms_errors[j-1] / rms_errors[j]) / np.log(h_values[j-1] / h_values[j])
    table[j] = [n, h_values[j], mean_squared_error, rms_errors[j], rates[j], run_seconds[j].mean()]

    # Save after each completed refinement, including raw sample errors.
    print(f"  RMS L2 error={rms_errors[j]:.6e}, rate={rates[j]:.2f}, "
          f"mean runtime={run_seconds[j].mean():.1f}s", flush=True)
    np.savetxt(folder / "errors.csv", table[:j+1], delimiter=",", comments="",
               header="n,h,mean_squared_L2_error,rms_L2_error,rate,mean_seconds")
    np.savez_compressed(
        folder / "errors.npz", mesh_counts=MESH_COUNTS, h=h_values,
        squared_errors=squared_errors, rms_errors=rms_errors, rates=rates,
        run_seconds=run_seconds, reference_seconds=reference_seconds,
        increments=increments, time_steps=TIME_STEPS, reference_n=REFERENCE_N,
        T=T, viscosity=VISCOSITY, sigma=SIGMA, seed=SEED,
        initial_data=getattr(INITIAL_DATA, "__name__", str(INITIAL_DATA)))

# 4. Plot all expected errors against mesh size, and save the figure.
fig, ax = plt.subplots()
ax.loglog(h_values, rms_errors, "o-", label="Terminal velocity RMS L2 error")
if rms_errors[0] > 0:
    guide = rms_errors[0] * (h_values / h_values[0])**REFERENCE_RATE
    ax.loglog(h_values, guide, "k--",
              label=rf"Reference: $h^{{{REFERENCE_RATE:g}}}$")
ax.set_xlabel(r"$h$")
ax.set_ylabel(r"$(M^{-1}\sum_{s=1}^M\|u_h^{(s)}(T)-u_{ref}^{(s)}(T)\|_{L^2}^2)^{1/2}$")
ax.set_title(f"Space refinement: N={TIME_STEPS}, nu={VISCOSITY:g}, sigma={SIGMA:g}, M={SAMPLES}")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(folder / "convergence.pdf")
fig.savefig(folder / "convergence.png", dpi=200)
print(f"Saved errors and plots in {folder}", flush=True)
plt.show()
