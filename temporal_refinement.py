"""Time refinement with smooth initial data and scalar noise sigma*u*dW.

For each sample we first compute a reference on the SAME spatial mesh.
Coarse Brownian increments are sums of that sample's fine increments.
The plotted strong error is sqrt(mean(||u_N(T)-u_ref(T)||_L2**2)).

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
MESH_N = 32                         # Fixed subdivisions in each spatial direction
STEP_COUNTS = [16, 32, 64, 128, 256]       # Time refinements
REFERENCE_STEPS = 16 * max(STEP_COUNTS)
SAMPLES = 100
SEED = 2026
REFERENCE_RATE = 0.5                # Comparison slope for the RMS strong time error
PREDICTION_SOLVER = "direct"        # Sparse LU avoids iterative prediction failures



mesh = create_unit_square(MPI.COMM_WORLD, MESH_N, MESH_N)
stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
folder = Path(__file__).resolve().parent / "results" / "temporal_refinement" / stamp
folder.mkdir(parents=True)
print(f"Results: {folder}", flush=True)

# Generate independent samples once. Reuse them at every refinement.
rng = np.random.default_rng(SEED)
fine_increments = rng.normal(0, np.sqrt(T / REFERENCE_STEPS),
                             size=(SAMPLES, REFERENCE_STEPS))

# 2. Compute one reference solution for each noise sample.
references = []
reference_seconds = np.zeros(SAMPLES)
for sample in range(SAMPLES):
    print(f"Reference sample {sample + 1}/{SAMPLES}, N={REFERENCE_STEPS}", flush=True)
    result = solve_projection(
        n=MESH_N, domain=mesh, num_steps=REFERENCE_STEPS, T=T,
        viscosity=VISCOSITY, noise_amplitude=SIGMA, initial_data=INITIAL_DATA,
        increments=fine_increments[sample], snapshot_count=0, verbose=False,
        prediction_solver=PREDICTION_SOLVER)
    references.append(result["velocity"])
    reference_seconds[sample] = result["metrics"]["wall_seconds"]

# 3. Outer loop: refinement. Inner loop: independent noise samples.
squared_errors = np.full((len(STEP_COUNTS), SAMPLES), np.nan)
run_seconds = np.full_like(squared_errors, np.nan)
rms_errors = np.full(len(STEP_COUNTS), np.nan)
rates = np.full(len(STEP_COUNTS), np.nan)
table = np.full((len(STEP_COUNTS), 6), np.nan)
dt_values = T / np.asarray(STEP_COUNTS)

for j, steps in enumerate(STEP_COUNTS):
    for sample in range(SAMPLES):
        print(f"N={steps}, sample {sample + 1}/{SAMPLES}", flush=True)
        # Sum consecutive fine increments; do not draw a new path here.
        increments = fine_increments[sample].reshape(steps, REFERENCE_STEPS // steps).sum(axis=1)
        result = solve_projection(
            n=MESH_N, domain=mesh, num_steps=steps, T=T,
            viscosity=VISCOSITY, noise_amplitude=SIGMA, initial_data=INITIAL_DATA,
            increments=increments, snapshot_count=0, verbose=False,
            prediction_solver=PREDICTION_SOLVER)
        difference = result["velocity"] - references[sample]
        error = fem.form(inner(difference, difference) * dx(domain=mesh))
        squared_errors[j, sample] = max(0.0, float(np.real(
            fem.assemble_scalar(error))))
        run_seconds[j, sample] = result["metrics"]["wall_seconds"]

    # Monte Carlo estimate of E[||error||_L2^2], then its square root.
    mean_squared_error = squared_errors[j].mean()
    rms_errors[j] = np.sqrt(mean_squared_error)
    if j > 0 and rms_errors[j] > 0 and rms_errors[j-1] > 0:
        rates[j] = np.log(rms_errors[j-1] / rms_errors[j]) / np.log(dt_values[j-1] / dt_values[j])
    table[j] = [steps, dt_values[j], mean_squared_error, rms_errors[j], rates[j], run_seconds[j].mean()]

    # Save after each completed refinement, including raw sample errors.
    print(f"  RMS L2 error={rms_errors[j]:.6e}, rate={rates[j]:.2f}, "
          f"mean runtime={run_seconds[j].mean():.1f}s", flush=True)
    np.savetxt(folder / "errors.csv", table[:j+1], delimiter=",", comments="",
               header="steps,dt,mean_squared_L2_error,rms_L2_error,rate,mean_seconds")
    np.savez_compressed(
        folder / "errors.npz", step_counts=STEP_COUNTS, dt=dt_values,
        squared_errors=squared_errors, rms_errors=rms_errors, rates=rates,
        run_seconds=run_seconds, reference_seconds=reference_seconds,
        fine_increments=fine_increments, n=MESH_N, reference_steps=REFERENCE_STEPS,
        T=T, viscosity=VISCOSITY, sigma=SIGMA, seed=SEED,
        prediction_solver=PREDICTION_SOLVER,
        initial_data=getattr(INITIAL_DATA, "__name__", str(INITIAL_DATA)))

# 4. Plot all expected errors against the time step, and save the figure.
fig, ax = plt.subplots()
ax.loglog(dt_values, rms_errors, "o-", label="Terminal velocity RMS L2 error")
# Scale a reference slope to the coarsest measured error.
if rms_errors[0] > 0:
    guide = rms_errors[0] * (dt_values / dt_values[0])**REFERENCE_RATE
    ax.loglog(dt_values, guide, "k--",
              label=rf"Reference: $(\Delta t)^{{{REFERENCE_RATE:g}}}$")
ax.set_xlabel(r"$\Delta t$")
ax.set_ylabel(r"$(M^{-1}\sum_{s=1}^M\|u_N^{(s)}(T)-u_{ref}^{(s)}(T)\|_{L^2}^2)^{1/2}$")
ax.set_title(f"Time refinement: n={MESH_N}, nu={VISCOSITY:g}, sigma={SIGMA:g}, M={SAMPLES}")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(folder / "convergence.pdf")
fig.savefig(folder / "convergence.png", dpi=200)
print(f"Saved errors and plots in {folder}", flush=True)
plt.show()
