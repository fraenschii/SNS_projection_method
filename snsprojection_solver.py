"""Serial projection method solver that can be called from elsewhere
This is the scheme as in equations (3.17)-(3.18) in the paper
Taylor-Hood P1/P2 elements are used for the spatial discretization. 
Everything is based on Fenicsx.
The default initial datum is the rough vortex sheet. 
Choose initial_data="smooth_vortex" for the convergence rate experiments.
"""
from time import perf_counter
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np

# import stuff for mixed elements
from basix.ufl import mixed_element
from ufl import TrialFunctions, TestFunctions
from dolfinx.fem import locate_dofs_topological
from dolfinx.mesh import exterior_facet_indices
from dolfinx.fem.petsc import LinearProblem

from dolfinx.fem import (
    Constant,
    Function,
    extract_function_spaces,
    functionspace,
    assemble_scalar,
    dirichletbc,
    form,
    locate_dofs_geometrical,
    Expression,
)
from dolfinx.fem.petsc import (
    assemble_matrix,
    assemble_vector,
    apply_lifting,
    create_vector,
    set_bc,
)
from dolfinx.mesh import create_unit_square
from basix.ufl import element
from ufl import (
    TestFunction,
    TrialFunction,
    div,
    dot,
    dx,
    inner,
    lhs,
    nabla_grad,
    rhs,
)

# for vortex initial data, import necessary functions from vortex_dynamics_data.py
from vortex_dynamics_data import (
    resolve_bumps,
    normalization_scale,
    initial_velocity as vortex_expression,
)


# smooth initial data for convergence experiments
def smooth_initial_velocity(x):
    """curl(sin(pi*x)^2 sin(pi*y)^2); smooth, divergence-free, zero trace."""
    values = np.zeros((2, x.shape[1]), dtype=PETSc.ScalarType)
    values[0] = np.pi * np.sin(np.pi*x[0])**2 * np.sin(2*np.pi*x[1])
    values[1] = -np.pi * np.sin(2*np.pi*x[0]) * np.sin(np.pi*x[1])**2
    return values


# this is the main solver with default values, solves one path
def solve_projection(*, n=64, num_steps=2048, T=2.0, viscosity=0.001,
                     noise_amplitude=0.15, initial_data="vortex_sheet",
                     increments=None, seed=2026, domain=None,
                     snapshot_count=5, verbose=True, prediction_solver="iterative"):
    """
    Here:
    n: how many subdivisions in each direction (2*n*n triangular cells).
    domain is unit-square mesh.
        None creates a unit-square mesh using n subdivisions per direction.
    increments: shape (num_steps,), actual Brownian increments (variance dt),
        None generates a path using seed.
        For time refinement, SUM increments of a common fine path; using the
        same seed alone on different time grids does not couple the paths.
    initial_data: one of the four named profiles above or an interpolation
        callable accepting coordinates with shape (2, number_of_points).
    snapshot_count: number of equally spaced saved snapshots; zero disables.
    prediction_solver: "iterative" preserves the original BCGS/HYPRE solve;
        "direct" uses sparse LU for the same prediction equations. 

    Returns a dict containing final FE fields (velocity, tentative,
    pressure, vorticity), mesh, spaces, snapshots, increments, metrics.
    wall_seconds includes setup/JIT;
    time_loop_seconds measures the time loop. Both are local elapsed times.
    The body force is zero and the noise is sigma*u*dW.
    """
    
    
    if prediction_solver not in ("iterative", "direct"):
        raise ValueError("prediction_solver must be 'iterative' or 'direct'")
    
   
    started = perf_counter()
    mesh = create_unit_square(MPI.COMM_WORLD, n, n) if domain is None else domain
    t = 0.0
    dt = T / num_steps
    # create the brownian motion
    increments = (np.random.default_rng(seed).normal(0, np.sqrt(dt), num_steps)
                  if increments is None else np.asarray(increments, dtype=float))
    if increments.shape != (num_steps,) or not np.all(np.isfinite(increments)):
        raise ValueError(f"increments must have finite shape ({num_steps},)")
    increments = increments.copy()
    # Finite element spaces
    v_cg2 = element("Lagrange", mesh.basix_cell(), 2,shape=(mesh.geometry.dim,),)
    s_cg1 = element("Lagrange", mesh.basix_cell(), 1)

    # Function spaces
    V = functionspace(mesh, v_cg2)
    Q = functionspace(mesh, s_cg1)
    W = functionspace(mesh, mixed_element([v_cg2, s_cg1]))

    Vp, velocity_map = W.sub(0).collapse()
    Qp, pressure_map = W.sub(1).collapse()

    # Trial and test functions
    u = TrialFunction(V)
    v = TestFunction(V)
    up, pp = TrialFunctions(W)
    vp, qp = TestFunctions(W)


    # identify boundary of domain

    def boundary(x):
        return (np.isclose(x[0], 0) | np.isclose(x[0], 1)
                | np.isclose(x[1], 0) | np.isclose(x[1], 1))


    # homogeneous Dirichlet boundary conditions
    boundary_dofs = locate_dofs_geometrical(V, boundary)
    zero_velocity = Function(V)
    bcu = [dirichletbc(zero_velocity, boundary_dofs)]

    # define variables and constants
    u_n = Function(V)
    u_n.name = "u_n"
    u_tilde = Function(V)
    u_tilde.name = "u_tilde"
    f = Constant(mesh, PETSc.ScalarType((0, 0)))
    k = Constant(mesh, PETSc.ScalarType(dt))
    mu = Constant(mesh, PETSc.ScalarType(viscosity))
    dW = Constant(mesh, PETSc.ScalarType(0.0))


    # Interpolate the initial datum
    # no additional discrete projection is applied at t=0.
    if callable(initial_data):
        u_n.interpolate(initial_data)
    elif initial_data == "smooth_vortex":
        u_n.interpolate(smooth_initial_velocity)
    else:
        from ufl import SpatialCoordinate, as_vector
        if initial_data == "vortex_sheet":
            x = SpatialCoordinate(mesh)
            psi0 = (256 * x[0]**2 * (1-x[0])**2 * x[1]**2 * (1-x[1])**2 * abs(x[1]-0.5))
            u0 = as_vector((psi0.dx(1), -psi0.dx(0)))
        elif initial_data in ("same_sign", "mixed_sign"):
            bumps = resolve_bumps(initial_data)
            u0 = vortex_expression(mesh, bumps, normalization_scale(bumps))
        else:
            raise ValueError("Unknown initial_data; use smooth_vortex, vortex_sheet, "
                             "same_sign, mixed_sign, or a callable x -> values")
        u_n.interpolate(Expression(u0, V.element.interpolation_points, comm=mesh.comm))
    set_bc(u_n.x.petsc_vec, bcu)
    u_tilde.x.array[:] = u_n.x.array[:]


    # define variational problem for the first (prediction) step
    p_n = Function(Q)
    p_n.name = "p_n"

    F1 = inner((u - u_n), v) * dx
    F1 += k*inner(dot(u_tilde, nabla_grad(u)), v) * dx
    F1 += 0.5* k * div(u_tilde) * inner(u, v) * dx
    F1 += k*mu * inner(nabla_grad(u), nabla_grad(v)) * dx
    F1 -= k*inner(f, v) * dx
    F1 -= noise_amplitude * dW * inner(u_n, v) * dx(domain=mesh)

    a1 = form(lhs(F1))
    L1 = form(rhs(F1))
    A1 = assemble_matrix(a1, bcs=bcu)
    A1.assemble()
    b1 = create_vector(extract_function_spaces(L1))

    # projection step

    # Step 1 writes the tentative velocity into this function
    u_ = Function(V)

    # Homogeneous velocity boundary conditions in the mixed space
    fdim = mesh.topology.dim - 1
    mesh.topology.create_connectivity(fdim, mesh.topology.dim)
    facets = exterior_facet_indices(mesh.topology)

    mixed_dofs = locate_dofs_topological(W.sub(0), fdim, facets)
    zero_mixed = Function(W)
    bc_projection = dirichletbc(zero_mixed, mixed_dofs)

    # Define variational problem for projection step
    # We multiply the continuity equation by -k to make the matrix symmetric
    a_projection = (inner(up, vp) - k * pp * div(vp)- k * div(up) * qp) * dx

    L_projection = inner(u_, vp) * dx

    # Preconditioner only: this does not change the projection equations
    P_projection = (inner(up, vp) + k**2 * inner(nabla_grad(pp), nabla_grad(qp))) * dx

    mixed_solution = Function(W)

    projection = LinearProblem(a_projection,L_projection,P=P_projection,
        u=mixed_solution,bcs=[bc_projection],
        petsc_options_prefix="chorin_projection_",
        petsc_options={"ksp_type": "minres","pc_type": "jacobi",
                       "ksp_rtol": 1e-12,"ksp_atol": 1e-14,"ksp_max_it": 10000,
                       "ksp_error_if_not_converged": True,},
        )

    # Nullspace: zero velocity, spatially constant pressure
    null_vector = Function(W)
    null_vector.x.array[pressure_map] = 1.0
    null_vector.x.petsc_vec.normalize()

    pressure_nullspace = PETSc.NullSpace().create(vectors=[null_vector.x.petsc_vec], 
                                                  comm=mesh.comm)
    
    projection.A.setNullSpace(pressure_nullspace)
    projection.A.setTransposeNullSpace(pressure_nullspace)

    corrected = Function(Vp)
    projected_pressure = Function(Qp)

    volume = assemble_scalar(form(Constant(mesh, PETSc.ScalarType(1)) * dx))
    pressure_integral = form(p_n * dx)


    # Solver for step 1
    solver1 = PETSc.KSP().create(mesh.comm)
    solver1.setOperators(A1)
    pc1 = solver1.getPC()
    if prediction_solver == "direct":
        solver1.setType(PETSc.KSP.Type.PREONLY)
        pc1.setType(PETSc.PC.Type.LU)
    else:
        solver1.setType(PETSc.KSP.Type.BCGS)
        pc1.setType(PETSc.PC.Type.HYPRE)
        pc1.setHYPREType("boomeramg")

    # set up vorticity variable
    Vomega = functionspace(mesh, ("DG", 1))
    omega = Function(Vomega)
    omega.name = "vorticity"

    omega_expression = Expression(u_n[1].dx(0) - u_n[0].dx(1),
        Vomega.element.interpolation_points,comm=mesh.comm,)

    # Evenly spaced snapshots, including initial and final states when count >= 2.
    snapshot_steps = set(np.rint(np.linspace(0, num_steps, snapshot_count)).astype(int))
    snapshots = []

    def collect_snapshot(time):
        omega.interpolate(omega_expression)

        snapshots.append({"time": float(time),"velocity": u_n.x.array.real.copy(),
            "vorticity": omega.x.array.real.copy(),})

    if snapshot_count:
        collect_snapshot(t)

    fallback_solver = None
    prediction_fallbacks = 0
    loop_started = perf_counter()
    try:
        for i in range(num_steps):
            # Update current time step
            t += dt
            dW.value = PETSc.ScalarType(increments[i]) # update noise

            # Step 1: Tentative veolcity step
            # Update the matrix using the previous tentative velocity
            A1.zeroEntries()
            assemble_matrix(A1, a1, bcs=bcu)
            A1.assemble()

            # Assemble the right-hand side
            with b1.localForm() as loc_1:
                loc_1.set(0)
            assemble_vector(b1, L1)
            apply_lifting(b1, [a1], [bcu])
            set_bc(b1, bcu)
            solver1.solve(b1, u_.x.petsc_vec)
            active_solver = solver1

            # checking if the iterative solver doesn't converge
            if solver1.getConvergedReason() <= 0 and prediction_solver == "iterative":
                print(f"Prediction iteration failed at step {i + 1}; "
                      "retrying the same system with direct LU.", flush=True)
                if fallback_solver is None:
                    fallback_solver = PETSc.KSP().create(mesh.comm)
                    fallback_solver.setType(PETSc.KSP.Type.PREONLY)
                    fallback_solver.getPC().setType(PETSc.PC.Type.LU)
                fallback_solver.setOperators(A1)
                # Discard the failed iterate; reuse the assembled system and noise.
                u_.x.petsc_vec.set(0)
                fallback_solver.solve(b1, u_.x.petsc_vec)
                active_solver = fallback_solver
                prediction_fallbacks += 1
            if active_solver.getConvergedReason() <= 0:
                raise RuntimeError(
                    f"Prediction solver failed at step {i + 1} "
                    f"(direct LU after iterative failure if applicable): "
                    f"PETSc reason={int(active_solver.getConvergedReason())}, "
                    f"iterations={active_solver.getIterationNumber()}, "
                    f"residual={active_solver.getResidualNorm():.3e}, dt={dt:.3e}")
            u_tilde.x.array[:] = u_.x.array[:]

            # Step 2: Coupled projection
            projection.solve()

            # Extract corrected velocity and pressure
            corrected.x.array[:] = mixed_solution.x.array[velocity_map]
            projected_pressure.x.array[:] = mixed_solution.x.array[pressure_map]

            u_n.interpolate(corrected)
            p_n.interpolate(projected_pressure)

            # Enforce zero spatial mean pressure
            pressure_mean = assemble_scalar(pressure_integral) / volume

            p_n.x.array[:] -= pressure_mean

            # collect snapshots    
            if i + 1 in snapshot_steps:
                collect_snapshot(t)
            # Print progress every 20th step and at the last step.
            if verbose and ((i % 20 == 0) or (i == num_steps - 1)):
                print(f"Time {t:.2f}", flush=True)

        omega.interpolate(omega_expression)
    finally:
        b1.destroy()
        if fallback_solver is not None:
            fallback_solver.destroy()
        solver1.destroy()
        A1.destroy()
        del projection
        pressure_nullspace.destroy()
    time_loop_seconds = perf_counter() - loop_started
    wall_seconds = perf_counter() - started
    result = dict(velocity=u_n, tentative=u_tilde, pressure=p_n, vorticity=omega,
                  mesh=mesh, V=V, Q=Q, Vomega=Vomega, snapshots=snapshots,
                  increments=increments,
                  metrics=dict(n=n, num_steps=num_steps, T=T, dt=dt,
                               viscosity=viscosity, noise_amplitude=noise_amplitude,
                               prediction_solver=prediction_solver,
                               prediction_fallbacks=prediction_fallbacks,
                               wall_seconds=wall_seconds,
                               time_loop_seconds=time_loop_seconds))
    
    return result
