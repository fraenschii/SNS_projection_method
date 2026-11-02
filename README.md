# SNS_projection_method
Fenicsx code to run simulations for the stochastic Navier-Stokes equations with multiplicative noise discretized with a finite element projection method.
The equations are
```math
du +((u\cdot \nabla) u +\nabla p -\nu \Delta u-f) dt = G(u) dW,\quad \mathrm{div} \ u = 0
```

Implementation of the Chorin projection method scheme with lowest order (P2/P1) Taylor-Hood finite elements for the space discretization. The scheme consists of the following two steps. Here $U_h$ is the finite element space for the velocity approximations and $Q_h$ the space for the pressure approximation.

### Step 1: Prediction

For each $m \in \{0,1,\dots,N-1\}$, given
$u_h^m,\widetilde{u}_h^m \in U_h$, find
$\widetilde{u}_h^{m+1} \in U_h$ such that, for all $v \in U_h$,

```math
\left(\widetilde{u}_h^{m+1}-u_h^m,v\right) + \Delta t\,b\left(\widetilde{u}_h^m,\widetilde{u}_h^{m+1},v\right) + \Delta t\,\nu\left(\nabla\widetilde{u}_h^{m+1},\nabla v\right) = \Delta t\,\langle f^m,v\rangle + \left(\mathcal{P}_{U_h}G(u_h^m)\Delta W_m,v\right).
```

### Step 2: Projection

Find $(u_h^{m+1},p_h^{m+1}) \in U_h \times Q_h$ such that,
for all $(v,q) \in U_h \times Q_h$,

$$
\begin{aligned}
\left(
\frac{u_h^{m+1}-\widetilde{u}_h^{m+1}}{\Delta t},v
\right)
&=
\left(p_h^{m+1},\mathrm{div}\ v\right),
\\
\left(\mathrm{div} \ u_h^{m+1},q\right)
&=0.
\end{aligned}
$$

Code largely written with Codex/ChatGPT.
