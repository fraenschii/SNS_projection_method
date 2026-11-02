# SNS_projection_method
Fenicsx code to run simulations for the stochastic Navier-Stokes equations discretized with a finite element projection method

Implementation of the Chorin projection method scheme with lowest order (P2/P1) Taylor-Hood finite elements for the space discretization. The scheme is

### Step 1: Prediction

For each $m \in \{0,1,\dots,N-1\}$, given
$u_h^m,\widetilde{u}_h^m \in U_h$, find
$\widetilde{u}_h^{m+1} \in U_h$ such that, for all $v \in U_h$,

$$
\begin{aligned}
&\left(\widetilde{u}_h^{m+1}-u_h^m,v\right)
+ \Delta t\,b\left(\widetilde{u}_h^m,\widetilde{u}_h^{m+1},v\right)
\\
&\qquad
+ \Delta t\,\nu\left(\nabla\widetilde{u}_h^{m+1},\nabla v\right)
\\
&=
\Delta t\,\langle f^m,v\rangle
+ \left(\mathcal{P}_{U_h}G(u_h^m)\Delta W_m,v\right).
\end{aligned}
$$

### Step 2: Projection

Find $(u_h^{m+1},p_h^{m+1}) \in U_h \times P_h$ such that,
for all $(v,q) \in U_h \times P_h$,

$$
\begin{aligned}
\left(
\frac{u_h^{m+1}-\widetilde{u}_h^{m+1}}{\Delta t},v
\right)
&=
\left(p_h^{m+1},\nabla\cdot v\right),
\\
\left(\nabla\cdot u_h^{m+1},q\right)
&=0.
\end{aligned}
$$

Code largely written with Codex/ChatGPT.
