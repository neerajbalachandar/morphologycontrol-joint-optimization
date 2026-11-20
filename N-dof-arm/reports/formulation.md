# Morphology and finite-time capability formulation

## 1. Study focus and configuration

The primary experiment compares general finite-time end-effector capability
across robot morphologies. The prescribed-trajectory controller is a separate
comparison. All new runs use the settings in `config.py`: baseline links
`[2, 2] m`, fixed total length `L0=4 m`, fixed total mass `M=2 kg`,
`N=1,...,5`, link bounds `[0.2,4] m`, gravity `9.81 m/s^2`, viscous damping
`0.05`, and horizon `T=2 s`.

The finite-p support-function and costate framing follows the ideas in Nguyen,
Cortés, and Kramer, “Reachability Analysis for Design Optimization,” ACC
2026. The projected multi-input support calculation below is stated directly
for this model and does not presume the paper's special planar, single-input
one-switch boundary theorem.

The primary actuator resource is a fixed total bound `sum_i tau_max,i=2 Nm`,
with `tau_max,i=2/N Nm`. The separate sensitivity convention uses
`tau_max,i=1 Nm` for every motor. A result records the selected convention.
The latter increases aggregate authority with N and must not be mixed with the
primary comparison.

The common dimensionless design proxy is

\[
J_m=w_L L/L_0+w_N N/N_0.
\]

The fixed-total-length experiment has `L=L0`, so the link term is constant and
the variable part measures joint/actuator count. This is a relative study
proxy, not a measured manufacturing price. Length nonuniformity is reported
only as a morphology descriptor, not charged as physical manufacturing cost.

## 2. Rigid-link mechanics

Angles are relative joint angles `q_i`; absolute link-i angle is
`theta_i=sum_{j<=i} q_j`. Link i is a uniform slender rigid rod with
`m_i=(M/L0) l_i`, centre of mass at `l_i/2`, and planar inertia
`I_i=m_i l_i^2/12`. The centre-of-mass translational Jacobian has horizontal
entry `Jv_x[j]=sum_{r=j}^{i-1} l_r+l_i/2` for `j<=i`; its vertical row is zero
at the downward equilibrium. The angular Jacobian is one for every upstream
joint. The mass matrix assembled in `model.py` is

\[
M(q=0)=\sum_i(m_iJ_{v,i}^T J_{v,i}+I_iJ_{\omega,i}^T J_{\omega,i}).
\]

It is symmetric and positive definite for positive link lengths and masses.
The gravitational stiffness is the Hessian of
`V=sum_i m_i g y_com,i` at the stable downward equilibrium. With diagonal
viscous damping `D=d I`, the linearized model is

\[
M\ddot q+D\dot q+Kq=u,\quad
A=\begin{bmatrix}0&I\\-M^{-1}K&-M^{-1}D\end{bmatrix},\quad
B=\begin{bmatrix}0\\M^{-1}\end{bmatrix}.
\]

For N=1 this gives `M=ml^2/3`, `K=mgl/2`, and the analytical A and B
specified in the experiment brief. This is a serial rigid-link benchmark; it
is not an exact continuum soft-robot model. A piecewise-curvature continuum
model remains a future extension.

## 3. Common output projection

The state is `x=[q; qdot] in R^(2N)`. At the downward equilibrium the
horizontal end-effector Jacobian is
`J=[sum_{r=j}^N l_r]_(j=1..N)`, and

\[
y(T)=C_N(l)x(T)=[x_{EE}(T),\dot x_{EE}(T)]^T.
\]

All N values are therefore compared in the same two-dimensional physical
output space. Geometry plots and primary area criteria use

\[
\tilde y=[x_{EE}/L_0,\;T\dot x_{EE}/L_0]^T.
\]

Physical outputs remain available through C and the tracking data. No
full-state volume is compared across different state dimensions. The current
linearization uses horizontal displacement; it does not claim a full
nonlinear planar task-space reachable set.

## 4. Input sets and projected support

Let `D_tau=diag(tau_max)` and `u=D_tau v`. The primary bounded-magnitude set
uses `||v||_(L_inf)<=1`. With

\[
G(t)=S C e^{A(T-t)}B D_\tau,
\]
where S is the output scaling, the projected support is

\[
h_\infty(s)=\int_0^T\|G(t)^Ts\|_1dt.
\]

This support calculation is valid for any N and multiple inputs. The special
one-switch planar single-input theorem is not assumed for this model.

For finite p in `{2,4,6,10}`, with conjugate exponent `q=p/(p-1)`,

\[
h_p(s)=c_p\left[\int_0^T\sum_i|(G(t)^Ts)_i|^qdt\right]^{1/q}.
\]

`raw_paper` sets `c_p=1`. `peak_equivalent` uses
`c_p=(NT)^(1/p)`, making the p-norm RMS per actuator and time equal to one.
Peak-equivalent is the primary comparison to the unit L-infinity box; raw
paper-norm results are a separately labelled sensitivity result. The budgets
are different and the two families must not be merged in one unlabelled
comparison.

The sampled boundary is obtained from extremizing controls. For finite p,

\[
v_p^*=c_p H_p^{1-q}\,\mathrm{sign}(G^Ts)\odot|G^Ts|^{q-1},
\]

and for p=infinity it is `sign(G^T s)`. Integrating `G(t)v*(t)` yields output
points; their convex hull estimates the area. The area is a polygonal numerical
estimate. Angular and time resolution convergence is visualized. Area does
not capture directional weakness, so support functions are also plotted.

## 5. L2 Gramian

For p=2, `q=2` and

\[
h_2(s)=c_2\sqrt{s^TW_y s},\qquad
W_y=\int_0^T G(t)G(t)^Tdt.
\]

This is the output-space energy Gramian under the normalized actuator map.
Its ellipsoid area is `pi*c_2^2*sqrt(det(W_y))`. The numerical boundary area
and analytical ellipse area are both stored for comparison. The L2 trace,
L6 area, and L-infinity area are optimized as distinct objectives.

## 6. Costate and zonotope references

The optional low-dimensional PMP reference samples initial costates using a
scrambled Sobol sequence in dimension `2N`, normalizes directions, and sweeps
amplitudes logarithmically relative to `||B D_tau||_F T`. For each costate it
uses the finite-p extremal switching relation, calculates its input norm, and
keeps the raw terminal point for inside/outside classification; only accepted
points enter the sampled hull. It propagates the held midpoint controls through
an exact-ZOH discretization of the linear system. Sampling becomes less
complete as 2N grows; this method is a validation visualization, not the
primary optimization objective.

The optional forward zonotope uses exact zero-order-hold discretization and
`Z={c+G xi: ||xi||_inf<=1}`. Generator reduction retains large generators and
replaces discarded generators by an axis-aligned enclosing box. This proves
outer inclusion for the discretized held-input reachable set. It does not
assert exact inclusion for arbitrary continuous-time switching between grid
points. Generator count and temporal discretization are recorded. No CORA
dependency is required.

## 7. Morphology optimization

Every fixed N obeys `sum_i l_i=L0` and `l_min<=l_i<=l_max`. A fixed-N inner
problem optimizes the first N-1 variables and reconstructs the final length
from the equality. Differential evolution uses feasible projected initial
populations and deterministic seeds. The outer problem enumerates all
feasible integer N and compares each independently optimized result. This is
the authoritative nested discrete-continuous workflow. Stochastic optimizer
success is numerical termination status, not proof of a global optimum.

Minimum-cost capability runs check gains 5%, 10%, 20%, and 30% over the
baseline L-infinity area. Since the design proxy varies with N only, the
selected design uses the minimum feasible N and then the largest capability
margin at that N. The mixed-integer padded scripts are retained as legacy
reference material and are not part of the new default command sequence.

## 8. Prescribed tracking comparison

Tracking is a separate experiment with the same physical reference
`x_d=x0+A sin(omega t)`, its analytic velocity, common horizon and fixed-total
actuator limits. The discretized linearized dynamics are condensed into a
finite-horizon convex quadratic objective in sampled torques, solved with
box-constrained L-BFGS-B. It minimizes output error and control effort; it
does not clip an unconstrained LQT controller. The task morphology minimizes
this task objective across feasible N after a fixed-N link search. This QP is
for the linearized horizontal output. `model.kinematic_points` supplies full
planar forward kinematics for morphology visualization, but a nonlinear
task-space dynamics/controller is not claimed here.

## 9. Experiments, outputs, and limits

The experiment suite contains: one-link length sensitivity at fixed total
mass (therefore variable linear density); Lp versus L-infinity boundaries;
support functions; p-to-infinity and numerical-resolution convergence;
extremizing torque histories; N=1,2 costate checks; optional outer zonotope;
fixed-N redistribution samples; three nested capability objectives; four
minimum-cost targets; and separate tracking results. Every run has a
configuration snapshot and is saved to an explicitly named directory under
`results/`. Numerical arrays are in NPZ; metadata and optimizer diagnostics
are in JSON. PNG and PDF figures are written under that run's `plots/` folder.

This benchmark omits nonlinear Coriolis terms away from equilibrium, joint
limits, actuator dynamics, collision constraints, flexible-link deformation,
and hardware-calibrated costs. Those are required before treating the proxy
as a physical design recommendation. Continuum soft-arm modeling is a
separate extension.
