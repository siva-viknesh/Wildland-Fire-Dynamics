## Role of Flow Topology in Wind-Driven Wildfire Propagation

![Model](https://github.com/siva-viknesh/Wildland_Fire_Dynamics/blob/main/0_Scaling_Analysis/Scaling_analysis.jpg)

This repository contains a GPU-enabled wildfire transport solver developed to investigate how wind flow topology influences wildfire propagation. The project adopts a fundamental reactive flow dynamics perspective and introduces a physics-informed framework based on a nonlinear convection-diffusion-reaction (CDR) model.

### 🔥 Project Overview

Wildfire behavior results from intricate interactions among wind, terrain, and fuel, often producing highly nonlinear and transient dynamics. This study aims to uncover the influence of wind velocity topology—particularly the role of flow manifolds—on wildfire transport and spread.

Key contributions include:

- **Revised Non-Dimensionalization:**  
  A new scaling approach incorporating three characteristic time scales, leading to the derivation of two critical non-dimensional parameters. These provide improved insights over conventional single-time-scale models.

- **State-Neutral Curve Identification:**  
  Analytical determination of parameter thresholds where initial fire either dies out or persists, offering a predictive boundary in parameter space.

- **Wildfire Solver Development:**  
  A finite difference solver was implemented using upwind compact schemes and implicit-explicit Runge-Kutta methods to model wildfire transport over two-dimensional domains.

- **Flow Topology Influence (Steady Wind):**  
  Examined wildfire propagation under steady saddle-point wind fields, emphasizing the alignment between firefront evolution and the stable/unstable manifolds of the velocity field.

- **Transient Wind Topology (Double-Gyre):**  
  Investigated the impact of unsteady, time-periodic wind structures using a benchmark double-gyre flow. The wildfire response was quantified using a transfer function-based approach across various Strouhal numbers and amplitudes.

- **Lagrangian Coherent Structures (LCS):**  
  To assess the correspondence between coherent structures and firefront evolution, LCS fields were computed using the [TBarrier](https://github.com/haller-group/TBarrier) MATLAB toolbox by the Haller Group. The alignment between transport barriers and wildfire fronts was explored under both steady and unsteady wind conditions.

### 🚀 Code Highlights

- GPU-accelerated solver for efficient wildfire transport simulations.
- Supports both steady and time-dependent wind velocity fields.
- Modular design for integrating custom wind models and parameter sets.
- Post-processing utilities to evaluate front evolution and firefront–manifold interactions.
