## Role of Flow Topology in Wind-Driven Wildfire Propagation

This repository contains a GPU-enabled wildfire transport solver developed to investigate how wind flow topology influences wildfire propagation. The project adopts a fundamental reactive flow dynamics perspective, based on a nonlinear convection-diffusion-reaction (CDR) PDE model.

### 🔥 Project Overview

Wildfire behavior results from intricate interactions among wind, terrain, and fuel, often producing highly nonlinear and transient dynamics. This study aims to uncover the influence of wind velocity topology—particularly the role of flow manifolds—on wildfire transport and spread.

#### Key Contributions

- **Revised Non-Dimensionalization:**  
  Introduces a revised wildfire model incorporating three characteristic time scales—convection, diffusion, and reaction—leading to the identification of two key non-dimensional numbers: the Damköhler number and a newly defined number, $\Phi$, representing the ratio of Damköhler to Péclet number. This contrasts with conventional approaches that rely on a single temporal scale.

- **State-Neutral Curve Identification:**  
  Analytically determines the critical conditions under which initial wildfires extinguish or propagate, culminating in a predictive *state-neutral curve* defined in the space of the two identified non-dimensional numbers.

- **Wildfire Solver Development:**  
  A GPU-enabled finite-difference solver in Python was developed using upwind compact schemes and implicit-explicit Runge-Kutta (IMEX-RK) methods to solve the wildfire transport PDE system.

- **Steady Wind Topology Analysis:**  
  Investigates wildfire behavior under steady wind conditions modeled by saddle-type fixed-point flows, with a focus on how firefronts align with the stable and unstable manifolds of the velocity field.

- **Unsteady Wind Influence (Double-Gyre):**  
  Analyzes wildfire spread under unsteady, time-periodic wind fields (double-gyre), characterizing the wildfire response through a transfer function (Bode plot) analysis across varying Strouhal numbers and wind oscillation amplitudes.

- **Lagrangian Coherent Structures (LCS):**  
   To assess the correspondence between coherent structures and firefront evolution, LCS fields were computed using the [TBarrier](https://github.com/haller-group/TBarrier) toolbox by the Haller ETH Group.

### 🚀 Code Highlights

- GPU-accelerated solver for efficient wildfire transport simulations.
- Supports both steady and unsteady/custom wind fields.
- Post-processing utilities to evaluate front evolution and firefront–manifold interactions.

### 👥 Authors & Affiliations

- **Siva Viknesh**, **Rob Stoll**, **Amirhossein Arzani** – University of Utah  
- **Ali Tohidi** – University of Maryland  
- **Fatemeh Afghah** – Clemson University

---

### 📄 Citation

If you use this code or find this work helpful, please cite:

Viknesh, S., Tohidi, A., Afghah, F., Stoll, R., & Arzani, A. (2025). Role of flow topology in wind-driven wildfire propagation. *Physics of Fluids*, 37(7), 076608. https://doi.org/10.1063/5.0268416

```bibtex
@article{viknesh_wildfire_2025,
  author  = {Viknesh, Siva and Tohidi, Ali and Afghah, Fatemeh and Stoll, Rob and Arzani, Amirhossein},
  title   = {Role of flow topology in wind-driven wildfire propagation},
  journal = {Physics of Fluids},
  volume  = {37},
  number  = {7},
  pages   = {076608},
  year    = {2025},
  doi     = {10.1063/5.0268416},
  url     = {https://doi.org/10.1063/5.0268416}
}
```
