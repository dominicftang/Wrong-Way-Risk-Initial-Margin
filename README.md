# Deconstructing Wrong-Way Risk in the Presence of Initial Margin

[![Degree](https://img.shields.io/badge/MSc-Financial_Mathematics_(UCL)-blue.svg)](https://www.ucl.ac.uk)
[![Grade](https://img.shields.io/badge/Grade-76%25_(Distinction)-green.svg)]()
[![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)]()

## Overview
This repository contains the dissertation and numerical code for my MSc Financial Mathematics dissertation at University College London (UCL).

The project investigates counterparty credit exposure under the BCBS-IOSCO uncleared margin rules. By engineering a scalable Monte Carlo simulation pipeline in Python, the research evaluates how trade flow spikes and correlated default intensity affect Credit Valuation Adjustment (CVA) in the presence of dynamic Initial Margin (IM).

---

## Key Highlights & Findings

### Context & Research Objectives
* **Real-World Motivation:** Inspired by structural vulnerabilities exposed during the 2021 collapse of Archegos Capital Management, this thesis investigates whether traditional Credit Valuation Adjustment (CVA) models can effectively handle leveraged Wrong-Way Risk (WWR) from counterparties.
* **Core Hypothesis:** Evaluated whether the market already prices in systemic tail-risk through its observed hazard rate ($\lambda^*$), proving that a naive application of traditional WWR models systematically overestimates risk by "double-counting" embedded market parameters.
* **Key Finding:** Demonstrated that while naive models significantly overestimate exposure, CVA for collateralised positions remains several times higher when accounting for true WWR, proving that structural correlations across global risks serve as a dominant, non-negligible risk factor.

### Technical Implementation & Methodology
* **Stochastic Modelling Pipeline:** Engineered a scalable 200,000-path Monte Carlo simulation engine in Python modelling portfolio increments across Geometric Brownian Motion (GBM), Heston Stochastic Volatility, and Merton Jump-Diffusion frameworks.
* **Algorithmic Stability:** Implemented a **Full Truncation Scheme** to resolve numerical instability and prevent invalid negative variances when Feller conditions are violated in high-volatility regimes.
* **Parameter Calibration:** Applied **Brent’s root-finding method** to numerically solve the target objective function $g(\lambda^*) = CVA_{\text{No IM}}(\lambda^*, \rho) - CVA^* = 0$, isolating recalibrated hazard rates across varying correlation spectrums ($\rho \in [0, 1]$).
* **Initial Margin (IM) Dynamics:** Evaluated 99% 10-day VaR IM rules (UMR), demonstrating that while IM suppresses baseline Expected Exposure between cash flow dates by over two orders of magnitude, it fails to mitigate trade-flow payment spikes, which ultimately dominate residual CVA.

---

## Repository Structure
* [`Dissertation_Dominic_Tang.pdf`](./Dissertation_Dominic_Tang.pdf) — Full MSc thesis paper.
* [`src/`](./src/) — Python source code for Monte Carlo simulation routines, stochastic models, and CVA amplification calculators.

---

## Author
**Dominic Tang**  
* MSc Financial Mathematics, University College London (UCL)  
* BSc Mathematics (1st Class Hons), University of Warwick  
* Email: dominicftang@gmail.com | [LinkedIn](https://www.linkedin.com)
