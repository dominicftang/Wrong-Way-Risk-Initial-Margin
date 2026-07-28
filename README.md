# Deconstructing Wrong-Way Risk in the Presence of Initial Margin

[![Degree](https://img.shields.io/badge/MSc-Financial_Mathematics_(UCL)-blue.svg)](https://www.ucl.ac.uk)
[![Grade](https://img.shields.io/badge/Grade-76%25_(Distinction)-green.svg)]()
[![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)]()

## Overview
This repository contains the full dissertation and numerical code for my MSc Financial Mathematics dissertation at University College London (UCL), awarded a **76% Distinction**.

The project investigates counterparty credit exposure under the BCBS-IOSCO uncleared margin rules (UMR). By engineering a scalable Monte Carlo simulation pipeline in Python, the research evaluates how trade flow spikes and correlated default intensity affect Credit Valuation Adjustment (CVA) in the presence of dynamic Initial Margin (IM).

---

## Key Highlights & Findings
* **Simulation Engine:** Built a 200,000-path Monte Carlo engine in Python modeling discrete pricing increments across geometric Brownian motion (GBM), Heston Stochastic Volatility, and Merton Jump-Diffusion frameworks.
* **Numerical Stability:** Applied the **Full Truncation Scheme** to resolve algorithmic instability and prevent invalid negative variances under violated Feller conditions during high-volatility regimes.
* **Parameter Isolation:** Utilized **Brent’s root-finding method** to isolate embedded risk parameters, isolating the "Equivalent Hazard Rate" ($\lambda^*$).
* **Core Result:** Demonstrated that while 99% 10-day VaR IM suppresses baseline Expected Exposure (EE) between cash flow dates by over two orders of magnitude, it fails to effectively suppress trade-flow payment spikes, causing spikes to dominate total CVA.

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
