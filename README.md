<h1 align="center">🔮 Probabilistic Models & Bayesian Inference: Telco Churn Project</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Inference-PyMC-blueviolet?style=for-the-badge&logo=python&logoColor=white" alt="PyMC">
  <img src="https://img.shields.io/badge/Diagnostics-ArviZ-blue?style=for-the-badge" alt="ArviZ">
  <img src="https://img.shields.io/badge/PGM-pgmpy-green?style=for-the-badge" alt="pgmpy">
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="MIT License">
</p>

<p align="center">
  <strong>An advanced, production-grade framework for probabilistic machine learning, continuous uncertainty quantification, and structural Bayesian inference optimized for enterprise risk management.</strong>
</p>

---

## 📌 Project Overview

In traditional enterprise machine learning pipelines, classification systems yield static, deterministic point predictions (e.g., a rigid, single-point probability calculation). In high-value customer retention environments, tracking the **epistemic uncertainty** surrounding those risk assumptions is crucial for optimal capital allocation.

This toolkit establishes a mathematically rigorous transition from frequentist engineering to a native **Bayesian paradigm**. By profiling structured behavioral matrices, contract typologies, and engagement data, this framework extracts comprehensive continuous parameter posteriors, updates model beliefs over streaming real-time pipelines, and maps structural causal dependencies through advanced graphical topologies.

---

## ⚡ Technical Core & Features

* 📊 **Parameter Uncertainty Quantification:** Circumvent overconfident Maximum Likelihood Estimations (MLE). This ecosystem leverages Maximum A Posteriori (MAP) optimizations alongside full Bayesian joint posteriors to expose parameter variance.
* 🔄 **Conjugate Sequential Learning:** Architect live tracking systems utilizing custom Beta-Binomial updating architectures, allowing model expectations to adapt dynamically as operational transaction signals arrive.
* 🕸️ **Causal Topology Mapping:** Deploy complex Probabilistic Graphical Models (PGMs) to map out exact directed independence constraints within Directed Acyclic Graphs (Bayesian Networks) and undirected frameworks (Markov Random Fields).
* ⚙️ **Production MCMC Engine:** Construct and scale Bayesian Generalized Linear Models (GLMs) driven by custom No-U-Turn Samplers (NUTS) and Markov Chain Monte Carlo (MCMC) protocols via **PyMC**.
* 🔍 **Airtight Chain Diagnostics:** Run continuous integration quality checks on backend sampler geometry evaluating Gelman-Rubin convergence ($\hat{R} \to 1.0$), Effective Sample Size (ESS), and energy isolation divergence boundaries via **ArviZ**.

---

## 📂 System Architecture

```text
├── W6_Probabilistic_Models_Assignment.ipynb  # Main structural notebook, modeling workflows, & diagnostics
├── Telco-Customer-Churn.csv                  # Standard operational dataset (7,043 customer accounts × 21 variables)
└── telco_bayes_lr_v1.pkl                     # Frozen post-MCMC serialized sampling trace for production inference
