# Probabilistic Models & Bayesian Inference: Telco Churn Project

[![Bayesian Inference](https://img.shields.io/badge/Inference-PyMC-blueviolet?style=flat-square&logo=python)](https://www.pymc.io/)
[![Visualization](https://img.shields.io/badge/Diagnostics-ArviZ-blue?style=flat-square)](https://arviz-devs.github.io/arviz/)
[![Probabilistic Graphical Models](https://img.shields.io/badge/PGM-pgmpy**-green?style=flat-square)](https://pgmpy.org/)

This repository provides an advanced, hands-on framework for probabilistic modeling, continuous uncertainty quantification, and Bayesian inference applied to telecom customer churn analysis. 

---

# 📌 Project Overview

In traditional machine learning pipelines, models output deterministic point predictions (e.g., a single probability score). In high-stakes business deployment, understanding the **epistemic uncertainty** around those predictions is crucial for managing financial risk.

This project shifts from rigid frequentist modeling to a comprehensive Bayesian paradigm. By evaluating customer behavior, contract configurations, and historical usage metrics, this workflow maps out parameter distributions, updates beliefs sequentially as new streaming data arrives, and isolates complex causal dependencies using graphical models.

---

## ⚡ Technical Core & Features

* **Quantifying Parameter Uncertainty:** Move beyond single-point estimates like Maximum Likelihood Estimation (MLE) by exploring Maximum A Posteriori (MAP) and full Bayesian posteriors that natively expose risk parameters.
* **Sequential Learning Systems:** Implement conjugate Beta-Binomial update loops to simulate streaming real-time data environments where customer risk profiles adapt dynamically with every new transaction.
* **Causal Dependency Mapping:** Construct Probabilistic Graphical Models (PGMs) to explore conditional independence structures inside Directed Acyclic Graphs (Bayesian Networks) and Undirected Networks (Markov Random Fields).
* **Modern MCMC Sampling Stack:** Write, tune, and evaluate Generalized Linear Models (GLMs) using Markov Chain Monte Carlo (MCMC) sampling strategies inside **PyMC**.
* **Rigorous Sampler Diagnostics:** Run convergence diagnostics like $\hat{R}$ (Gelman-Rubin), Effective Sample Size (ESS), and energy isolation plots using **ArviZ** to ensure absolute posterior validity.

---

## 📂 Repository Structure

* **`W6_Probabilistic_Models_Assignment.ipynb`** — The primary architecture workspace containing mathematical formulations, code implementations, MCMC sampler configurations, and diagnostic validation.
* **`Telco-Customer-Churn.csv`** — The baseline dataset profiling 7,043 historical customer accounts across 21 feature fields.
* **`telco_bayes_lr_v1.pkl`** — A serialized trace and model artifact storing the post-MCMC sampling state for downstream inference deployment.

---

## ⚙️ Core Pipeline Architecture

The engine is organized into five progressive pillars of probabilistic engineering:

```mermaid
graph TD
    A[1. Frequentist vs. Bayesian Estimation] --> B[2. Sequential Bayesian Updating]
    B --> C[3. Probabilistic Graphical Models]
    C --> D[4. MCMC Bayesian Logistic Regression]
    D --> E[5. Trace Diagnostics & Inference]