<h1 align="center">🔮 Probabilistic Models & Bayesian Inference for Telco Customer Churn</h1>

<p align="center">
  <img src="https://img.shields.io/badge/PyMC-Bayesian%20Inference-blueviolet?style=for-the-badge&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/ArviZ-Posterior%20Diagnostics-blue?style=for-the-badge">
  <img src="https://img.shields.io/badge/pgmpy-Probabilistic%20Graphical%20Models-green?style=for-the-badge">
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge">
</p>

<p align="center">
  <strong>
    A probabilistic machine learning framework for customer churn prediction using Bayesian inference, uncertainty quantification, and graphical models.
  </strong>
</p>

---

# 📌 Project Overview

Traditional churn prediction systems typically produce deterministic outputs, such as:

> "Customer A has a 78% chance of churning."

However, these predictions do not express **how certain the model is** about that estimate.

This project adopts a **Bayesian Machine Learning** approach to:

- Quantify prediction uncertainty
- Continuously update beliefs as new customer data arrives
- Discover probabilistic dependencies between customer behaviors
- Build interpretable risk models for decision-making

The project uses the **Telco Customer Churn Dataset** to demonstrate practical applications of:

- Bayesian Statistics
- Sequential Learning
- Probabilistic Graphical Models (PGMs)
- Bayesian Logistic Regression
- MCMC Sampling and Diagnostics

---

# 🚀 Key Features

### 📊 Uncertainty Quantification
Move beyond point estimates by estimating complete posterior distributions of model parameters.

### 🔄 Sequential Bayesian Updating
Update churn probabilities dynamically using Beta-Binomial conjugate models.

### 🕸️ Probabilistic Graphical Models
Model causal and dependency relationships between customer attributes and churn behavior.

### ⚙️ Bayesian Logistic Regression
Build interpretable Bayesian classifiers using **PyMC** and **NUTS sampling**.

### 🔍 Posterior Diagnostics
Evaluate sampler quality using:

- Gelman-Rubin Statistic ($\hat{R}$)
- Effective Sample Size (ESS)
- Trace plots
- Autocorrelation analysis
- Divergence checks

---

# 🏗️ Project Structure

```text
📦 telco-churn-bayesian-inference
│
├── W6_Probabilistic_Models_Assignment.ipynb
│   └── Main notebook containing all experiments
│
├── Telco-Customer-Churn.csv
│   └── Dataset (7043 customers × 21 variables)
│
├── telco_bayes_lr_v1.pkl
│   └── Serialized posterior trace
│
└── README.md
```

---

# ⚙️ Pipeline Architecture

```mermaid
graph LR
    A[Parameter Estimation]
    B[Sequential Updating]
    C[Graphical Models]
    D[Bayesian Logistic Regression]
    E[Posterior Diagnostics]

    A --> B
    B --> C
    C --> D
    D --> E
```

---

# 📚 Methodology

---

## 1️⃣ Frequentist vs Bayesian Estimation

Compare traditional statistical estimation techniques with Bayesian approaches.

### Frequentist Approach
- Maximum Likelihood Estimation (MLE)
- Point estimates only
- No uncertainty representation

### Bayesian Approach
- Prior distributions
- Posterior distributions
- Credible intervals
- Parameter uncertainty estimation

---

## 2️⃣ Beta-Binomial Sequential Updating

A conjugate Bayesian model is implemented to continuously update churn beliefs.

### Workflow

1. Define prior belief:

```math
p \sim Beta(\alpha,\beta)
```

2. Observe customer outcomes.

3. Update posterior:

```math
p|data \sim Beta(\alpha+x,\beta+n-x)
```

### Benefits

- Online learning capability
- Real-time probability updates
- Reduced uncertainty with increasing observations

---

## 3️⃣ Probabilistic Graphical Models (PGMs)

Construct Bayesian Networks to model relationships among variables such as:

- Contract Type
- Payment Method
- Internet Service
- Tenure
- Churn

### Objectives

- Discover conditional dependencies
- Perform probabilistic inference
- Analyze causal relationships

Tools used:

- `pgmpy`
- Variable Elimination
- Bayesian Networks
- Markov Random Fields (MRFs)

---

## 4️⃣ Bayesian Logistic Regression

A Bayesian Generalized Linear Model (GLM) is implemented using **PyMC**.

### Model Components

#### Priors

```math
\beta_i \sim Normal(0,\sigma)
```

#### Likelihood

```math
y_i \sim Bernoulli(p_i)
```

where:

```math
p_i = sigmoid(X_i\beta)
```

### Sampling Method

- Markov Chain Monte Carlo (MCMC)
- No-U-Turn Sampler (NUTS)

### Outputs

- Posterior parameter distributions
- Credible intervals
- Churn probability estimates
- Parameter importance analysis

---

# 📈 Posterior Diagnostics

Posterior quality is assessed using **ArviZ**.

Metrics include:

| Metric | Purpose |
|---------|----------|
| $\hat{R}$ | Convergence diagnostic |
| ESS | Effective sample size |
| Trace Plots | Mixing assessment |
| Energy Plots | Geometry diagnostics |
| Divergences | Sampling stability |

Ideal conditions:

```text
R̂ ≈ 1.00
No divergences
High ESS
```

---

# 📊 Dataset Information

### Dataset
Telco Customer Churn Dataset

| Attribute | Value |
|------------|--------|
| Records | 7,043 |
| Features | 21 |
| Target | Churn |
| Type | Binary Classification |
| Missing Values | Empty strings in numerical columns |

### Example Variables

- Gender
- SeniorCitizen
- Partner
- Dependents
- Tenure
- Contract
- MonthlyCharges
- TotalCharges
- PaymentMethod
- Churn

---

# 🚀 Installation

## Clone Repository

```bash
git clone https://github.com/your-username/telco-churn-bayesian-inference.git

cd telco-churn-bayesian-inference
```

---

## Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux/Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install pymc arviz pgmpy scikit-learn scipy statsmodels pandas numpy matplotlib seaborn
```

---

# ▶️ Running the Project

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
W6_Probabilistic_Models_Assignment.ipynb
```

Execute notebook cells sequentially.

---

# 📦 Main Libraries Used

| Library | Purpose |
|----------|----------|
| PyMC | Bayesian Modeling |
| ArviZ | Posterior Analysis |
| pgmpy | Graphical Models |
| Scikit-learn | Data Preprocessing |
| Pandas | Data Manipulation |
| NumPy | Numerical Computing |
| Matplotlib | Visualization |

---

# 🎯 Learning Outcomes

Through this project, you will learn:

✅ Bayesian parameter estimation

✅ Prior and posterior modeling

✅ Sequential Bayesian updating

✅ Probabilistic graphical models

✅ MCMC and NUTS sampling

✅ Posterior diagnostics

✅ Interpretable uncertainty-aware churn prediction systems

---

# 💡 Future Improvements

- Hierarchical Bayesian Models
- Bayesian Neural Networks
- Time-dependent churn modeling
- Real-time streaming inference
- Causal Bayesian Networks
- Deployment using FastAPI

---

# 📜 License

This project is licensed under the **MIT License**.

---

# 👨‍💻 Author

**Sandesh Bohara**

BSc (Hons) Information Technology  
Techspire College | APIIT / APU  
Aspiring Cybersecurity Engineer & AI Enthusiast

---

<p align="center">
⭐ If you found this project useful, consider giving it a star.
</p>