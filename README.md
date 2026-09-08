# BIG DATA — Customer Intelligence PRO

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVTruong473/BIG-DATA/blob/main/BIG_DATA_Customer_Intelligence_PRO_Colab.ipynb)

> **Single-notebook Big Data & Customer Intelligence demo** built with Apache Spark, SVD, CURE clustering, FP-Growth and recommendation models. The full processing flow, evaluation, visualizations, dashboards and explanations are visible directly inside one Google Colab notebook.

## Overview

This project turns several Big Data / Machine Learning tasks into one end-to-end **Customer Intelligence pipeline** that can be inspected, executed and demonstrated from a single notebook.

The main goals are:

- keep the required algorithms visible and auditable;
- evaluate models with meaningful metrics instead of reporting one isolated score;
- demonstrate both **model performance** and **Spark/system performance**;
- convert technical outputs into interpretable customer/business insights;
- provide interactive visualizations suitable for lecturers, reviewers, recruiters and demo customers;
- keep the project runnable in **Google Colab** with minimal setup.

## Main Notebook

### `BIG_DATA_Customer_Intelligence_PRO_Colab.ipynb`

This is the **recommended and primary entry point** for the repository.

All important code is written directly inside the notebook. It does **not** depend on separate project `.py` pipeline files, so a reviewer can follow the complete workflow from top to bottom.

➡️ **Open directly in Google Colab:**

https://colab.research.google.com/github/NVTruong473/BIG-DATA/blob/main/BIG_DATA_Customer_Intelligence_PRO_Colab.ipynb

Or open the notebook on GitHub:

https://github.com/NVTruong473/BIG-DATA/blob/main/BIG_DATA_Customer_Intelligence_PRO_Colab.ipynb

---

## What the Project Does

The project is organized as a complete analytical pipeline:

```text
Raw Data
   ↓
Data Quality & Profiling
   ↓
Spark RDD / DataFrame Analytics
   ↓
RFV Customer Analytics
   ↓
Market Basket Intelligence — FP-Growth
   ↓
SVD Latent Representation
   ↓
CURE Customer Segmentation
   ↓
Recommendation Model Benchmarking
   ↓
Statistical Validation
   ↓
Spark Performance / Scalability Lab
   ↓
Executive Dashboard
   ↓
Automatic Conclusions & Exports
```

The purpose is not only to run algorithms, but to answer four questions:

1. **What is happening in the data?**
2. **Which customer/product patterns are meaningful?**
3. **Which model performs best and why?**
4. **Can the solution scale and be explained clearly to a non-technical audience?**

---

## Datasets

The project uses two small datasets suitable for a reproducible classroom / portfolio demonstration.

### `baskets.csv`

Transactional grocery-basket data used for:

- Spark RDD/DataFrame analysis;
- product popularity;
- basket behavior;
- temporal analysis;
- RFV-style customer segmentation;
- FP-Growth association-rule mining.

Important fields include:

- `Member_number`
- `Date`
- `itemDescription`
- `year`
- `month`
- `day`
- `day_of_week`

### `ratings2k.csv`

User–item rating data used for:

- SVD;
- latent user/item representations;
- CURE clustering;
- collaborative filtering;
- ALS;
- recommendation benchmarking.

Important fields include:

- `user`
- `item`
- `rating`

The notebook automatically reads the repository datasets from:

```text
end/end/baskets.csv
end/end/ratings2k.csv
```

---

# Pipeline Details

## 1. Environment, Data Quality & Executive KPI Shell

The notebook first installs and configures the main analytical stack and initializes a Spark session.

Before any model is trained, it checks:

- schema;
- missing values;
- duplicated records;
- dataset sizes;
- number of users/items/transactions;
- rating range;
- matrix sparsity;
- basic descriptive statistics.

This prevents later metrics from being interpreted without understanding the data quality and structure.

### Main technologies

- Python
- Apache Spark / PySpark
- NumPy
- pandas
- scikit-learn
- Plotly
- NetworkX
- UMAP
- pyclustering

---

## 2. Spark RDD & DataFrame Analytics

The project retains the Big Data foundations by implementing analytical operations with Spark RDDs and/or DataFrames.

Examples include:

- most frequently purchased products;
- high-activity customers;
- monthly purchasing patterns;
- distinct basket counts;
- reusable Spark transformations and aggregations.

The objective is not merely to obtain results, but to demonstrate how distributed data abstractions can express the same analytical problem in different ways.

---

## 3. RFV Customer Analytics

The basket dataset does not contain transaction prices, therefore a true monetary-value `M` cannot be calculated honestly.

Instead, the notebook uses an **RFV-style** representation:

- **R — Recency:** how recently the customer purchased;
- **F — Frequency:** how frequently the customer created baskets;
- **V — Volume proxy:** purchase/item volume used as an activity-value proxy.

This distinction is intentional: the project avoids inventing monetary values that do not exist in the data.

The RFV analysis is used to create interpretable customer profiles that can later complement clustering and business dashboards.

---

## 4. Market Basket Intelligence — FP-Growth

Spark MLlib `FPGrowth` is used to discover frequently co-occurring products and association rules.

The notebook explores and/or tunes parameters such as:

- `minSupport`
- `minConfidence`

Important rule metrics include:

- **Support** — how often the combination occurs;
- **Confidence** — how frequently the consequent occurs when the antecedent occurs;
- **Lift** — how much stronger the relationship is than random co-occurrence.

The project goes beyond printing association rules by presenting them as potential **cross-sell / bundle opportunities** with interactive visualizations and a product-relation network.

---

## 5. SVD — Latent Representation Learning

SVD is used to reduce the sparse user–item rating space into a compact latent representation.

The notebook:

1. builds the user–item matrix;
2. handles the observed-rating structure;
3. computes truncated/distributed SVD;
4. inspects singular-value energy;
5. chooses a practical latent dimensionality;
6. creates user embeddings and item embeddings;
7. uses these embeddings in downstream visualization and clustering.

The objective is not simply dimensionality reduction. The latent space provides a compact representation of preference patterns that can support:

- customer segmentation;
- similarity analysis;
- recommendation;
- 2D visual exploration.

---

## 6. CURE Customer Segmentation

The clustering section uses **CURE (Clustering Using REpresentatives)** rather than presenting K-Means under a CURE label.

CURE represents a cluster using multiple representative points that are shrunk toward the cluster center, making it more flexible than a single-centroid method for irregular cluster shapes and outliers.

The notebook evaluates candidate configurations using several signals instead of relying on one metric:

- Silhouette Score;
- Davies–Bouldin Index;
- Calinski–Harabasz Score;
- cluster balance;
- stability via Adjusted Rand Index where appropriate.

### Visual explanations

The latent/customer space is projected for presentation using techniques such as:

- PCA;
- UMAP;
- t-SNE.

Interactive Plotly visualizations allow a reviewer to hover over points and inspect cluster behavior more easily.

The project also creates **cluster personas/profiles**, making the segmentation understandable in business terms instead of returning anonymous labels such as `cluster 0`, `cluster 1`, etc.

---

## 7. Recommendation Engine Benchmark

A major goal of the project is to avoid claiming that a recommender is good based on only one model or one metric.

The notebook compares multiple approaches, including suitable baselines and collaborative-filtering methods such as:

- Global Mean baseline;
- User Mean baseline;
- Item Mean baseline;
- Popularity baseline;
- User-based Collaborative Filtering;
- SVD-based prediction/reconstruction;
- Spark ALS.

### Rating-prediction metrics

- **RMSE** — Root Mean Squared Error
- **MAE** — Mean Absolute Error

Lower is better for both.

### Top-N recommendation metrics

- **Precision@K** — proportion of recommended items that are relevant;
- **Recall@K** — proportion of relevant items successfully retrieved.

The project therefore distinguishes:

```text
Rating Prediction Quality ≠ Ranking / Recommendation Quality
```

A model can have good RMSE and still generate weak Top-N recommendations, so both perspectives are reported.

---

## 8. Hyperparameter Tuning

The project performs controlled hyperparameter exploration rather than relying entirely on default parameters.

Examples include tuning:

### ALS

- rank;
- regularization parameter;
- number of iterations.

### SVD

- latent dimensionality / number of concepts.

### CURE

- number of clusters;
- representative-point settings;
- compression/shrink factor where applicable.

### FP-Growth

- support threshold;
- confidence threshold.

The search space is intentionally bounded so that the project remains practical in Google Colab while still providing meaningful model selection.

---

## 9. Statistical Validation

A single random train/test split can produce an accidentally optimistic or pessimistic result.

For that reason, the notebook supports repeated evaluation with multiple random seeds and summarizes model quality using statistics such as:

```text
mean ± standard deviation
```

This makes comparisons more defensible than reporting a single lucky run.

---

## 10. Spark Performance & Scalability Lab

Because this is a Big Data project, model accuracy alone is not enough.

The notebook also examines **computational performance**, including experiments such as:

- increasing data scale;
- execution time;
- throughput;
- Spark caching/reuse;
- partitions and execution behavior;
- Spark physical plans where useful.

A key principle of the project is:

> Spark is not automatically faster on tiny local datasets.

For small demo data, Python/pandas may be faster because Spark has scheduling and distributed-computation overhead. The notebook therefore does not manipulate the benchmark to force Spark to win.

Instead, the project demonstrates why Spark becomes valuable when the workload grows, is partitioned, reused, or needs distributed execution.

This gives the project two distinct performance perspectives:

| Dimension | Examples |
|---|---|
| **Model performance** | RMSE, MAE, Precision@K, Recall@K, clustering metrics |
| **System performance** | runtime, throughput, cache benefit, scalability |

---

# Executive Dashboard

The notebook contains an Executive Dashboard designed with a clean **corporate / modern AI / minimal** presentation style.

Depending on the completed stages, it can summarize information such as:

- dataset size;
- users and items;
- transactions;
- sparsity;
- best recommendation model;
- RMSE / MAE;
- Precision@K / Recall@K;
- selected SVD dimension;
- best CURE configuration;
- clustering quality;
- important association rules;
- Spark performance observations.

Plotly charts include hover information where useful so a presenter can explain why a point, bar, rule or KPI has its value without filling the notebook with unnecessary static text.

---

# Automatic Conclusions

Near the end of the notebook, the project automatically summarizes the results calculated during the run.

Examples of questions it is designed to answer are:

- Which recommendation model is the current champion?
- Which model predicts ratings most accurately?
- Which model gives the strongest Top-N recommendations?
- Which SVD dimension provides a useful compact representation?
- Which CURE configuration produces the most defensible segmentation?
- Which FP-Growth rules are potentially actionable?
- What did the Spark scalability experiment demonstrate?

Results are generated from the current execution rather than hard-coded into the notebook.

---

# How to Run

## Recommended: Google Colab

### Step 1 — Open the notebook

Click the badge at the top of this README or use:

https://colab.research.google.com/github/NVTruong473/BIG-DATA/blob/main/BIG_DATA_Customer_Intelligence_PRO_Colab.ipynb

### Step 2 — Run all cells

In Google Colab:

```text
Runtime → Run all
```

The notebook installs its required Python packages and executes the workflow from top to bottom.

### Step 3 — Allow the full pipeline to finish

The repository datasets are intentionally small enough for demonstration, but the notebook includes tuning, repeated validation, CURE analysis and performance experiments.

Actual runtime depends on the current Colab CPU/RAM allocation and the selected quality configuration.

The notebook uses a quality-oriented configuration by default so complete execution may take significantly longer than a minimal classroom script.

### Optional quality setting

The notebook contains a configuration similar to:

```python
QUALITY_MODE = "max"
```

`max` prioritizes stronger analysis and validation. A lighter/balanced configuration can be used when a faster demonstration is preferred.

---

# Recommended Presentation Order

For a lecturer, interviewer or customer-style demonstration, it is usually clearer **not** to begin by scrolling through every line of code.

A recommended presentation order is:

1. **Problem & project objective**
2. **Executive Dashboard**
3. **Data quality and dataset structure**
4. **Market Basket / FP-Growth insights**
5. **RFV customer behavior**
6. **SVD latent representation**
7. **CURE segmentation + interactive 2D visualization**
8. **Recommendation model comparison**
9. **RMSE / MAE + Precision@K / Recall@K**
10. **Spark scalability evidence**
11. **Automatic conclusions**
12. **Code walkthrough for technical questions**

This order first answers **“What value did the system produce?”** and then explains **“How was it built?”**

---

# Outputs & Exports

The notebook can generate/export analytical artifacts such as:

- model evaluation tables;
- user/item embeddings;
- latent concepts;
- cluster assignments;
- CURE representative information;
- cluster profiles/personas;
- RFV segments;
- association rules;
- Spark benchmark results;
- trained recommendation artifacts where supported;
- JSON summaries;
- interactive HTML visualizations/dashboard;
- bundled result files.

The exact outputs depend on the cells executed and the successful completion of the relevant pipeline stages.

---

# Repository Structure

```text
BIG-DATA/
│
├── README.md
├── BIG_DATA_Customer_Intelligence_PRO_Colab.ipynb   # MAIN NOTEBOOK
│
├── end/
│   └── end/
│       ├── baskets.csv
│       ├── ratings2k.csv
│       ├── Task1.ipynb
│       ├── Task2.ipynb
│       ├── Task3.ipynb
│       └── ...
│
├── midterm/                                         # legacy/reference work
└── Proccess 2/                                      # legacy/reference work
```

> **Important:** For reviewing or running the current project, start with `BIG_DATA_Customer_Intelligence_PRO_Colab.ipynb`. The older notebooks are retained as historical/reference material and are not required for the main demonstration.

---

# Design Principles

This version of the project follows several principles that are intentionally stricter than a normal classroom demo:

### 1. No hidden processing pipeline

Core processing code is kept directly inside the main notebook so reviewers can inspect the actual workflow.

### 2. No fake metric improvement

The project does not hard-code favorable model scores or hide weak experimental results.

### 3. Algorithm names should match implementations

For example, the clustering stage uses CURE behavior rather than simply renaming K-Means as CURE.

### 4. Separate prediction from recommendation quality

RMSE/MAE and Precision@K/Recall@K measure different aspects and are reported separately.

### 5. Do not invent unavailable business data

Because basket prices/revenue are not present, the project uses an RFV-style activity proxy instead of pretending true monetary-value information exists.

### 6. Spark claims require evidence

Spark scalability is examined experimentally rather than claimed from theory alone.

### 7. Technical results should be interpretable

Interactive charts, cluster personas, executive KPIs and automatic conclusions translate model outputs into information that can be presented to non-specialists.

---

# Technology Stack

| Area | Technologies |
|---|---|
| Big Data | Apache Spark, PySpark, Spark SQL, RDD, DataFrame |
| Association Mining | Spark MLlib FP-Growth |
| Dimensionality Reduction | SVD / Spark RowMatrix, PCA |
| Clustering | CURE, scikit-learn metrics |
| Recommendation | UserCF, SVD-based methods, Spark ALS, baselines |
| Evaluation | RMSE, MAE, Precision@K, Recall@K, Silhouette, DBI, CH, ARI |
| Visualization | Plotly, UMAP, t-SNE, PCA, NetworkX |
| Data Analysis | Python, pandas, NumPy |
| Runtime | Google Colab |

---

# Notes on Reproducibility

- Random seeds are controlled where practical.
- Model results are calculated during execution and can vary slightly depending on library/runtime behavior.
- Colab hardware can differ between sessions.
- Interactive visualizations may render differently between GitHub preview and a live Colab session.
- For the complete experience, run the notebook in Colab rather than relying only on GitHub's static notebook preview.

---

# Project Status

The current version is designed as a **demo / academic / portfolio-grade Customer Intelligence system** rather than a production service with real-time APIs or a production data warehouse.

The small datasets make the full workflow easy to reproduce, while the architecture demonstrates how the same analytical ideas can be expanded to larger distributed datasets.

Potential future extensions include:

- larger-scale transaction and rating datasets;
- temporal recommendation splits;
- cold-start handling with metadata;
- item/content embeddings;
- hybrid recommendation;
- distributed CURE approximation for substantially larger populations;
- experiment tracking;
- scheduled Spark jobs;
- model serving APIs;
- cloud object storage / lakehouse integration;
- production monitoring and drift detection.

---

## Quick Start

If you only want to see the project working:

1. Click **Open in Colab** at the top of this page.
2. Select **Runtime → Run all**.
3. Follow the notebook sections in order.
4. Start your presentation from the **Executive Dashboard and automatic conclusions**, then use the preceding cells to explain the evidence behind each result.

---

### Main entry point

**`BIG_DATA_Customer_Intelligence_PRO_Colab.ipynb`**

https://github.com/NVTruong473/BIG-DATA/blob/main/BIG_DATA_Customer_Intelligence_PRO_Colab.ipynb
