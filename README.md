# TruePitch Analytics: Player Influence & Anomaly Detection Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Framework: Scikit-Learn](https://img.shields.io/badge/Framework-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Library: Pandas](https://img.shields.io/badge/Library-Pandas-darkblue.svg)](https://pandas.pydata.org/)
[![Domain: Sports Analytics](https://img.shields.io/badge/Domain-Sports%20Analytics-green.svg)](https://github.com/)

A data science and machine learning project designed to quantify the true operational impact of professional football players on match outcomes. By moving beyond high-visibility bias metrics (goals and assists), this system aggregates role-specific performance metrics to identify under-evaluated defensive/midfield anchors and expose over-valued players using clustering and anomaly detection pipelines.

---

## 📌 Problem & Objective

In professional football, public perception and player valuations are heavily biased toward attacking players due to visible statistics like goals and assists. Consequently, defensive midfielders, center-backs, and goalkeepers are systematically marginalized despite driving team success. 

**Objectives:**
* Compute a position-agnostic **True Influence Score** mapping individual metrics directly to match outcomes.
* Segment players based on functional performance profiles rather than nominal positions.
* Isolate market anomalies (highly under-valued or over-valued players) using robust statistical outlier detection.

---

## 📊 Data Engineering Pipeline

### 1. Data Sourcing & Features
Aggregated dataset compiled from elite European league records and public sports analytics platforms (`FBref`, `WhoScored`, `SofaScore`, `Transfermarkt`).
* **Match Context:** Historical scores, team lineups, and minute-by-minute squad compositions.
* **Role-Specific Feature Matrices:**
  * **Forwards:** Goals, expected goals (xG), assists, progressive carries, shots on target.
  * **Midfielders:** Progressive passes, completion rate, interceptions, recoveries.
  * **Defenders:** In-the-box tackles, aerial duels won, clearances, clean sheets.
  * **Goalkeepers:** Post-shot expected goals minus goals allowed (PSxG+/-), high claims, save percentage.

### 2. Preprocessing & Cleaning
* Handled structured noise, missing data vectors, and input typos via role-specific imputation.
* Applied **Z-score Standardisation** and Min-Max scaling to ensure structural equity across features before running distance-based algorithms.

---

## 🛠️ Methodology & Machine Learning Architecture

### 1. Exploratory Data Analysis (EDA)
Executed multi-variable correlation matrix analyses and linear regressions to find hidden relationships between specific off-the-ball actions and a team's final goal differential.

### 2. Functional Unsupervised Clustering
* **Algorithms:** `K-Means` and `DBSCAN`
* **Implementation:** Grouped players into distinct performance tiers and tactical clusters based on actual on-pitch metrics rather than official team roster designations.

### 3. Statistical Anomaly Detection
* **Algorithms:** `Isolation Forest`, `Z-Score Filtering`, and `Boxplot Outlier Analysis`
* **Implementation:** Isolated individual data points that deviated heavily from their cluster centers. Players with exceptionally high statistical impact but low mainstream match ratings/market values were flagged as **underrated gems**.
