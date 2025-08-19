# 🌍 Global Suicide Data Analysis and Visualization


## 📌 Overview
An interactive data analysis project exploring global suicide trends across **118K+ records** with **18 features** (GDP, population, socioeconomic, employment, inflation).  
Built a **Dash + Plotly dashboard**, deployed on **Google Cloud Run**, to visualize patterns and support data-driven insights for public health interventions.  

🔗 **Live Dashboard:** [View Here](https://dashapp-473886336048.us-east1.run.app/)

---

## 🚀 Highlights
- Designed a **data pipeline** for cleaning, preprocessing, outlier detection, and feature engineering.  
- Built an **interactive dashboard** with dynamic filters and visualizations (line charts, heatmaps, violin plots, 3D scatter).  
- Applied **machine learning models** (regression, classification, clustering).  
  - Best model: **KNN (92% accuracy)**.  
- Derived insights on **age, gender, and socioeconomic correlations** with suicide rates.

---

## 🛠️ Tech Stack
- **Languages & Libraries:** Python, Pandas, NumPy, Scikit-learn, Plotly, Dash, Seaborn, Matplotlib  
- **ML Techniques:** PCA, Regression, Classification, Clustering, Isolation Forest  
- **Deployment:** Google Cloud Run  

---

## 📂 Dataset
- Source: [Kaggle – Global Suicide Rates (1990–2022)](https://www.kaggle.com/datasets/ronaldonyango/global-suicide-rates-1990-to-2022)

---

## ⚡ Quick Start
```bash
# Clone repo
git clone https://github.com/yourusername/global-suicide-data-viz.git
cd global-suicide-data-viz

# Install dependencies
pip install -r requirements.txt

# Run dashboard
python app.py
