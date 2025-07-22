# 🧠 Customer Segmentation Web App

This Streamlit app segments customers into distinct groups based on Age, Annual Income, and Spending Score using **K-Means clustering**. It's built using synthetic data generated from a public CSV file (height & weight) enriched with randomly created demographic features.

# 🧠 [Live Demo → Customer Segmentation Web App](https://customer-segmentation-519.streamlit.app/)

---

## 🚀 Features

- ✅ Loads and augments real-world CSV data (no manual file upload needed)
- ✅ Applies standard scaling and K-Means clustering
- ✅ Visualizes cluster distribution
- ✅ Allows interactive 2D scatter plots with cluster labels
- ✅ Displays the full clustered dataset

---

## 📊 Dataset Source

- Base CSV: [Height & Weight Dataset](https://people.sc.fsu.edu/~jburkardt/data/csv/hw_200.csv)
- Added fields: Age, Annual Income, and Spending Score generated using NumPy

---

## 🛠️ Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Seaborn
- Matplotlib

---

## 📦 Installation

1. Clone the repository or download the files
2. Create a virtual environment (optional but recommended)
3. Install dependencies:

```bash
pip install -r requirements.txt
streamlit run app.py
```
