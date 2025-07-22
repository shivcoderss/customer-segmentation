import streamlit as st
import pandas as pd
import numpy as np  # ✅ Import numpy separately
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(page_title="Customer Segmentation", layout="centered")

st.title("🧠 Customer Segmentation with K-Means")
st.markdown("Segmenting customers based on Age, Income, and Spending Score.")

@st.cache_data
def load_data():
    url = "https://people.sc.fsu.edu/~jburkardt/data/csv/hw_200.csv"
    df = pd.read_csv(url)
    df = df.rename(columns={
        "Height(Inches)": "Height",
        "Weight(Pounds)": "Weight"
    })
    df["Age"] = np.random.randint(18, 60, size=len(df))
    df["Annual Income (k$)"] = np.random.randint(30, 130, size=len(df))
    df["Spending Score (1-100)"] = np.random.randint(1, 101, size=len(df))
    return df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]

df = load_data()

st.subheader("Raw Dataset")
st.dataframe(df.head())

# Preprocessing
X = df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Clustering
kmeans = KMeans(n_clusters=5, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# Cluster distribution
st.subheader("🎯 Cluster Distribution")
fig1, ax1 = plt.subplots()
sns.countplot(x=df["Cluster"], ax=ax1, palette="Set2")
st.pyplot(fig1)

# 2D Visualization
st.subheader("🧬 Cluster Visualization (2 Features)")
option = st.selectbox("Choose X-Axis", X.columns)
option2 = st.selectbox("Choose Y-Axis", [col for col in X.columns if col != option])

fig2, ax2 = plt.subplots()
sns.scatterplot(data=df, x=option, y=option2, hue="Cluster", palette="Set2", s=100)
st.pyplot(fig2)

# Show full data
st.subheader("📊 Clustered Data")
st.dataframe(df)