import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
import pickle

df = pd.read_csv("Mall_Customers.csv")

X = df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]

pipeline = Pipeline([
    ("scaler", StandardScaler())
])

X_scaled = pipeline.fit_transform(X)

kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X_scaled)

pickle.dump(pipeline, open("clustering_pipeline.pkl", "wb"))
pickle.dump(kmeans, open("segmentation_model.pkl", "wb"))
