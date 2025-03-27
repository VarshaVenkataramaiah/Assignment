import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Create directory for models
os.makedirs("models", exist_ok=True)

# Generate synthetic data
np.random.seed(42)
n = 1000

df = pd.DataFrame({
    "size": np.random.randint(600, 10000, n),
    "demand": np.round(np.random.uniform(0.1, 1.0, n), 2),
    "cost": np.random.randint(100000, 3000000, n),
    "location": np.random.choice(["New York", "London", "Toronto", "San Francisco", "Dubai"], n),
    "use_type": np.random.choice(["Residential", "Commercial", "Industrial", "Mixed-use"], n),
    "risk_score": np.random.uniform(0, 1, n),
    "roi": np.random.uniform(3, 15, n),
    "tax_saving": np.random.uniform(0, 100, n),
    "demand_forecast": np.random.uniform(0.1, 1.0, n),
    "material_cost": np.random.uniform(25000, 350000, n),
    "land_score": np.random.uniform(1.0, 10.0, n),
    "investor_score": np.random.uniform(0, 100, n),
    "recommendation_score": np.random.randint(0, 2, n),
    "aesthetic_score": np.random.uniform(30, 100, n),
    "amenity_score": np.random.uniform(20, 90, n),
    "view_score": np.random.uniform(10, 100, n)
})

# Encode location & use_type
le_location = LabelEncoder()
le_use = LabelEncoder()
df["location_encoded"] = le_location.fit_transform(df["location"])
df["use_type_encoded"] = le_use.fit_transform(df["use_type"])

# Save encoders
joblib.dump(le_location, "models/location_encoder.pkl")
joblib.dump(le_use, "models/use_type_encoder.pkl")

# Input features
X = df[["size", "demand", "cost", "location_encoded", "use_type_encoded"]]

# Target features for models
targets = {
    "risk_model": df["risk_score"],
    "roi_model": df["roi"],
    "tax_model": df["tax_saving"],
    "demand_model": df["demand_forecast"],
    "material_model": df["material_cost"],
    "land_model": df["land_score"],
    "investor_model": df["investor_score"],
    "recommend_model": df["recommendation_score"],
    "aesthetic_model": df["aesthetic_score"],
    "amenity_model": df["amenity_score"],
    "view_model": df["view_score"]
}

# Train and save all models
for name, y in targets.items():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, f"models/{name}.pkl")
    print(f"✅ Saved {name}.pkl")

print("✅ All models saved in models/")
