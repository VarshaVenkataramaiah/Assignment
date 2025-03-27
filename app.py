from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
import numpy as np
import sqlite3
from datetime import datetime

app = Flask(__name__)
CORS(app)

model_dir = "models"

# Load models
risk_model = joblib.load(os.path.join(model_dir, "risk_model.pkl"))
roi_model = joblib.load(os.path.join(model_dir, "roi_model.pkl"))
tax_model = joblib.load(os.path.join(model_dir, "tax_model.pkl"))
demand_model = joblib.load(os.path.join(model_dir, "demand_model.pkl"))
material_model = joblib.load(os.path.join(model_dir, "material_model.pkl"))
land_model = joblib.load(os.path.join(model_dir, "land_model.pkl"))
investor_model = joblib.load(os.path.join(model_dir, "investor_model.pkl"))
recommend_model = joblib.load(os.path.join(model_dir, "recommend_model.pkl"))
aesthetic_model = joblib.load(os.path.join(model_dir, "aesthetic_model.pkl"))
amenity_model = joblib.load(os.path.join(model_dir, "amenity_model.pkl"))
view_model = joblib.load(os.path.join(model_dir, "view_model.pkl"))

# Load encoders
location_encoder = joblib.load(os.path.join(model_dir, "location_encoder.pkl"))
use_type_encoder = joblib.load(os.path.join(model_dir, "use_type_encoder.pkl"))

# Setup database
conn = sqlite3.connect("real_estate_logs.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''
CREATE TABLE IF NOT EXISTS report_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    size REAL, demand REAL, cost REAL, location TEXT, useType TEXT,
    timestamp TEXT
)''')
conn.commit()

@app.route("/api/generate-report", methods=["POST"])
def generate_report():
    try:
        data = request.get_json()
        size = data.get("size")
        demand = data.get("demand")
        cost = data.get("cost")
        location = data.get("location")
        use_type = data.get("useType")

        if None in [size, demand, cost, location, use_type]:
            return jsonify({"error": "Missing input fields"}), 400

        # Encode categorical
        try:
            location_enc = location_encoder.transform([location])[0]
        except:
            location_enc = 0
        try:
            use_enc = use_type_encoder.transform([use_type])[0]
        except:
            use_enc = 0

        features = np.array([[size, demand, cost, location_enc, use_enc]])

        # Predict
        risk = float(risk_model.predict(features)[0])
        roi = float(roi_model.predict(features)[0])
        tax = float(tax_model.predict(features)[0])
        demand_forecast = float(demand_model.predict(features)[0])
        material_cost = float(material_model.predict(features)[0])
        land_score = float(land_model.predict(features)[0])
        investor_score = float(investor_model.predict(features)[0])
        recommend_score = float(recommend_model.predict(features)[0])

        # Emotional appeal scores
        aesthetic = float(aesthetic_model.predict(features)[0])
        amenity = float(amenity_model.predict(features)[0])
        view = float(view_model.predict(features)[0])

        # Save to database
        cursor.execute('''
            INSERT INTO report_logs (size, demand, cost, location, useType, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (size, demand, cost, location, use_type, datetime.now().isoformat()))
        conn.commit()

        return jsonify({
            "zoning": f"Zoning potential score: {roi:.2f}",
            "scenario": f"Simulated development risk level: {risk:.2f}",
            "financial": f"Estimated tax burden: {tax:.2f}%",
            "risk": f"{risk:.2f}",
            "roi": f"{roi:.2f}",
            "tax": f"{tax:.2f}%",
            "demandForecast": f"{demand_forecast:.2f}",
            "materialEstimate": f"${material_cost:,.0f}",
            "landAcquisition": f"{land_score:.2f}",
            "investorScore": f"{investor_score:.2f}",
            "recommendation": "Pursue investment with zoning flexibility" if recommend_score > 0.5 else "Delay development for better yield",
            "aestheticScore": f"{aesthetic:.2f}",
            "amenityScore": f"{amenity:.2f}",
            "viewScore": f"{view:.2f}"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)
