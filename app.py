from flask import Flask, render_template, request
import numpy as np
from predict import hybrid_predict

app = Flask(__name__)

type_map = {"transfer": 0, "payment": 1, "debit": 2}
device_map = {"mobile": 0, "web": 1}
location_map = {"same": 0, "different": 1}

@app.route("/", methods=["GET", "POST"])
def home():

    prediction_text = ""
    risk_value = 0

    if request.method == "POST":
        try:
            amount = float(request.form.get("amount", 0))
            tx_type = type_map.get(request.form.get("type", ""), 0)
            time = float(request.form.get("time", 0))
            freq = float(request.form.get("freq", 0))
            device = device_map.get(request.form.get("device", ""), 0)
            location = location_map.get(request.form.get("location", ""), 0)
            url = request.form.get("url", "")

            # ✅ FIXED FEATURE VECTOR (meaningful, not random padding)
            features = np.array([
                amount,
                tx_type,
                time,
                freq,
                device,
                location,
                len(url),
                url.count("."),
                sum(c.isdigit() for c in url),
                1 if "http" in url else 0
            ] + [0]*20)

            prediction, risk_value = hybrid_predict(features, url)

            if prediction == 1:
                prediction_text = "🚨 FRAUDULENT TRANSACTION DETECTED"
            else:
                prediction_text = "✅ LEGITIMATE TRANSACTION"

        except Exception as e:
            prediction_text = f"Error: {str(e)}"
            risk_value = 0

    return render_template("index.html",
                           prediction_text=prediction_text,
                           risk_value=risk_value)

if __name__ == "__main__":
    app.run(debug=True, port=8000)