from flask import Flask, render_template, request
import numpy as np
from predict import predict_transaction

app = Flask(__name__)

def check_url(url):
    suspicious_keywords = ["secure", "verify", "login", "update", ".xyz", ".ru"]
    for word in suspicious_keywords:
        if word in url.lower():
            return "⚠️ Suspicious URL detected"
    return "✅ URL looks safe"

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        amount = float(request.form.get("amount", 0))
        oldbalanceOrg = float(request.form.get("oldbalanceOrg", 0))
        newbalanceOrig = float(request.form.get("newbalanceOrig", 0))

        oldbalanceDest = 0
        newbalanceDest = amount

        url = request.form.get("url", "")

        values = [amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest]
        while len(values) < 30:
            values.append(0)

        values = np.array(values)

        prob, result = predict_transaction(values)
        url_result = check_url(url)

        return render_template("index.html",
                               result=result,
                               prob=round(prob * 100, 2),
                               url_result=url_result)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, port=5001)