import numpy as np
import re
from tensorflow import keras

# ── LOAD MODEL ──
try:
    model = keras.models.load_model("final_model.h5", compile=False)
    print("✅ Model loaded")
except:
    model = None
    print("❌ Model not loaded")

# ── HDC ENCODING ──
DIM = 1000

def encode_sample(sample):
    np.random.seed(42)
    hv = np.zeros(DIM)

    for value in sample:
        rand_vec = np.random.choice([-1, 1], size=DIM)
        hv += value * rand_vec

    return np.sign(hv)

# ── URL FEATURES ──
def url_features(url):
    if not url:
        return 0

    score = 0
    keywords = ["login","verify","bank","secure","update","password","free","paypal"]

    if any(k in url.lower() for k in keywords):
        score += 3
    if len(url) > 70:
        score += 2
    if url.count(".") > 3:
        score += 1
    if "@" in url:
        score += 2

    return score

# ── TRANSACTION RISK ──
def transaction_risk(features):
    score = 0

    amount = features[0]

    if amount > 1000:
        score += 3
    elif amount > 500:
        score += 1

    extreme = sum(1 for v in features if abs(v) > 3)

    if extreme > 8:
        score += 5
    elif extreme > 4:
        score += 3
    elif extreme > 2:
        score += 1

    return score

# ── HYBRID PREDICT ──
def hybrid_predict(features, url):
    try:
        x = encode_sample(features).reshape(1, -1)

        # ML prediction
        if model:
            prob = float(model.predict(x, verbose=0)[0][0])
        else:
            prob = 0.0

        # FIXED THRESHOLD
        ml_flag = 1 if prob > 0.35 else 0

        url_score = min(url_features(url), 10)
        txn_score = min(transaction_risk(features), 12)

        print("\n--- DEBUG ---")
        print("ML Prob:", prob)
        print("URL:", url_score)
        print("Txn:", txn_score)

        risk = (prob * 3.0) + (url_score * 0.9) + (txn_score * 1.3)

        print("Risk:", risk)

        if risk >= 4.0:
            confidence = min(99, int(60 + risk * 5))
            return 1, confidence
        else:
            confidence = min(99, int(92 - risk * 10))
            return 0, confidence

    except Exception as e:
        print("Error:", e)
        return 0, 10