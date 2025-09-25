import os, json, re
from html import unescape
from pathlib import Path
from flask import Flask, request, render_template, jsonify
import joblib

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "model"

model = joblib.load(MODEL_DIR / "best_model.joblib")
tfv = joblib.load(MODEL_DIR / "tfidf_vectorizer.joblib")
with open(MODEL_DIR / "metadata.json","r") as f:
    metadata = json.load(f)
THRESHOLD = float(metadata.get("threshold",0.5))

def clean_text(text):
    if text is None: return ""
    s = str(text)
    s = unescape(s)
    s = re.sub(r'http\\S+|www\\.' + r'\\S+', ' ', s)
    s = re.sub(r'@\\w+', ' ', s)
    s = re.sub(r'#', ' ', s)
    s = re.sub(r'[^0-9A-Za-z\\s]', ' ', s)
    s = re.sub(r'\\s+', ' ', s)
    return s.strip().lower()

app = Flask(__name__, template_folder="templates")

@app.route('/', methods=['GET','POST'])
def index():
    result = None; text=""
    if request.method == 'POST':
        text = request.form.get('tweet','')
        cleaned = clean_text(text)
        X = tfv.transform([cleaned])
        prob = float(model.predict_proba(X)[:,1][0])
        pred = int(prob >= THRESHOLD)
        result = {"pred": pred, "prob_positive": prob}
    return render_template('index.html', result=result, text=text)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True)
    if not data or 'text' not in data:
        return jsonify({'error':'Please provide JSON with field \"text\"'}),400
    text = data['text']
    cleaned = clean_text(text)
    X = tfv.transform([cleaned])
    prob = float(model.predict_proba(X)[:,1][0])
    pred = int(prob >= THRESHOLD)
    return jsonify({'text': text, 'cleaned': cleaned, 'pred': pred, 'prob_positive': prob})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT',5000)), debug=True)