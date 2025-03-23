from flask import Flask, request, jsonify
from flask_cors import CORS  # <-- ajoute ça
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

app = Flask(__name__)
CORS(app)  # <-- et ça aussi pour autoriser toutes les requêtes cross-origin

@app.route('/')
def home():
    return "Bienvenue sur l'API de prévision !"

@app.route('/forecast', methods=['POST'])
def forecast():
    # Récupère les données envoyées par le frontend
    data = request.json['data']
    model_type = request.json.get('model', 'ARIMA')
    order = request.json.get('order', [1, 0, 1])  # Par défaut ARIMA(1,0,1)

    # Convertit les données en DataFrame Pandas
    df = pd.DataFrame(data, columns=['value'])

    # Sélection du modèle à appliquer selon le paramètre envoyé
    if model_type == 'AR':
        order = (order[0], 0, 0)  # AR(p)
        model = ARIMA(df, order=order)

    elif model_type == 'MA':
        order = (0, 0, order[2])  # MA(q)
        model = ARIMA(df, order=order)

    elif model_type == 'ARMA':
        order = (order[0], 0, order[2])  # ARMA(p,q)
        model = ARIMA(df, order=order)

    elif model_type == 'SARIMAX':
        seasonal_order = request.json.get('seasonal_order', [1, 0, 1, 12])
        model = SARIMAX(df, order=order, seasonal_order=seasonal_order)

    else:  # Par défaut ARIMA
        model = ARIMA(df, order=order)

    # Entraînement du modèle
    model_fit = model.fit()

    # Faire des prédictions sur les 5 prochaines valeurs
    predictions = model_fit.forecast(steps=5)

    # Retourner les prédictions en JSON au frontend
    return jsonify({
        "model_used": model_type,
        "predictions": predictions.tolist()
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)