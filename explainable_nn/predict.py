import numpy as np

def predict_raw(model, x):
    """
    Ejecuta forward() sobre un solo registro sin escalar.
    Retorna: (clase_predicha, probabilidades)
    """
    probs = model.forward(x)
    pred = int(np.argmax(probs))
    return pred, probs


def predict_scaled(model, scaler, x):
    """
    Escala un solo registro antes de pasarlo a la red.
    Retorna: (clase_predicha, probabilidades)
    """
    x_scaled = scaler.transform([x])[0]
    probs = model.forward(x_scaled)
    pred = int(np.argmax(probs))
    return pred, probs


def predict_batch(model, scaler, X, scaled=True):
    """
    Predicción por lotes.
    X puede ser un DataFrame o array.
    Retorna: (lista_predicciones, lista_probabilidades)
    """
    preds = []
    probs_list = []

    # Convertir DataFrame a numpy si aplica
    if hasattr(X, "values"):
        X = X.values

    if scaled:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X

    for x in X_scaled:
        prob = model.forward(x)
        pred = int(np.argmax(prob))
        preds.append(pred)
        probs_list.append(prob)

    return np.array(preds), np.array(probs_list)
