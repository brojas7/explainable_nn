import numpy as np

def predict(model, scaler, x_input):
    """
    Hace una predicción simple con la XNN.
    
    Params:
        model  -> instancia de ExplainableNeuralNet
        scaler -> scaler usado para entrenar
        x_input -> lista o array con los valores crudos
    
    Return:
        predicted_class, probabilities
    """
    x_scaled = scaler.transform([x_input])
    probs = model.forward(x_scaled[0])
    pred_class = int(np.argmax(probs))
    return pred_class, probs.tolist()


def predict_with_explanation(model, scaler, x_input, explainer_llm, feature_context, business_context):
    """
    Hace predicción con explicación usando Gemini.
    """
    x_scaled = scaler.transform([x_input])
    log_nn = model.explain(x_scaled[0])

    explanation = explainer_llm.explain(
        log_nn,
        feature_context=feature_context,
        business_context=business_context
    )

    return {
        "prediction": log_nn["prediccion_clase"],
        "probabilities": log_nn["probabilidades"],
        "technical_log": log_nn,
        "explanation": explanation
    }
