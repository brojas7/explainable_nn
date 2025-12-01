import numpy as np
from pprint import pprint
from explainable_nn.utils import load_model
from explainable_nn.gemini_wrapper import NeuralExplainerLLM

API_KEY = "TU_API_KEY"

feature_context = {
    "0": "Longitud del sépalo (cm)",
    "1": "Ancho del sépalo (cm)",
    "2": "Longitud del pétalo (cm)",
    "3": "Ancho del pétalo (cm)"
}

business_context = """
0 = Setosa
1 = Versicolor
2 = Virginica
"""

nuevo = np.array([5.1, 3.8, 1.6, 0.2])

saved = load_model("modelo.pkl")
nn = saved["modelo"]
scaler = saved["scaler"]

nuevo_scaled = scaler.transform([nuevo])
log = nn.explain(nuevo_scaled[0])

print("\n--- Log Técnico ---")
pprint(log)

explainer = NeuralExplainerLLM(api_key=API_KEY)
explicacion = explainer.explain(log, feature_context, business_context)

print("\n--- Explicación Natural ---")
print(explicacion)
