# Coloca aquí el archivo completo 'explainable_nn_complete.py'
# Incluye red neuronal desde cero, logging, explicación con Gemini y demo con Iris.

#############################################################
#     EXPLICABLE NEURAL NETWORK + GEMINI (UN SOLO ARCHIVO)
#############################################################

import numpy as np
from pprint import pprint
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import google.generativeai as genai
import dill

#############################################################
# AUXILIARY FUNCTIONS
#############################################################

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / np.sum(ex)

def cross_entropy_loss(probs, y_true_index):
    return -np.log(probs[y_true_index] + 1e-12)

#############################################################
# EXPLAINABLE NEURAL NETWORK (FROM SCRATCH)
#############################################################

class ExplainableNeuralNet:
    def __init__(self, layer_sizes, learning_rate=0.01, seed=42):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.rng = np.random.default_rng(seed)

        self.W = []
        self.b = []

        # Xavier initialization
        for i in range(len(layer_sizes) - 1):
            n_in = layer_sizes[i]
            n_out = layer_sizes[i + 1]
            limit = np.sqrt(6 / (n_in + n_out))
            W_l = self.rng.uniform(-limit, limit, (n_in, n_out))
            b_l = np.zeros(n_out)
            self.W.append(W_l)
            self.b.append(b_l)

    #########################################################
    # FORWARD WITH LOGGING OPTION
    #########################################################
    def forward(self, x, log_detail=False):
        a = x.reshape(1, -1)
        activaciones = []
        pre_activaciones = []

        # Hidden layers
        for l in range(len(self.W) - 1):
            z = a @ self.W[l] + self.b[l]
            a = relu(z)

            if log_detail:
                pre_activaciones.append(z.copy())
                activaciones.append(a.copy())

        # Output layer
        z_final = a @ self.W[-1] + self.b[-1]
        logits = z_final.flatten()
        probs = softmax(logits)

        if log_detail:
            pre_activaciones.append(z_final.copy())
            activaciones.append(probs.reshape(1, -1))

            log = {
                "input": x.tolist(),
                "pre_activaciones": [z.flatten().tolist() for z in pre_activaciones],
                "activaciones": [a_l.flatten().tolist() for a_l in activaciones],
                "logits": logits.tolist(),
                "probabilidades": probs.tolist()
            }
            return probs, log

        return probs

    #########################################################
    # BACKPROP
    #########################################################
    def train_step(self, x, y_true_index):
        a = x.reshape(1, -1)
        a_list = [a]
        z_list = []

        for l in range(len(self.W) - 1):
            z = a @ self.W[l] + self.b[l]
            a = relu(z)
            z_list.append(z)
            a_list.append(a)

        z_final = a @ self.W[-1] + self.b[-1]
        z_list.append(z_final)

        logits = z_final.flatten()
        probs = softmax(logits)
        loss = cross_entropy_loss(probs, y_true_index)

        # BACKPROP
        grad_logits = probs.copy()
        grad_logits[y_true_index] -= 1
        grad_logits = grad_logits.reshape(1, -1)

        # Output layer
        dW_last = a_list[-1].T @ grad_logits
        db_last = grad_logits.sum(axis=0)
        grad_a_prev = grad_logits @ self.W[-1].T

        self.W[-1] -= self.learning_rate * dW_last
        self.b[-1] -= self.learning_rate * db_last

        # Hidden layers
        for l in reversed(range(len(self.W) - 1)):
            z_l = z_list[l]
            a_prev = a_list[l]

            grad_z = grad_a_prev * relu_derivative(z_l)
            dW_l = a_prev.T @ grad_z
            db_l = grad_z.sum(axis=0)

            if l > 0:
                grad_a_prev = grad_z @ self.W[l].T

            self.W[l] -= self.learning_rate * dW_l
            self.b[l] -= self.learning_rate * db_l

        return loss

    #########################################################
    # EXPLANATION: MAIN PATH
    #########################################################
    def explain(self, x, top_k=5):
        probs, log = self.forward(x, log_detail=True)
        pred_class = int(np.argmax(probs))

        activaciones = [np.array(a) for a in log["activaciones"]]

        rutas = []

        for l in range(len(self.W) - 1):
            a_l = activaciones[l].flatten()
            W_next = self.W[l + 1]
            weight_strength = np.sum(np.abs(W_next), axis=1)
            contrib = np.abs(a_l) * weight_strength

            for j in range(len(contrib)):
                rutas.append({
                    "capa": l,
                    "neurona": j,
                    "contribucion": float(contrib[j])
                })

        rutas_ordenadas = sorted(rutas, key=lambda r: r["contribucion"], reverse=True)
        ruta_principal = rutas_ordenadas[:top_k]

        return {
            "input": log["input"],
            "prediccion_clase": pred_class,
            "probabilidades": log["probabilidades"],
            "logits": log["logits"],
            "activaciones": log["activaciones"],
            "ruta_principal": ruta_principal
        }

#############################################################
# GEMINI LLM WRAPPER
#############################################################

class NeuralExplainerLLM:
    def __init__(self, api_key, model="gemini-2.5-flash"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)

    def explain(self, nn_log, feature_context=None, business_context=None):

        SYSTEM_PROMPT = """
Eres un experto en interpretar redes neuronales explicables.
No inventes nada.
Usa la ruta_principal, activaciones, logits y probabilidades.
Usa el contexto de variables y de negocio si existe.
Ofrece una explicacion al final en lenguaje de negocio que personas no tecnicas puedan entender. Expilca las relaciones entre variables y el resultado.
"""

        user_prompt = f"""
Aquí está el log técnico de la red neuronal:

```json
{nn_log}
```

Contexto de variables:
```json
{feature_context}
```

Contexto de negocio:
```
{business_context}
```

Por favor explica en lenguaje natural por qué la red tomó la decisión final.
"""

        # FORMATO CORRECTO PARA GEMINI
        response = self.model.generate_content(
            [
                {"text": SYSTEM_PROMPT},
                {"text": user_prompt}
            ]
        )

        return response.text