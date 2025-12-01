import numpy as np
from pprint import pprint

#############################################################
# ACTIVACIONES Y FUNCIONES AUXILIARES
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
# RED NEURONAL EXPLICABLE
#############################################################

class ExplainableNeuralNet:
    def __init__(self, layer_sizes, learning_rate=0.01, seed=42):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.rng = np.random.default_rng(seed)

        self.W = []
        self.b = []

        # Xavier
        for i in range(len(layer_sizes) - 1):
            n_in = layer_sizes[i]
            n_out = layer_sizes[i + 1]
            limit = np.sqrt(6 / (n_in + n_out))
            self.W.append(self.rng.uniform(-limit, limit, (n_in, n_out)))
            self.b.append(np.zeros(n_out))

    #########################################################
    # FORWARD + LOGGING
    #########################################################
    def forward(self, x, log_detail=False):
        a = x.reshape(1, -1)
        activaciones = []
        pre_activaciones = []

        for l in range(len(self.W) - 1):
            z = a @ self.W[l] + self.b[l]
            a = relu(z)

            if log_detail:
                pre_activaciones.append(z.copy())
                activaciones.append(a.copy())

        z_final = a @ self.W[-1] + self.b[-1]
        logits = z_final.flatten()
        probs = softmax(logits)

        if log_detail:
            pre_activaciones.append(z_final.copy())
            activaciones.append(probs.reshape(1, -1))

            return probs, {
                "input": x.tolist(),
                "pre_activaciones": [z.flatten().tolist() for z in pre_activaciones],
                "activaciones": [a_l.flatten().tolist() for a_l in activaciones],
                "logits": logits.tolist(),
                "probabilidades": probs.tolist()
            }

        return probs

    #########################################################
    # BACKPROPAGATION
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

        # Output
        zf = a @ self.W[-1] + self.b[-1]
        z_list.append(zf)
        logits = zf.flatten()
        probs = softmax(logits)

        loss = cross_entropy_loss(probs, y_true_index)

        grad_logits = probs.copy()
        grad_logits[y_true_index] -= 1
        grad_logits = grad_logits.reshape(1, -1)

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
    # EXPLICACIÓN
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

            for j, c in enumerate(contrib):
                rutas.append({
                    "capa": l,
                    "neurona": j,
                    "contribucion": float(c)
                })

        rutas_ordenadas = sorted(rutas, key=lambda r: r["contribucion"], reverse=True)

        return {
            "input": log["input"],
            "prediccion_clase": pred_class,
            "probabilidades": log["probabilidades"],
            "logits": log["logits"],
            "activaciones": log["activaciones"],
            "ruta_principal": rutas_ordenadas[:top_k]
        }


#############################################################
# PREDICCIÓN GENERIC REUSABLE
#############################################################

def predict_with_explanation(
    model,
    x_input,
    scaler=None,
    llm=None,
    feature_context=None,
    business_context=None
):
    """
    Ejecuta una predicción explicable con cualquier red y cualquier scaler.
    - model: instancia de ExplainableNeuralNet ya entrenada
    - x_input: vector numpy con el input sin escalar
    - scaler: StandardScaler (o None si ya viene escalado)
    - llm: instancia de NeuralExplainerLLM (opcional)
    """

    # 1. Escalado
    if scaler is not None:
        x_processed = scaler.transform([x_input])[0]
    else:
        x_processed = x_input

    # 2. Explicación técnica (ruta principal, activaciones, logits)
    log_nn = model.explain(x_processed)

    # 3. Explicación natural usando LLM (opcional)
    if llm is not None:
        explanation = llm.explain(
            log_nn,
            feature_context=feature_context,
            business_context=business_context
        )
    else:
        explanation = None

    return log_nn, explanation
