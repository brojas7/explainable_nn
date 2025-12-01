# Explainable Neural Network (XNN)

Una librería de Python diseñada para construir, entrenar y explicar redes neuronales **desde cero**, con soporte para **explicabilidad estructural** (rutas internas, activaciones, contribuciones por neurona) y explicaciones en **lenguaje natural** mediante **Gemini**.

Este paquete es ideal para:
- Cursos de IA y ciencia de datos
- Proyectos que requieren interpretabilidad
- Investigaciones en modelos explicables
- Integración con sistemas reales que necesiten auditabilidad

---

## Características Principales

### ✔ Red neuronal implementada **desde cero**
- Forward y backward propagation manuales
- Inicialización Xavier
- Funciones de activación ReLU y softmax
- Cálculo explícito de gradientes

### ✔ Explicabilidad profunda (XAI integrada)
- Registro de activaciones y preactivaciones por capa
- Extracción de la *ruta principal* de contribución
- Desglose cuantitativo por neurona
- Log técnico completo

### ✔ Explicación en lenguaje natural con Gemini
- Resumen amigable de decisiones internas
- Análisis basado en activaciones, logits y pesos
- Personalizable con contexto de negocio y de variables

### ✔ Pipeline listo para producción
- Guardado y carga con `dill`
- Función genérica `predict_with_explanation()`
- Integración modular con scalers
- Diseño limpio, desacoplado y extensible

---

## 📦 Instalación

Clonar el repositorio en tu entorno:

```bash
git clone https://github.com/brojas7/explainable_nn.git
cd explainable_nn
pip install -e .
```

---

## 📁 Estructura del Proyecto

```text
explainable_nn/
│
├── explainable_nn/            # Código fuente del paquete
│   ├── core.py                # Red neuronal + explicabilidad
│   ├── gemini_wrapper.py      # Wrapper para LLM de Gemini
│   ├── utils.py               # Guardado y carga de modelos
│   └── __init__.py
│
├── examples/                  # Ejemplos prácticos
│   └── demo_iris.py           # Demo con dataset Iris
│
├── tests/                     # Pruebas unitarias
│   └── test_basic.py
│
├── requirements.txt
├── README.md
└── setup.py
```

---

## 🔧 Uso Rápido

### 1. Entrenamiento de una red neuronal

```python
from explainable_nn.core import ExplainableNeuralNet
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

iris = load_iris()
X = iris.data
y = iris.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

nn = ExplainableNeuralNet([4,6,6,3], learning_rate=0.02)

for epoch in range(300):
    for xi, yi in zip(X_train, y_train):
        nn.train_step(xi, yi)
```

---

### 2. Guardar el modelo entrenado

```python
from explainable_nn.utils import save_model
save_model("modelo.pkl", nn, scaler)
```

---

### 3. Cargar y realizar una predicción explicable

```python
from explainable_nn.utils import load_model
from explainable_nn.core import predict_with_explanation

saved = load_model("modelo.pkl")
nn2 = saved["modelo"]
scaler2 = saved["scaler"]

nuevo = [5.1, 3.8, 1.6, 0.2]

log, explicacion = predict_with_explanation(nn2, nuevo, scaler=scaler2)
print(log)
```

---

## 4. Explicación con Gemini

```python
from explainable_nn.gemini_wrapper import NeuralExplainerLLM

API_KEY = "TU_API_KEY"
explainer = NeuralExplainerLLM(api_key=API_KEY)

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

log, natural = predict_with_explanation(
    model=nn2,
    x_input=nuevo,
    scaler=scaler2,
    llm=explainer,
    feature_context=feature_context,
    business_context=business_context
)

print(natural)
```

---

##  Ejecución en Google Colab
[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1jT2KGJKl1UY4QF9pAPD9YFGKSaHlqGii?usp=sharing)


```python
!git clone https://github.com/brojas7/explainable_nn.git
%cd explainable_nn
!pip install -e .
```

Y luego puedes ejecutar los ejemplos desde:

```python
%run examples/demo_iris.py
```

---

## Roadmap Futuro

- Implementación modular de capas (Dense, Dropout, Normalization)
- Métricas avanzadas de explicabilidad (SHAP, LIME, Integrated Gradients)
- Visualizaciones automáticas de rutas explicables
- Implementación GPU opcional

---

## 📄 Licencia

Este proyecto está licenciado bajo **MIT License**.
Puedes usarlo libremente en proyectos personales, académicos y comerciales.

---

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Puedes abrir:
- Issues
- Pull Requests
- Mejoras en documentación
- Nuevos ejemplos o datasets

---

## 👤 Autor
**Bernal Rojas**
Profesor, Universidad Cenfotec


