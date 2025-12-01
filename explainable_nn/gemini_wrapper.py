#############################################################
#     EXPLICABLE NEURAL NETWORK + GEMINI (UN SOLO ARCHIVO)
#############################################################
import google.generativeai as genai


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