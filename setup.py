from setuptools import setup, find_packages

setup(
    name="explainable-nn",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "dill",
        "scikit-learn",
        "google-generativeai"
    ],
    author="Bernal Rojas",
    description="Red neuronal desde cero con explicabilidad + Gemini",
    license="MIT"
)
