import dill

def save_model(path, model, scaler):
    with open(path, "wb") as f:
        dill.dump({"modelo": model, "scaler": scaler}, f)

def load_model(path):
    with open(path, "rb") as f:
        return dill.load(f)