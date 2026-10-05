import json
import numpy as np

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite import Interpreter


def load_model(meta_path="models/model_meta.json", model_path="models/subsidence_model.tflite"):
    with open(meta_path, "r") as f:
        meta = json.load(f)
    interp = Interpreter(model_path=model_path)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    return meta, interp, inp, out


def predict(reading: dict, meta=None, interp=None, inp=None, out=None):
    if meta is None or interp is None:
        meta, interp, inp, out = load_model()

    # 1. Order features exactly as in training, then standardise
    x = np.array([reading[f] for f in meta["features"]], dtype=np.float32)
    x = (x - np.array(meta["scaler_mean"])) / np.array(meta["scaler_scale"])

    # 2. Run the model
    interp.set_tensor(inp["index"], x[None, :].astype(np.float32))
    interp.invoke()
    z = interp.get_tensor(out["index"])[0][0]

    # 3. Undo the target transform -> subsidence rate in mm/day
    rate = max(float(np.expm1(z * meta["y_std"] + meta["y_mean"])), 0.0)

    # 4. Map to an alert level
    level = "SAFE" if rate < 0.5 else "WARNING" if rate < 5 else "DANGER"
    return rate, level


if __name__ == "__main__":
    try:
        meta, interp, inp, out = load_model()
        # Example: replace with a real feature dict from your gateway
        sample = {f: 0.0 for f in meta["features"]}
        rate, level = predict(sample, meta, interp, inp, out)
        print(f"Prediction: {rate:.4f} mm/day -> Alert level: {level}")
    except FileNotFoundError:
        print("Note: models/model_meta.json and models/subsidence_model.tflite not found.")
        print("Run notebooks/minepulse_edge_ai.ipynb to generate them.")
