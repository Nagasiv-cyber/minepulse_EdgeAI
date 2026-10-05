# Model Artifacts

This directory is designated for the exported model files:
- `subsidence_model.tflite`: The quantised TensorFlow Lite model (~9.7 KB) designed for on-gateway inference on Raspberry Pi.
- `model_meta.json`: Exported scaler parameters (`scaler_mean`, `scaler_scale`), target normalization parameters (`y_mean`, `y_std`), and feature list order.

### How to generate these files:
1. Open [`notebooks/minepulse_edge_ai.ipynb`](../notebooks/minepulse_edge_ai.ipynb) on Kaggle or a local Jupyter environment.
2. Attach the [Mine Subsidence Dataset](https://www.kaggle.com/datasets/swastisomwanshi/mine-subsidence-dataset).
3. Run all cells in the notebook.
4. Download the generated `subsidence_model.tflite` and `model_meta.json` from the output directory into this folder.
