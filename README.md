# minepulse_EdgeAI

Edge AI inference pipeline for early warning of ground subsidence on edge devices (such as Raspberry Pi) using TensorFlow Lite.

## Features

- **Feature Engineering**: Incremental rolling buffer (3h, 6h, 24h stats), EWMA calculation, tilt rates, and accelerations matching model training pipeline.
- **Persistent State**: Recovers rolling buffer and EWMA state across power loss or reboots via atomic JSON state saves.
- **Two Execution Modes**:
  - `run_live()`: Hourly sensor ingestion loop with alert callbacks (`SAFE`, `WARNING`, `DANGER`).
  - `run_replay()`: Evaluation and feature parity verification against historical training/test CSV data.

## Requirements & Prerequisites

### Required Files
Place the following model artifacts in the same directory as the script:
- `subsidence_model.tflite`
- `model_meta.json`

### Dependencies
On Raspberry Pi OS (64-bit) or local environment:
```bash
pip install numpy ai-edge-litert
```
*(Fallback for LiteRT)*:
```bash
pip install tflite-runtime
```

## Usage

### Live Sensor Loop
Runs the continuous hourly monitoring loop:
```bash
python3 subsidence_edge.py
```

### Replay & Parity Validation
Replays historical CSV records to benchmark against ground-truth and check feature parity:
```bash
python3 subsidence_edge.py --replay data.csv
```
