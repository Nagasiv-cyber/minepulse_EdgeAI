#!/usr/bin/env python3
"""
Subsidence early-warning inference for Raspberry Pi (TFLite).

Files needed next to this script:
    subsidence_model.tflite   model_meta.json   (both from the Kaggle notebook)

Usage:
    python3 subsidence_edge.py                       # live loop, one reading per hour
    python3 subsidence_edge.py --replay data.csv     # test on the training CSV (no sensors needed)

Install (Pi OS 64-bit):  pip install numpy ai-edge-litert
                         (fallback: pip install tflite-runtime)
"""
import argparse, csv, json, math, os, time
from collections import deque
from datetime import datetime

import numpy as np

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite import Interpreter

MODEL_PATH = "subsidence_model.tflite"
META_PATH = "model_meta.json"
STATE_PATH = "edge_state.json"          # survives reboots / power loss
LOG_PATH = "predictions_log.csv"

SAMPLE_PERIOD_S = 3600                  # training data is ONE reading per hour
MAX_GAP_S = 3 * SAMPLE_PERIOD_S         # longer gap -> history is stale, warm up again
WINDOW = 24                             # longest rolling window (hours)
EWMA_ALPHA = 2 / (60 + 1)               # matches training: ewm(span=60, adjust=False)


# --------------------------------------------------------------------------- #
# Feature engineering: must reproduce the training columns exactly.
# --------------------------------------------------------------------------- #
class FeatureBuilder:
    def __init__(self, state_path=None):
        self.state_path = state_path
        self.buf = deque(maxlen=WINDOW)   # (temp, pressure, humidity, tilt)
        self.ewma = None
        self.prev_rate = None
        self.last_ts = None
        if state_path and os.path.exists(state_path):
            self._load()

    # ---- persistence -------------------------------------------------------
    def _load(self):
        try:
            s = json.load(open(self.state_path))
            self.buf = deque((tuple(r) for r in s["buf"]), maxlen=WINDOW)
            self.ewma, self.prev_rate, self.last_ts = s["ewma"], s["prev_rate"], s["last_ts"]
            print(f"Restored state: {len(self.buf)}/{WINDOW} readings buffered")
        except Exception as e:
            print("Could not load state, starting fresh:", e)

    def _save(self):
        if not self.state_path:
            return
        tmp = self.state_path + ".tmp"
        json.dump({"buf": list(self.buf), "ewma": self.ewma,
                   "prev_rate": self.prev_rate, "last_ts": self.last_ts}, open(tmp, "w"))
        os.replace(tmp, self.state_path)  # atomic: no half-written file if power drops

    # ---- main entry --------------------------------------------------------
    def update(self, ts, temp, pres, hum, tilt):
        """Feed one hourly reading. Returns feature dict, or None while warming up."""
        if self.last_ts is not None and ts - self.last_ts > MAX_GAP_S:
            print("Gap in readings too long; restarting warm-up")
            self.buf.clear(); self.ewma = None; self.prev_rate = None

        prev_tilt = self.buf[-1][3] if self.buf else None
        rate = None if prev_tilt is None else (tilt - prev_tilt) / 3600.0
        accel = None if (rate is None or self.prev_rate is None) else (rate - self.prev_rate) / 3600.0
        self.ewma = tilt if self.ewma is None else EWMA_ALPHA * tilt + (1 - EWMA_ALPHA) * self.ewma
        self.prev_rate = rate
        self.buf.append((temp, pres, hum, tilt))
        self.last_ts = ts
        self._save()

        if len(self.buf) < WINDOW or accel is None:
            return None

        arr = np.array(self.buf, dtype=np.float64)
        f = {
            "temperature_celsius": temp,
            "pressure_kpa": pres,
            "humidity_pct": hum,
            "tilt_degrees": tilt,
            "tilt_magnitude_radians": math.radians(tilt),
            "tilt_rate": rate,
            "tilt_acceleration": accel,
            "tilt_magnitude_ewma": self.ewma,
            "tilt_humidity_interaction": tilt * hum,
            "tilt_magnitude_temp_corrected": tilt - 0.01 * temp,
        }
        for col, name in ((3, "tilt_degrees"), (1, "pressure_kpa"), (2, "humidity_pct")):
            for w, tag in ((3, "3h"), (6, "6h"), (24, "24h")):
                x = arr[-w:, col]
                f[f"{name}_mean_{tag}"] = x.mean()
                f[f"{name}_std_{tag}"] = x.std(ddof=1)     # pandas default (sample std)
        f["pressure_anomaly"] = pres - f["pressure_kpa_mean_24h"]
        return f


# --------------------------------------------------------------------------- #
# Model wrapper
# --------------------------------------------------------------------------- #
class Predictor:
    def __init__(self):
        self.meta = json.load(open(META_PATH))
        self.interp = Interpreter(model_path=MODEL_PATH)
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]
        self.mean = np.array(self.meta["scaler_mean"], dtype=np.float32)
        self.scale = np.array(self.meta["scaler_scale"], dtype=np.float32)
        self.log_target = self.meta.get("log_target", False)
        print("Target decoding:", "expm1 (log target)" if self.log_target
              else "linear  <-- if you trained with log1p, add \"log_target\": true to model_meta.json")

    def rate_mm_day(self, feats):
        x = np.array([feats[n] for n in self.meta["features"]], dtype=np.float32)
        x = ((x - self.mean) / self.scale)[None, :].astype(np.float32)
        self.interp.set_tensor(self.inp["index"], x)
        self.interp.invoke()
        z = float(self.interp.get_tensor(self.out["index"]).ravel()[0])
        v = z * self.meta["y_std"] + self.meta["y_mean"]
        return max(0.0, math.expm1(v) if self.log_target else v)


def level(mm):
    return "SAFE" if mm < 0.5 else "WARNING" if mm < 5 else "DANGER"


# --------------------------------------------------------------------------- #
# Sensors: fill this in for your hardware
# --------------------------------------------------------------------------- #
def read_sensors():
    """Return (temp_celsius, pressure_kpa, humidity_pct, tilt_degrees).

    Units must match training: pressure in kPa (a BME280 reports hPa -> divide by 10),
    tilt in degrees with the same zero point and sign convention as the training data.
    Take a short average (e.g. 10-60 samples) to reduce noise.
    """
    raise NotImplementedError("Add your sensor code here (e.g. BME280 + an inclinometer)")


def on_alert(lvl, mm):
    """Hook for a buzzer / LED / SMS / MQTT message."""
    print(f"  !!! {lvl} alert: {mm:.2f} mm/day")


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #
def run_live():
    fb, model = FeatureBuilder(STATE_PATH), Predictor()
    new_log = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="") as fh:
        w = csv.writer(fh)
        if new_log:
            w.writerow(["time", "rate_mm_day", "level"])
        while True:
            t0 = time.time()
            feats = fb.update(t0, *read_sensors())
            stamp = datetime.fromtimestamp(t0).isoformat(timespec="seconds")
            if feats is None:
                print(f"{stamp}  warming up ({len(fb.buf)}/{WINDOW} readings)")
            else:
                mm = model.rate_mm_day(feats)
                lvl = level(mm)
                print(f"{stamp}  {mm:6.2f} mm/day  {lvl}")
                w.writerow([stamp, round(mm, 3), lvl]); fh.flush()
                if lvl != "SAFE":
                    on_alert(lvl, mm)
            time.sleep(max(0, SAMPLE_PERIOD_S - (time.time() - t0)))


def run_replay(csv_path, eval_from=0.8, parity_from=300):
    """Feed the training CSV through the exact live code path and check it."""
    rows = list(csv.DictReader(open(csv_path)))
    fb, model = FeatureBuilder(None), Predictor()
    start = int(len(rows) * eval_from)
    max_diff, y_true, y_pred, lv_true, lv_pred = {}, [], [], [], []

    for i, r in enumerate(rows):
        ts = datetime.strptime(r["timestamp"], "%Y-%m-%d %H:%M:%S").timestamp()
        feats = fb.update(ts, *(float(r[k]) for k in
                        ("temperature_celsius", "pressure_kpa", "humidity_pct", "tilt_degrees")))
        if feats is None:
            continue
        if i >= parity_from:   # EWMA needs a few days to forget its start value
            for k in model.meta["features"]:
                d = abs(feats[k] - float(r[k]))
                max_diff[k] = max(max_diff.get(k, 0.0), d)
        if i >= start:
            mm = model.rate_mm_day(feats)
            y_true.append(float(r["subsidence_rate_mm_day"])); y_pred.append(mm)
            lv_true.append(r["danger_level"]); lv_pred.append(level(mm))

    print(f"\nFeature parity vs CSV (rows >= {parity_from}); worst 5 by relative error:")
    rel = {k: d / (abs(np.std([float(r[k]) for r in rows])) + 1e-12) for k, d in max_diff.items()}
    for k in sorted(rel, key=rel.get, reverse=True)[:5]:
        print(f"  {k:32s} max abs diff {max_diff[k]:.3e}  (relative to feature std {rel[k]:.1e})")

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    lv_true, lv_pred = np.array(lv_true), np.array(lv_pred)
    ss_res, ss_tot = ((y_true - y_pred) ** 2).sum(), ((y_true - y_true.mean()) ** 2).sum()
    print(f"\nReplay on last {len(y_true)} rows:  MAE {np.abs(y_true - y_pred).mean():.3f}  "
          f"R2 {1 - ss_res / ss_tot:.3f}")
    print(f"Danger-level accuracy {np.mean(lv_true == lv_pred):.3f}")
    for c in ("DANGER", "WARNING", "SAFE"):
        m = lv_true == c
        print(f"  {c:8s} recall {np.mean(lv_pred[m] == c):.2f}  ({m.sum()} rows)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", metavar="CSV", help="test against the training CSV instead of live sensors")
    args = ap.parse_args()
    run_replay(args.replay) if args.replay else run_live()
