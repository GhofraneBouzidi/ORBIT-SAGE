# ORBIT-SAGE: Risk-Aware Semantic Downlink for Satellite Ship Detection

**Team Umbriel** ·  Ghofrane Bouzidi, Donia El Manaa · IEEE IAS ENIG Student Branch Chapter, Gabès, Tunisia
IASTAM 6.0 Technical Challenge · Track 4 (Networks and Communications) · Problem 7: *Transmitting Information Rather Than Raw Data*

A satellite cannot send every image. ORBIT-SAGE runs a ship detector on board and chooses the **smallest packet that still lets the ground decide**:

| Mode | When | What is sent |
|---|---|---|
| M0 | no ship detected | JSON metadata only |
| M1 | confidence ≥ 0.90 | metadata + ship crops |
| M2 | 0.60 – 0.90 | metadata + crops + 256 px thumbnail |
| M3 | < 0.60 | metadata + full JPEG-50 image |
| M4 | battery critical (simulated) | stored on board, not sent |

## Results (all numbers come from the files in `results/`)

Detector: YOLOv8s fine-tuned on HRSC2016 (844 train / 105 val / 106 test, random split, seed 42, imgsz 800). Confidence threshold 0.65 chosen on validation only.

| Problem 7 indicator | Result on the 106 unseen test images |
|---|---|
| Data reduction | **81.3 %** vs raw (mean packet 57.3 KB vs 306.7 KB); 27.2 % below JPEG-50 |
| Information preserved | **96.5 %** of ground-truth ships (279 / 289) reach the ground |
| Accuracy (detector alone) | mAP@0.5 0.973, mAP@0.5:0.95 0.859, precision 95.6 %, recall 91.0 % |
| Latency | link time 1.2 s vs 6.3 s raw at 50 KB/s; detector ≈ 33 ms/image (Colab, after warm-up) |

Simulated contact window (120 s, `src/phase3_simulation.py`, share of ships delivered): nominal 50 KB/s — raw 15 %, ORBIT-SAGE 92 %, with priority queue 95 %; link-limited 10 KB/s — 2 % / 20 % / 29 %.

Empty scenes (`results/pipeline_empty_scenes.json`): on 76 ship-free harbour crops 62 % were correctly reduced to a 133-byte record (M0), **38 % were false alarms**.

## Honest limitations
- **Hardware is simulated** (virtual Raspberry Pi CM4 + ESP32-S3); nothing was measured on physical hardware, and the parts are not flight-qualified.
- **Energy is a model** (7 W inference, 3 W radio), not a measurement. The contact-window length (120 s) and the "critical battery starts at 15 %" case are prototype assumptions.
- "Raw" in the evaluation is the q95 JPEG conversion of each HRSC image, so reduction against the true original is larger.
- The detector was trained **without ship-free images**, hence the false alarms on empty scenes. All 106 test images contain ships. Training with ship-free tiles (`train_full.py --negatives 0.5`) is the next step.
- On the ship test set the single M0 decision was a miss (2 ships). Novelty detection and raw-image recovery on ground request are not implemented.
- The split is random, not the official HRSC2016 split.

## Reproduce
Requires a Kaggle token (dataset `guofeng/hrsc2016`) and a Colab GPU for training; evaluation also runs on CPU.
1. `notebooks/ORBIT_SAGE_train_full.ipynb` : download data, convert to YOLO, train, export ONNX (≈1.5–3 h on a T4).
2. `notebooks/ORBIT_SAGE_reconnect_and_empty_test.ipynb` : rebuild the same split from a saved model, re-run the evaluation, run the empty-scene test (no retraining).
3. Command line equivalents:
```
python src/train_full.py --root HRSC2016_extracted --data hrsc_yolo --negatives 0 --epochs 100 --imgsz 800
python src/evaluate_pipeline.py --data hrsc_yolo --weights models/best.pt --out results/pipeline_results.json
python src/phase3_simulation.py --results results/pipeline_results.json --window_s 120
python src/make_empty_tiles.py --data hrsc_yolo --out hrsc_empty
python src/make_example.py --data hrsc_yolo --weights models/best.pt
```
`--negatives 0` reproduces the reported run (no ship-free training tiles).

## Repository layout
`src/` code · `notebooks/` Colab notebooks · `results/` raw measurements · `examples/` real packets for M1, M2, M3 · `models/` how to get the trained weights.

Dataset: HRSC2016 — Liu et al., *HRSC2016: a high-resolution ship detection dataset in Google Earth images*, IEEE DataPort, 2017.
