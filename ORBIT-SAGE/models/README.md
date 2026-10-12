# Trained model

Trained weights (YOLOv8s fine-tuned on HRSC2016).

| File | Download |
|---|---|
| `best.pt` (PyTorch weights, used by the scripts in `src/`) | https://drive.google.com/file/d/1uZAvlZzyOh5CR739k1PUbvBuWzlX4lI7/view?usp=sharing |
| `best.onnx` (exported model, for the Raspberry Pi) | https://drive.google.com/file/d/15gr0AB_p9mwMXh9GzHhxJcC9JpeoM8OL/view?usp=sharing |
| Both files in one folder | https://drive.google.com/drive/folders/1vUps5jBcH6s2rxxWyi08MPbv-TnVbVph?usp=sharing |

Put `best.pt` at `models/best.pt`, or pass `--weights path/to/best.pt` to the scripts in `src/`.

## Training setup
YOLOv8s, 100 epochs, image size 800, random 80/10/10 split with seed 42 (844 train, 105 validation, 106 test images). Detector results on the test set are in the main README.
