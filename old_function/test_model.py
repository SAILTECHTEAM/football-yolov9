from pathlib import Path

from models.common import DetectMultiBackend
model = DetectMultiBackend("./weight/yolov9-s-converted.pt")
print(model.pt)  # True = regular PyTorch, False = TorchScript