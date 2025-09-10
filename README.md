# YOLO-v9 project for Football pipeline 

## Installation
We provide the `Dockerfile` and `docker-compose.yml` for building the docker image used for this project.
```{shell}
docker-compose up --build
docker exec -it yolov9 bash
```

## Training
We use the codes from [WongKinYu/yolov9](https://github.com/WongKinYiu/yolov9/tree/main) to train our ball detection model. For player detection model, we directly use the model weight trained on MS COCO provided by the above repo. Both model architecture are based on YOLOv9-S.

## Inference
See `scripts/inference.sh` for more details
```{shell}
bash scripts/inference.sh
```

## Visualise Detection and Tracking Result
Prepare your jsonl file and run this command:
```{shell}
cd tools
python render_specific_period.py
```

