import torch
import argparse
from models.yolo import Model
import os

def ensure_directory_exists(file_path):
    """Create all directories in the path to a file if they don't exist."""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    return directory

def convert_yolov9(config_path, checkpoint_path, num_classes, output_path):
    """
    Convert YOLOv9 model from training mode to inference mode by reparameterizing.
    
    Args:
        config_path: Path to model config yaml file
        checkpoint_path: Path to checkpoint pt file
        num_classes: Number of classes
        output_path: Path to save the converted model
    """
    device = torch.device("cpu")
    
    # Initialize model from config
    model = Model(config_path, ch=3, nc=num_classes, anchors=3)
    model = model.to(device)
    _ = model.eval()
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    model.names = ckpt['model'].names
    model.nc = ckpt['model'].nc
    
    # Determine model architecture based on config path
    model_type = None
    if "gelan-s" in config_path:
        model_type = "s"
        print("Detected YOLOv9-S architecture")
    elif "gelan-m" in config_path:
        model_type = "m"
        print("Detected YOLOv9-M architecture")
    elif "gelan-c" in config_path:
        model_type = "c"
        print("Detected YOLOv9-C architecture")
    elif "gelan-e" in config_path:
        model_type = "e"
        print("Detected YOLOv9-E architecture")
    else:
        raise ValueError(f"Unsupported model architecture in config: {config_path}")
    
    # Reparameterize the model based on its architecture
    idx = 0
    print(f"Reparameterizing YOLOv9-{model_type.upper()} model...")
    
    # YOLOv9-S
    if model_type == "s":
        for k, v in model.state_dict().items():
            if "model.{}.".format(idx) in k:
                if idx < 22:
                    kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx))
                    model.state_dict()[k] -= model.state_dict()[k]
                    model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                    print(k, "perfectly matched!!")
                elif "model.{}.cv2.".format(idx) in k:
                    kr = k.replace("model.{}.cv2.".format(idx), "model.{}.cv4.".format(idx+7))
                    model.state_dict()[k] -= model.state_dict()[k]
                    model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                    print(k, "perfectly matched!!")
                elif "model.{}.cv3.".format(idx) in k:
                    kr = k.replace("model.{}.cv3.".format(idx), "model.{}.cv5.".format(idx+7))
                    model.state_dict()[k] -= model.state_dict()[k]
                    model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                    print(k, "perfectly matched!!")
                elif "model.{}.dfl.".format(idx) in k:
                    kr = k.replace("model.{}.dfl.".format(idx), "model.{}.dfl2.".format(idx+7))
                    model.state_dict()[k] -= model.state_dict()[k]
                    model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                    print(k, "perfectly matched!!")
            else:
                while True:
                    idx += 1
                    if "model.{}.".format(idx) in k:
                        break
                if idx < 22:
                    kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx))
                    model.state_dict()[k] -= model.state_dict()[k]
                    model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                    print(k, "perfectly matched!!")
                elif "model.{}.cv2.".format(idx) in k:
                    kr = k.replace("model.{}.cv2.".format(idx), "model.{}.cv4.".format(idx+7))
                    model.state_dict()[k] -= model.state_dict()[k]
                    model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                    print(k, "perfectly matched!!")
                elif "model.{}.cv3.".format(idx) in k:
                    kr = k.replace("model.{}.cv3.".format(idx), "model.{}.cv5.".format(idx+7))
                    model.state_dict()[k] -= model.state_dict()[k]
                    model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                    print(k, "perfectly matched!!")
                elif "model.{}.dfl.".format(idx) in k:
                    kr = k.replace("model.{}.dfl.".format(idx), "model.{}.dfl2.".format(idx+7))
                    model.state_dict()[k] -= model.state_dict()[k]
                    model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                    print(k, "perfectly matched!!")
    
    # YOLOv9-M or YOLOv9-C
    elif model_type in ["m", "c"]:
        for k, v in model.state_dict().items():
            if "model.{}.".format(idx) in k:
                if idx < 22:
                    kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx+1))
                    model.state_dict()[k] -= model.state_dict()[k]
                    model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                    if model_type == "m":
                        print(k, "perfectly matched!!")
                elif "model.{}.cv2.".format(idx) in k:
                    kr = k.replace("model.{}.cv2.".format(idx), "model.{}.cv4.".format(idx+16))
                    model.state_dict()[k] -= model.state_dict()[k]
                    model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                    if model_type == "m":
                        print(k, "perfectly matched!!")
                elif "model.{}.cv3.".format(idx) in k:
                    kr = k.replace("model.{}.cv3.".format(idx), "model.{}.cv5.".format(idx+16))
                    model.state_dict()[k] -= model.state_dict()[k]
                    model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                    if model_type == "m":
                        print(k, "perfectly matched!!")
                elif "model.{}.dfl.".format(idx) in k:
                    kr = k.replace("model.{}.dfl.".format(idx), "model.{}.dfl2.".format(idx+16))
                    model.state_dict()[k] -= model.state_dict()[k]
                    model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                    if model_type == "m":
                        print(k, "perfectly matched!!")
            else:
                while True:
                    idx += 1
                    if "model.{}.".format(idx) in k:
                        break
                if idx < 22:
                    kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx+1))
                    model.state_dict()[k] -= model.state_dict()[k]
                    model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                    if model_type == "m":
                        print(k, "perfectly matched!!")
                elif "model.{}.cv2.".format(idx) in k:
                    kr = k.replace("model.{}.cv2.".format(idx), "model.{}.cv4.".format(idx+16))
                    model.state_dict()[k] -= model.state_dict()[k]
                    model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                    if model_type == "m":
                        print(k, "perfectly matched!!")
                elif "model.{}.cv3.".format(idx) in k:
                    kr = k.replace("model.{}.cv3.".format(idx), "model.{}.cv5.".format(idx+16))
                    model.state_dict()[k] -= model.state_dict()[k]
                    model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                    if model_type == "m":
                        print(k, "perfectly matched!!")
                elif "model.{}.dfl.".format(idx) in k:
                    kr = k.replace("model.{}.dfl.".format(idx), "model.{}.dfl2.".format(idx+16))
                    model.state_dict()[k] -= model.state_dict()[k]
                    model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                    if model_type == "m":
                        print(k, "perfectly matched!!")
    
    # YOLOv9-E
    elif model_type == "e":
        for k, v in model.state_dict().items():
            if "model.{}.".format(idx) in k:
                if idx < 29:
                    kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx))
                    model.state_dict()[k] -= model.state_dict()[k]
                    model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                    print(k, "perfectly matched!!")
                elif idx < 42:
                    kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx+7))
                    model.state_dict()[k] -= model.state_dict()[k]
                    model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                    print(k, "perfectly matched!!")
                elif "model.{}.cv2.".format(idx) in k:
                    kr = k.replace("model.{}.cv2.".format(idx), "model.{}.cv4.".format(idx+7))
                    model.state_dict()[k] -= model.state_dict()[k]
                    model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                    print(k, "perfectly matched!!")
                elif "model.{}.cv3.".format(idx) in k:
                    kr = k.replace("model.{}.cv3.".format(idx), "model.{}.cv5.".format(idx+7))
                    model.state_dict()[k] -= model.state_dict()[k]
                    model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                    print(k, "perfectly matched!!")
                elif "model.{}.dfl.".format(idx) in k:
                    kr = k.replace("model.{}.dfl.".format(idx), "model.{}.dfl2.".format(idx+7))
                    model.state_dict()[k] -= model.state_dict()[k]
                    model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                    print(k, "perfectly matched!!")
            else:
                while True:
                    idx += 1
                    if "model.{}.".format(idx) in k:
                        break
                if idx < 29:
                    kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx))
                    model.state_dict()[k] -= model.state_dict()[k]
                    model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                    print(k, "perfectly matched!!")
                elif idx < 42:
                    kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx+7))
                    model.state_dict()[k] -= model.state_dict()[k]
                    model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                    print(k, "perfectly matched!!")
                elif "model.{}.cv2.".format(idx) in k:
                    kr = k.replace("model.{}.cv2.".format(idx), "model.{}.cv4.".format(idx+7))
                    model.state_dict()[k] -= model.state_dict()[k]
                    model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                    print(k, "perfectly matched!!")
                elif "model.{}.cv3.".format(idx) in k:
                    kr = k.replace("model.{}.cv3.".format(idx), "model.{}.cv5.".format(idx+7))
                    model.state_dict()[k] -= model.state_dict()[k]
                    model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                    print(k, "perfectly matched!!")
                elif "model.{}.dfl.".format(idx) in k:
                    kr = k.replace("model.{}.dfl.".format(idx), "model.{}.dfl2.".format(idx+7))
                    model.state_dict()[k] -= model.state_dict()[k]
                    model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                    print(k, "perfectly matched!!")
    
    _ = model.eval()
    
    # Create checkpoint dictionary
    m_ckpt = {
        'model': model.half(),
        'optimizer': None,
        'best_fitness': None,
        'ema': None,
        'updates': None,
        'opt': None,
        'git': None,
        'date': None,
        'epoch': -1
    }
    
    # Ensure output directory exists
    ensure_directory_exists(output_path)
    
    # Save the converted model
    torch.save(m_ckpt, output_path)
    print(f"✅ Converted YOLOv9-{model_type.upper()} model saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="YOLOv9 Model Reparameterization")
    parser.add_argument("--config", type=str, required=True, 
                        help="Path to model config YAML (e.g., ./models/detect/gelan-s.yaml)")
    parser.add_argument("--checkpoint", type=str, required=True, 
                        help="Path to checkpoint PT file (e.g., ./checkpoint/yolov9-s.pt)")
    parser.add_argument("--classes", type=int, required=True, 
                        help="Number of classes in the model")
    parser.add_argument("--output", type=str, required=True, 
                        help="Path to save the converted model (e.g., ./checkpoint/yolov9-s-converted.pt)")
    
    args = parser.parse_args()
    
    convert_yolov9(args.config, args.checkpoint, args.classes, args.output)

if __name__ == "__main__":
    main()
# Example usage:
# python reparam-yolov9.py --config ./models/detect/gelan-s.yaml --checkpoint ./checkpoint/yolov9-s.pt --classes 80 --output ./checkpoint/yolov9-s-converted.pt