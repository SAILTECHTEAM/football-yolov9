import argparse
import torch
from collections import OrderedDict

def strip_module_prefix(k: str) -> str:
    return k[7:] if k.startswith("module.") else k


def add_model_prefix(k: str) -> str:
    return k if k.startswith("model.") else f"model.{k}"


def convert_checkpoint(old_ckpt: str, new_ckpt: str):
    ckpt = torch.load(old_ckpt, map_location="cpu", weights_only=False)

    # print(ckpt.keys()) 
    # print(ckpt['pytorch-lightning_version'])
    sd = ckpt["state_dict"]

    fixed = OrderedDict()
    for k, v in sd.items():
        k = strip_module_prefix(k)
        print(f"Processing key: {k}")
        print(f"Value shape: {v.shape}")
        # Add "model." if it isn't there already
        if not k.startswith("model."):
            k = f"model.{k}"
        fixed[k] = v
        print("-----")

    ckpt["state_dict"] = fixed
    torch.save(ckpt, new_ckpt)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert ParSeq checkpoint to have 'model.' prefix in state_dict keys")
    parser.add_argument("old_ckpt", type=str, help="Path to the old checkpoint file")
    parser.add_argument("new_ckpt", type=str, help="Path to save the converted checkpoint file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert_checkpoint(args.old_ckpt, args.new_ckpt)

