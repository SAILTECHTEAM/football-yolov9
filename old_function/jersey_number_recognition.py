# Standalone code for testing only
import os
import json
import string
import argparse
from PIL import Image
from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint, parse_model_args
from tqdm import tqdm

def run_inference(model, data_root, result_file, img_size):
    # load images one by one, save paths and result
    file_dir = os.path.join(data_root, 'imgs')
    filenames = os.listdir(file_dir)
    filenames.sort()
    results = {}
    for filename in tqdm(filenames):
        image = Image.open(os.path.join(file_dir, filename)).convert('RGB')
        transform = SceneTextDataModule.get_transform(img_size)
        image = transform(image)
        image = image.unsqueeze(0)
        logits = model.forward(image.to(model.device))
        #convert to 3 by 10
        probs_full = logits[:,:3,:11].softmax(-1)
        preds, probs = model.tokenizer.decode(probs_full)
        logits = logits[:,:3,:11].cpu().detach().numpy()[0].tolist()
        # probs = logits.softmax(-1)
        # preds, probs = model.tokenizer.decode(probs)
        probs_full = probs_full.cpu().detach().numpy()[0].tolist()
        confidence = probs[0].cpu().detach().numpy().squeeze().tolist()
        results[filename] = {'label':preds[0], 'confidence':confidence, 'raw': probs_full, 'logits':logits}
    with open(result_file, 'w') as f:
        json.dump(results, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help="Model checkpoint (or 'pretrained=<model_id>')")
    parser.add_argument('--data_root', default='data')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--cased', action='store_true', default=False, help='Cased comparison')
    parser.add_argument('--punctuation', action='store_true', default=False, help='Check punctuation')
    parser.add_argument('--std', action='store_true', default=False, help='Evaluate on standard benchmark datasets')
    parser.add_argument('--new', action='store_true', default=False, help='Evaluate on new benchmark datasets')
    parser.add_argument('--custom', action='store_true', default=True, help='Evaluate on custom personal datasets')
    parser.add_argument('--rotation', type=int, default=0, help='Angle of rotation (counter clockwise) in degrees.')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--inference', action='store_true', default=False, help='Run inference and store prediction results')
    parser.add_argument('--tune_temperature', action='store_true', default=False,
                        help='Find best t-scale')
    parser.add_argument('--result_file', default='outputs/preds.json')
    args, unknown = parser.parse_known_args()
    kwargs = parse_model_args(unknown)

    charset_test = string.digits # + string.ascii_lowercase
    if args.cased:
        charset_test += string.ascii_uppercase
    if args.punctuation:
        charset_test += string.punctuation
    kwargs.update({'charset_test': charset_test})
    print(f'Additional keyword arguments: {kwargs}')

    model = load_from_checkpoint(args.checkpoint, **kwargs).eval().to(args.device)
    hp = model.hparams

    if args.inference:
        run_inference(model, args.data_root, args.result_file, hp.img_size) # hp.img_size = (32, 128)
        exit()

