import torch
from datasets import load_dataset, load_from_disk
from typing import Any, Mapping
import json
import numpy as np
import os
from statistics import mean, stdev


def read_json(filename: str) -> Mapping[str, Any]:
    """Returns a Python dict representation of JSON object at input file."""
    with open(filename) as fp:
        return json.load(fp)


def get_dataset(args):
    if 'laion' in args.dataset_path:
        dataset = load_dataset(args.dataset)['train']
        prompt_key = 'TEXT'
    elif 'coco' in args.dataset_path:
        with open('fid_outputs/coco/meta_data.json') as f:
            dataset = json.load(f)
            dataset = dataset['annotations']
            prompt_key = 'caption'
    else:
        if os.path.isdir(args.dataset_path):
            loaded = load_from_disk(args.dataset_path)
            # load_from_disk may return Dataset or DatasetDict
            if hasattr(loaded, 'column_names') and isinstance(loaded.column_names, list):
                # It's a Dataset directly
                dataset = loaded
            else:
                # It's a DatasetDict, get 'train' split
                dataset = loaded['train']
        else:
            # Try loading with local cache first to avoid repeated downloads
            local_cache = os.path.join('.cache_datasets', args.dataset_path.replace('/', '_'))
            if os.path.isdir(local_cache):
                print(f"[INFO] Loading dataset from local cache: {local_cache}")
                dataset = load_from_disk(local_cache)
                # load_from_disk may return Dataset or DatasetDict
                if not (hasattr(dataset, 'column_names') and isinstance(dataset.column_names, list)):
                    dataset = dataset['train']
            else:
                print(f"[INFO] Downloading dataset: {args.dataset_path}")
                dataset = load_dataset(args.dataset_path)['train']
                # Save to local cache for future runs
                os.makedirs(os.path.dirname(local_cache), exist_ok=True)
                try:
                    dataset.save_to_disk(local_cache)
                    print(f"[INFO] Dataset cached to: {local_cache}")
                except Exception as e:
                    print(f"[WARN] Failed to cache dataset: {e}")
        if 'Prompt' in dataset.column_names:
            prompt_key = 'Prompt'
        elif 'prompt' in dataset.column_names:
            prompt_key = 'prompt'
        elif 'text' in dataset.column_names:
            prompt_key = 'text'
        else:
            raise KeyError(f"Cannot find prompt column. Available columns: {dataset.column_names}")
    return dataset, prompt_key


def save_metrics(args, tpr_detection, tpr_traceability, acc, clip_scores):
    names = {
        'jpeg_ratio': "Jpeg.txt",
        'random_crop_ratio': "RandomCrop.txt",
        'random_drop_ratio': "RandomDrop.txt",
        'gaussian_blur_r': "GauBlur.txt",
        'gaussian_std': "GauNoise.txt",
        'median_blur_k': "MedBlur.txt",
        'resize_ratio': "Resize.txt",
        'sp_prob': "SPNoise.txt",
        'brightness_factor': "Color_Jitter.txt"
    }
    filename = "Identity.txt"
    for option, name in names.items():
        if getattr(args, option) is not None:
            filename = name

    if args.reference_model is not None:
        with open(args.output_path + filename, "a") as file:
            file.write('tpr_detection:' + str(tpr_detection / args.num) + '      ' +
                       'tpr_traceability:' + str(tpr_traceability / args.num) + '      ' +
                       'mean_acc:' + str(mean(acc)) + '      ' + 'std_acc:' + str(stdev(acc)) + '      ' +
                       'mean_clip_score:' + str(mean(clip_scores)) + '      ' + 'std_clip_score:' + str(stdev(clip_scores)) + '      ' +
                       '\n')

    else:
        with open(args.output_path + filename, "a") as file:
            file.write('tpr_detection:' + str(tpr_detection / args.num) + '      ' +
                       'tpr_traceability:' + str(tpr_traceability / args.num) + '      ' +
                       'mean_acc:' + str(mean(acc)) + '      ' + 'std_acc:' + str(stdev(acc)) + '      ' +
                       '\n')

    return





