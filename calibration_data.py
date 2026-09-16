from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: Optional[int]):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def get_c4(nsamples, seqlen, tokenizer, eval_mode=False):
    from datasets import load_dataset

    if eval_mode:
        valdata = load_dataset(
            "allenai/c4",
            "default",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            split="validation",
            revision="607bd4c8450a42878aa9ddc051a65a055450ef87",
        )
        random.seed(0)
        valenc = []
        for _ in range(256):
            while True:
                i = random.randint(0, len(valdata) - 1)
                tmp = tokenizer(valdata[i]["text"], return_tensors="pt")
                if tmp.input_ids.shape[1] >= seqlen:
                    break
            i = random.randint(0, max(tmp.input_ids.shape[1] - seqlen - 1, 0))
            valenc.append(tmp.input_ids[:, i : i + seqlen])
        return torch.hstack(valenc)

    traindata = load_dataset(
        "allenai/c4",
        "default",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
        revision="607bd4c8450a42878aa9ddc051a65a055450ef87",
    )
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        trainloader.append(trainenc.input_ids[:, i : i + seqlen])
    return trainloader


def get_loaders(
    name,
    nsamples=128,
    seed=0,
    seqlen=2048,
    eval_mode=False,
    model_path=None,
    use_fast_tokenizer=True,
    trust_remote_code=True,
):
    from transformers import AutoTokenizer

    set_seed(seed)
    if name.lower() != "c4":
        raise ValueError("CCDR pilot calibration is pinned to c4")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=use_fast_tokenizer,
        trust_remote_code=trust_remote_code,
    )
    data = get_c4(nsamples, seqlen, tokenizer, eval_mode=eval_mode)
    print(f"Loaded data from {name}; len(data)={len(data)} sequences")
    return data
