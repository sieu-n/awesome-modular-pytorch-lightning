import os

import requests
import torch
from tqdm import tqdm


def download_model_state_dict(url, name="./pretrained.pth"):
    if not os.path.exists(os.path.dirname(name)):
        os.makedirs(os.path.dirname(name))
    # download from url
    print(f"Downloading pretrained model state dict into `{name}` from: {url}")
    r = requests.get(url, allow_redirects=True)
    open(name, "wb").write(r.content)
    return name


def load_model_weights(
    model, state_dict_path="./pretrained.pth", url=None, is_ckpt=False
):
    if url is not None:
        state_dict_path = download_model_state_dict(url=url, name=state_dict_path)

    assert os.path.exists(state_dict_path), "Pre-trained model not found!"

    print("Loading model weights")
    state = torch.load(state_dict_path, map_location="cpu")
    if is_ckpt:
        state = state["state_dict"]
    for key in tqdm(model.state_dict()):
        if "num_batches_tracked" in key:
            continue
        p = model.state_dict()[key]
        if key in state:
            ip = state[key]
            if p.shape == ip.shape:
                p.data.copy_(ip.data)  # Copy the data of parameters
            else:
                print(
                    "could not load layer: {}, mismatch shape {} ,{}".format(
                        key, (p.shape), (ip.shape)
                    )
                )
        else:
            print("could not load layer: {}, not in checkpoint".format(key))
    return model
