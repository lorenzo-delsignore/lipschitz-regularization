import torch
import numpy as np
from pathlib import Path
import argparse
import json


def sphere(xyz):
    c = torch.tensor([[0.5, 0.5, 0.5]], device=xyz.device)
    return (xyz - c).norm(dim=-1, keepdim=True) - 0.3


def torus(xyz):
    t = torch.tensor([0.2, 0.1], device=xyz.device)
    c = torch.tensor([[0.5, 0.5, 0.5]], device=xyz.device)
    p = xyz - c
    q = torch.cat(
        [p[..., [0, 2]].norm(dim=-1, keepdim=True) - t[0:1, None], p[..., 1:2]], dim=-1
    )
    return q.norm(dim=-1, keepdim=True) - t[1:, None]


def write_to_npz(xyz, sdfs, filename):
    num_vert = len(xyz)
    pos = []
    neg = []

    for i in range(num_vert):
        v = xyz[i]
        s = sdfs[i]
        if s > 0:
            for j in range(3):
                pos.append(v[j])
            pos.append(s)
        else:
            for j in range(3):
                neg.append(v[j])
            neg.append(s)
    np.savez(
        filename,
        pos=np.array(pos).reshape(-1, 4).astype(np.float32),
        neg=np.array(neg).reshape(-1, 4).astype(np.float32),
    )


def write_to_json(dataset_name, category, mesh_split, hash):
    file_path = Path(mesh_split) / f"{dataset_name}_{mesh_split.name}.json"
    try:
        with open(file_path, "r") as json_file:
            data = json.load(json_file)
    except FileNotFoundError:
        data = {dataset_name: {category: []}}
    data[dataset_name][category].append(hash)
    with open(file_path, "w") as json_file:
        json.dump(data, json_file, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ds_root", type=str, required=True, help="Path to the dataset root directory."
    )
    parser.add_argument(
        "--ds_name", type=str, required=True, help="Path to the dataset root directory."
    )
    args = parser.parse_args()
    dataset_path = Path(args.ds_root)

    sigma = 0.2
    center = torch.tensor([[0.5, 0.5, 0.5]])
    nsamples = 100000
    samples_normal = center + torch.randn(int(nsamples / 2), 3) * sigma
    samples_unif = torch.rand(int(nsamples / 2), 3)  # questi sono già in [0; 1]
    samples = torch.cat([samples_normal, samples_unif], dim=0)
    signed_distances_sphere = sphere(samples).flatten().tolist()
    signed_distances_torus = torus(samples).flatten().tolist()
    samples = samples.tolist()
    write_to_npz(samples, signed_distances_sphere, dataset_path / "sphere.npz")
    write_to_npz(samples, signed_distances_torus, dataset_path / "torus.npz")
    write_to_json(args.ds_name, "", dataset_path / "train", "0")
    write_to_json(args.ds_name, "", dataset_path / "train", "1")
