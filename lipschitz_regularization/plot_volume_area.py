import argparse
import torch
import trimesh
import matplotlib.pyplot as plt
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder_lipschitzmlp",
        "-lmlp",
        type=str,
        required=True,
        help="Path to the interpolations of Lipschitz.",
    )
    parser.add_argument(
        "--folder_standardmlp",
        "-smlp",
        type=str,
        required=True,
        help="Path to the interpolations of Lipschitz.",
    )
    args = parser.parse_args()
    mesh_paths = list(Path(args.folder_lipschitzmlp).glob("*.ply"))
    volume_lipschitz = []
    area_lipschitz = []
    x = torch.linspace(0, 1, 100)
    for mesh_path in mesh_paths:
        print(mesh_path.name)
        mesh_path = trimesh.load(mesh_path)
        volume_lipschitz.append(mesh_path.volume)
        area_lipschitz.append(mesh_path.area)
    volume_mlp = []
    area_mlp = []
    mesh_paths = list(Path(args.folder_standardmlp).glob("*.ply"))
    for mesh_path in mesh_paths:
        print(mesh_path.name)
        mesh_path = trimesh.load(mesh_path)
        volume_mlp.append(mesh_path.volume)
        area_mlp.append(mesh_path.area)
    plt.rcParams.update({"font.size": 20})
    fig, ax = plt.subplots()
    ax.plot(x, area_lipschitz, linestyle="-", color="green", label="Lipschitz MLP")
    ax.plot(x, volume_lipschitz, linestyle="-", color="green", label="Lipschitz MLP")
    ax.plot(x, area_mlp, linestyle="-", color="red", label="Standard MLP")
    ax.plot(x, volume_mlp, linestyle="-", color="red", label="Standard MLP")
    ax.set_xlabel("Interpolation Steps")
    ax.set_ylabel("Volume ($m^3$)")
    ax.legend(loc="upper right")
    plt.show()
