import argparse
import hashlib
import igl
import json
import numpy as np
import trimesh
from pathlib import Path


def calculate_signed_distance(obj_file, n_points=100000, sigma_min=0.01, sigma_max=0.2):
    mesh = trimesh.load_mesh(obj_file)
    vertices = mesh.vertices
    centroid = mesh.centroid
    mesh.vertices = ((vertices - np.expand_dims(centroid, 0)) + 1) / 2
    faces = mesh.faces
    sample_points = mesh.sample(n_points)
    closed_points = sample_points + np.random.randn(n_points, 3) * sigma_min
    far_away_points = sample_points + np.random.randn(n_points, 3) * sigma_max
    query_points = np.concatenate((closed_points, far_away_points))
    signed_distances = igl.signed_distance(query_points, mesh.vertices, faces)[0]
    return query_points, signed_distances


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
    parser.add_argument("--ds_name", type=str, required=True, help="Dataset name")
    args = parser.parse_args()
    meshes = list(Path(args.ds_root).glob("*/*.obj"))
    for mesh_path in meshes:
        print(mesh_path)
        with open(mesh_path, "rb") as f:
            digest = hashlib.file_digest(f, "md5")
        hash_obj = digest.hexdigest()
        split = mesh_path.parent
        new_path_dir = split / hash_obj
        new_path_dir.mkdir()
        mesh_path = mesh_path.rename(new_path_dir / mesh_path.name)
        query_points, signed_distances = calculate_signed_distance(mesh_path)
        write_to_npz(
            query_points, signed_distances, Path(new_path_dir) / f"{hash_obj}.npz"
        )
        write_to_json(args.ds_name, "", split, hash_obj)
