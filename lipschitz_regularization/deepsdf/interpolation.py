#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import argparse
import json
import logging
import os
import random
import time
import torch
from pathlib import Path
import random
import trimesh

import lipschitz_regularization.deepsdf.deep_sdf
import lipschitz_regularization.deepsdf.deep_sdf.workspace as ws


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(
        description="Use a trained DeepSDF decoder to reconstruct a shape given SDF "
        + "samples."
    )
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory which includes specifications and saved model "
        + "files to use for reconstruction",
    )
    arg_parser.add_argument(
        "--checkpoint",
        "-c",
        dest="checkpoint",
        default="latest",
        help="The checkpoint weights to use. This can be a number indicated an epoch "
        + "or 'latest' for the latest weights (this is the default)",
    )
    arg_parser.add_argument(
        "--data",
        "-d",
        dest="data_source",
        required=True,
        help="The data source directory.",
    )
    arg_parser.add_argument(
        "--split",
        "-s",
        dest="split_filename",
        required=True,
        help="The split to reconstruct.",
    )
    arg_parser.add_argument(
        "--iters",
        dest="iterations",
        default=800,
        help="The number of iterations of latent code optimization to perform.",
    )
    arg_parser.add_argument(
        "--skip",
        dest="skip",
        action="store_true",
        help="Skip meshes which have already been reconstructed.",
    )
    lipschitz_regularization.deepsdf.deep_sdf.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    lipschitz_regularization.deepsdf.deep_sdf.configure_logging(args)

    def empirical_stat(latent_vecs, indices):
        lat_mat = torch.zeros(0).cuda()
        for ind in indices:
            lat_mat = torch.cat([lat_mat, latent_vecs[ind]], 0)
        mean = torch.mean(lat_mat, 0)
        var = torch.var(lat_mat, 0)
        return mean, var

    specs_filename = os.path.join(args.experiment_directory, "specs.json")

    if not os.path.isfile(specs_filename):
        raise Exception(
            'The experiment directory does not include specifications file "specs.json"'
        )

    specs = json.load(open(specs_filename))

    arch = __import__("lipschitz_regularization.deepsdf.networks." + specs["NetworkArch"], fromlist=["Decoder"])

    latent_size = specs["CodeLength"]

    decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"])

    decoder = torch.nn.DataParallel(decoder)

    saved_model_state = torch.load(
        os.path.join(
            args.experiment_directory, ws.model_params_subdir, args.checkpoint + ".pth"
        )
    )
    saved_model_epoch = saved_model_state["epoch"]

    decoder.load_state_dict(saved_model_state["model_state_dict"])

    decoder = decoder.module.cuda()

    with open(args.split_filename, "r") as f:
        split = json.load(f)

    npz_filenames = lipschitz_regularization.deepsdf.deep_sdf.data.get_instance_filenames(args.data_source, split)

    random_items = random.sample(npz_filenames, 2)

    #path_sdf_A = Path(random_items[0])
    path_sdf_A = Path("C:\\Users\\loren\\Desktop\\RepoBitbucket\\lipschitz-regularization\\lipschitz_regularization\\deepsdf\\data\\faust\\train\\57d97bcb79d0b70e6b5f74c67958b51e\\57d97bcb79d0b70e6b5f74c67958b51e.npz")
    path_folder_A = path_sdf_A.parent
    latent_code_B = torch.load(path_folder_A / (path_sdf_A.stem + '.pth'))[0]
    obj_file_A = trimesh.load(path_folder_A / (path_sdf_A.stem + '.obj'))
    vertices_A = obj_file_A.vertices


    #path_sdf_B = Path(random_items[1])
    path_sdf_B = Path("C:\\Users\\loren\\Desktop\\RepoBitbucket\\lipschitz-regularization\\lipschitz_regularization\\deepsdf\\data\\faust\\train\\d6c2ddfb5131c68dfd4bf7321cd7179b\d6c2ddfb5131c68dfd4bf7321cd7179b.npz")
    path_folder_B = path_sdf_B.parent
    latent_code_A = torch.load(path_folder_B / (path_sdf_B.stem + '.pth'))[0]
    obj_file_B = trimesh.load(path_folder_A / (path_sdf_A.stem + '.obj'))
    vertices_B = obj_file_B.vertices

    interpolation_percentages = [0.33, 0.66]
    metrics = {}
    metrics[obj_file_B.name]["gt_area"] = obj_file_B.area
    metrics[obj_file_A.name]["gt_area"] = obj_file_A.area
    metrics[obj_file_B.name]["gt_volume"] = obj_file_B.volume
    metrics[obj_file_A.name]["gt_volume"] = obj_file_A.volume


    for t_value in interpolation_percentages:

        interpolated_latent_code = (1 - t_value) * latent_code_A + t_value * latent_code_B
        start = time.time()
        with torch.no_grad():
            filename = lipschitz_regularization.deepsdf.deep_sdf.mesh.create_mesh(
                decoder, interpolated_latent_code, str(path_folder_A / (path_sdf_A.stem + "_interpolation" + t)), N=256, max_batch=int(2 ** 18)
                    )
            logging.debug("total time: {}".format(time.time() - start))
        mesh = trimesh.load(filename)
        metrics[filename]["area"] = mesh.area
        metrics[filename]["volume"] = mesh.volume

        interpolated_vertices = (1 - t_value) * vertices_A + t_value * vertices_B

        start = time.time()
        with torch.no_grad():
            filename = lipschitz_regularization.deepsdf.deep_sdf.mesh.create_mesh(
                decoder, interpolated_vertices, str(path_folder_A / (path_sdf_A.stem + "_interpolation" + t_value)), N=256, max_batch=int(2 ** 18)
                    )
            logging.debug("total time: {}".format(time.time() - start))
        mesh = trimesh.load(filename)
        metrics[filename]["area"] = mesh.area
        metrics[filename]["volume"] = mesh.volume

    with open(Path(args.experiment) / "metrics.json", "w") as json_file:
        json.dump(metrics, json_file)




