import argparse
from vedo import *
from pathlib import Path
from vedo import Plotter, Mesh, settings

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
    parser.add_argument(
        "--save_screenshot",
        "-s",
        type=str,
        required=True,
        help="Path to save the screenshot of the meshes.",
    )
    args = parser.parse_args()
    settings.renderer_frame_alpha = 0
    meshes_path = list(Path(args.folder_lipschitzmlp).glob("*.ply"))
    i = 0
    plt = Plotter(shape=(2, 11), offscreen=0, sharecam=True).add_shadows()
    index = 0
    for mesh_path in meshes_path:
        if "tensor" in str(mesh_path):
            if i % 10 == 0:
                m = Mesh(str(mesh_path)).c("blue8")
                plt.at(index).show(m)
                print(mesh_path.name)
                index += 1
            i += 1
            if i == 100:
                m = Mesh(str(mesh_path)).c("blue8")
                plt.at(index).show(m)
                print(mesh_path.name)
    meshes_path = list(Path(args.folder_standardmlp).glob("*.ply"))
    index += 1
    i = 0
    for mesh_path in meshes_path:
        if "tensor" in str(mesh_path):
            if i % 10 == 0:
                m = Mesh(str(mesh_path)).c("green")
                plt.at(index).show(m)
                print(mesh_path.name)
                index += 1
            i += 1
            if i == 100:
                m = Mesh(str(mesh_path)).c("green")
                plt.at(index).show(m)
                print(mesh_path.name)
    plt.reset_camera(tight=0.05)
    plt.screenshot(args.save_screenshot)
    plt.show(interactive=True).close()
