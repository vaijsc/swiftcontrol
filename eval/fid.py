# python eval_fid.py ../data/eval/generate/controlnet_sd15 --ref-dir /path/to/reference/directory --outpath ./eval/fid/sb_gen.txt --crop True --eval-size 256 --batch-size 1024

import sys
from functools import partial
from pathlib import Path

import numpy as np
import torchvision.transforms as transforms
import typer
from PIL import Image

sys.path.append(Path(__file__).parent.parent.as_posix())
from cleanfid import fid


class CenterCropLongEdge(object):
    """
    this code is borrowed from https://github.com/ajbrock/BigGAN-PyTorch
    MIT License
    Copyright (c) 2019 Andy Brock
    """

    def __call__(self, img):
        return transforms.functional.center_crop(img, min(img.size))

    def __repr__(self):
        return self.__class__.__name__


center_crop_trsf = CenterCropLongEdge()


def resize_and_center_crop(image_np, resize_size):
    image_pil = Image.fromarray(image_np)
    image_pil = center_crop_trsf(image_pil)

    if resize_size is not None:
        image_pil = image_pil.resize(
            (resize_size, resize_size), Image.Resampling.LANCZOS
        )
    return np.array(image_pil)


app = typer.Typer()
#../data/eval/generate/controlnet_sd15  ../data/eval/generate/controlnet_sd21  ../data/eval/generate/repcontrolnet_sd21

@app.command()
def main(
    eval_dir: Path = typer.Argument(..., help="evaldir"),
    ref_dir: Path = typer.Option(
        "/lustre/scratch/client/vinai/users/ngannh9/RepControlNet/data/eval/MSCOCO/image",
        help="evaldir",
    ),
    outpath: Path = typer.Option("./eval/fid/sb_gen.txt", help="evaldir"),
    crop: bool = typer.Option(True, help="crop image"),
    eval_size: int = typer.Option(256, help="size to eval, use if crop==true"),
    batch_size: int = typer.Option(1024, help="Batch size"),
):
    eval_dir = eval_dir.resolve()
    ref_dir = ref_dir.resolve()
    kwargs = dict(
        fdir1=ref_dir.as_posix(),
        fdir2=eval_dir.as_posix(),
        batch_size=batch_size,
        model_name="inception_v3",
        # fdir2_count=fdir2_nsamples,
    )
    if crop:
        custom_image_tranform = partial(resize_and_center_crop, resize_size=eval_size)
        kwargs["custom_image_tranform"] = custom_image_tranform
    print(f"Kwargs: {kwargs}")
    score = fid.compute_fid(**kwargs)
    print(f"FID={score}")
    outpath.parent.mkdir(exist_ok=True, parents=True)
    with open(outpath.as_posix(), "a") as f:
        ref_name = ref_dir.parent.parent.name + "_" + ref_dir.parent.name + "_" + ref_dir.name
        eval_name = eval_dir.parent.parent.name + "_" + eval_dir.parent.name + "_" + eval_dir.name
        f.write(f"{ref_name}\t{eval_name}\t{score}\n")


if __name__ == "__main__":
    app()