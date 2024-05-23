import os
import os.path as osp
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import List
from nnunet_mednext.paths import nnUNet_raw_data

from batchgenerators.utilities.file_and_folder_operations import nifti_files, join, maybe_mkdir_p, save_json
import numpy as np


def make_out_dirs(task_id: int, task_name="Synapse_Abdomen"):
    dataset_name = f"Task{task_id:03d}_{task_name}"

    out_dir = Path(nnUNet_raw_data.replace('"', "")) / dataset_name
    out_train_dir = out_dir / "imagesTr"
    out_labels_train_dir = out_dir / "labelsTr"
    out_val_dir = out_dir / "imagesVal"
    out_labels_val_dir = out_dir / "labelsVal"
    out_test_dir = out_dir / "imagesTs"

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_train_dir, exist_ok=True)
    os.makedirs(out_labels_train_dir, exist_ok=True)
    os.makedirs(out_val_dir, exist_ok=True)
    os.makedirs(out_labels_val_dir, exist_ok=True)
    os.makedirs(out_test_dir, exist_ok=True)

    return out_dir, out_train_dir, out_labels_train_dir, out_val_dir, out_labels_val_dir, out_test_dir


def copy_files(src_data_folder: Path,
               train_dir: Path, labels_train_dir: Path,
               val_dir: Path, labels_val_dir: Path,
               test_dir: Path):
    """Copy files from the Synapse_Abdomen dataset to the nnUNet dataset folder."""
    images_train = sorted([f for f in (src_data_folder / 'Training' / 'img').iterdir()])
    labels_train = sorted([f for f in (src_data_folder / 'Training' / 'label').iterdir()])
    images_val = sorted([f for f in (src_data_folder / 'Validating' / 'img').iterdir()])
    labels_val = sorted([f for f in (src_data_folder / 'Validating' / 'label').iterdir()])
    images_test = sorted([f for f in (src_data_folder / 'Testing' / 'img').iterdir()])

    dst_images_train = []
    dst_labels_train = []
    dst_images_val = []
    dst_labels_val = []
    dst_images_test = []

    # Copy training files and corresponding labels.
    num_training_cases = 0
    for path in images_train:
        stem = path.stem.split('.')[0]
        name, num = stem[:3], stem[3:]
        dst = train_dir / f"{name}_{num}_0000.nii.gz"
        dst_images_train.append(train_dir / f"{name}_{num}.nii.gz")
        shutil.copy(path, dst)
        num_training_cases += 1

    for path in labels_train:
        stem = path.stem.split('.')[0]
        name, num = stem[:5], stem[5:]
        dst = labels_train_dir / f"{name}_{num}.nii.gz"
        dst_labels_train.append(dst)
        shutil.copy(path, dst)

    for path in images_val:
        stem = path.stem.split('.')[0]
        name, num = stem[:3], stem[3:]
        dst = val_dir / f"{name}_{num}_0000.nii.gz"
        dst_images_val.append(test_dir / f"{name}_{num}.nii.gz")
        shutil.copy(path, dst)

    for path in labels_val:
        stem = path.stem.split('.')[0]
        name, num = stem[:5], stem[5:]
        dst = labels_val_dir / f"{name}_{num}.nii.gz"
        dst_labels_val.append(dst)
        shutil.copy(path, dst)

    for path in images_test:
        stem = path.stem.split('.')[0]
        name, num = stem[:3], stem[3:]
        dst = test_dir / f"{name}_{num}_0000.nii.gz"
        dst_images_test.append(test_dir / f"{name}_{num}.nii.gz")
        shutil.copy(path, dst)

    return num_training_cases, dst_images_train, dst_labels_train, dst_images_val, dst_labels_val, dst_images_test


def convert_synapse_abdomen(src_data_folder: str, task_id: int):
    out_dir, train_dir, labels_train_dir, val_dir, labels_val_dir, test_dir = make_out_dirs(task_id)
    num_training_cases, images_train, labels_train, images_val, labels_val, images_test = (
        copy_files(Path(src_data_folder), train_dir, labels_train_dir, val_dir, labels_val_dir, test_dir))

    json_dict = OrderedDict()
    json_dict['modality'] = {
        0: "CT",
    }
    # labels differ for ACDC challenge
    json_dict['labels'] = {
        0: "background",
        1: "spleen",
        2: "right kidney",
        3: "left kidney",
        4: "gallbladder",
        5: "esophagus",
        6: "liver",
        7: "stomach",
        8: "aorta",
        9: "inferior vena cava",
        10: "portal vein and splenic vein",
        11: "pancreas",
        12: "right adrenal gland",
        13: "left adrenal gland",
    }
    json_dict['numTraining'] = num_training_cases
    json_dict['numVal'] = len(images_val)
    json_dict['numTest'] = len(images_test)
    json_dict['training'] = [{"image": str(img.relative_to(out_dir)), "label": str(label.relative_to(out_dir))} for img, label in
                          zip(images_train, labels_train)]
    json_dict['val'] = [{"image": str(img.relative_to(out_dir)), "label": str(label.relative_to(out_dir))} for img, label in
                        zip(images_val, labels_val)]
    json_dict['test'] = [{"image": str(img.relative_to(out_dir))} for img in images_test]

    save_json(json_dict, os.path.join(out_dir, "dataset.json"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_folder",
        type=str,
        help="The processed Synapse_Abdomen dataset dir.",
    )
    parser.add_argument(
        "-d", "--task_id", required=False, type=int, default=200, help="nnU-Net Dataset ID, default: 200"
    )
    args = parser.parse_args()
    print("Converting...")
    convert_synapse_abdomen(args.input_folder, args.task_id)
    print("Done!")
