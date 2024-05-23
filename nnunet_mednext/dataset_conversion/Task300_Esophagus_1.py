import os
import os.path as osp
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import List
from nnunet_mednext.paths import nnUNet_raw_data

from batchgenerators.utilities.file_and_folder_operations import nifti_files, join, maybe_mkdir_p, save_json
import numpy as np


def make_out_dirs(task_id: int, task_name="Esophagus_1"):
    dataset_name = f"Task{task_id:03d}_{task_name}"

    out_dir = Path(nnUNet_raw_data.replace('"', "")) / dataset_name
    out_train_dir = out_dir / "imagesTr"
    out_labels_train_dir = out_dir / "labelsTr"
    out_test_dir = out_dir / "imagesTs"
    out_labels_test_dir = out_dir / "labelsTs"

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_train_dir, exist_ok=True)
    os.makedirs(out_labels_train_dir, exist_ok=True)
    os.makedirs(out_test_dir, exist_ok=True)
    os.makedirs(out_labels_test_dir, exist_ok=True)

    return out_dir, out_train_dir, out_labels_train_dir, out_test_dir, out_labels_test_dir


def create_esophagus_1_split(labelsTr_folder: str, seed: int = 0) -> List[dict[str, List]]:
    nii_files = nifti_files(labelsTr_folder, join=False)
    patients = np.unique([i[:len('patient000')] for i in nii_files])
    rs = np.random.RandomState(seed)
    rs.shuffle(patients)
    splits = []
    for fold in range(5):
        val_patients = patients[fold::5]
        train_patients = [i for i in patients if i not in val_patients]
        val_cases = [i[:-7] for i in nii_files for j in val_patients if i.startswith(j)]
        train_cases = [i[:-7] for i in nii_files for j in train_patients if i.startswith(j)]
        splits.append({'train': train_cases, 'val': val_cases})
    return splits


def copy_files(src_data_folder: Path, train_dir: Path, labels_train_dir: Path, test_dir: Path, labels_test_dir: Path):
    """Copy files from the Esophagus_1 dataset to the nnUNet dataset folder. Returns the number of training cases."""
    images_train = sorted([f for f in (src_data_folder / 'imagesTr').iterdir()] + [f for f in (src_data_folder / 'imagesVal').iterdir()])
    labels_train = sorted([f for f in (src_data_folder / 'labelsTr').iterdir()] + [f for f in (src_data_folder / 'labelsVal').iterdir()])
    images_test = sorted([f for f in (src_data_folder / 'imagesTs').iterdir()])
    labels_test = sorted([f for f in (src_data_folder / 'labelsTs').iterdir()])

    dst_images_train = []
    dst_labels_train = []
    dst_images_test = []
    dst_labels_test = []

    # Copy training files and corresponding labels.
    num_training_cases = 0
    for path in images_train:
        stem = path.stem.split('.')[0]
        sp = stem.split('_', 1)
        num = sp[0]
        name = sp[1].rsplit('_', 1)[0]
        dst = train_dir / f"{name}_{num}_0000.nii.gz"
        dst_images_train.append(train_dir / f"{name}_{num}.nii.gz")
        shutil.copy(path, dst)
        num_training_cases += 1

    for path in labels_train:
        stem = path.stem.split('.')[0]
        sp = stem.split('_', 1)
        num = sp[0]
        name = sp[1].rsplit('_', 1)[0]
        dst = labels_train_dir / f"{name}_{num}.nii.gz"
        dst_labels_train.append(dst)
        shutil.copy(path, dst)

    for path in images_test:
        stem = path.stem.split('.')[0]
        sp = stem.split('_', 1)
        num = sp[0]
        name = sp[1].rsplit('_', 1)[0]
        dst = test_dir / f"{name}_{num}_0000.nii.gz"
        dst_images_test.append(test_dir / f"{name}_{num}.nii.gz")
        shutil.copy(path, dst)

    for path in labels_test:
        stem = path.stem.split('.')[0]
        sp = stem.split('_', 1)
        num = sp[0]
        name = sp[1].rsplit('_', 1)[0]
        dst = labels_test_dir / f"{name}_{num}.nii.gz"
        dst_labels_test.append(dst)
        shutil.copy(path, dst)

    return num_training_cases, dst_images_train, dst_labels_train, dst_images_test, dst_labels_test


def convert_esophagus_1(src_data_folder: str, task_id: int):
    out_dir, train_dir, labels_train_dir, test_dir, labels_test_dir = make_out_dirs(task_id)
    num_training_cases, images_train, labels_train, images_test, labels_test = (
        copy_files(Path(src_data_folder), train_dir, labels_train_dir, test_dir, labels_test_dir))

    json_dict = OrderedDict()
    json_dict['modality'] = {
        0: "CT",
    }
    # labels differ for ACDC challenge
    json_dict['labels'] = {
        0: "background",
        1: "cancer"
    }
    json_dict['numTraining'] = num_training_cases
    json_dict['numTest'] = len(images_test)
    json_dict['training'] = [{"image": str(img.relative_to(out_dir)), "label": str(label.relative_to(out_dir))} for img, label in
                             zip(images_train, labels_train)]
    json_dict['test'] = [{"image": str(img.relative_to(out_dir)), "label": str(label.relative_to(out_dir))} for img, label in
                         zip(images_test, labels_test)]

    save_json(json_dict, os.path.join(out_dir, "dataset.json"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_folder",
        type=str,
        help="The downloaded Esophagus_1 dataset dir. Should contain extracted 'pre-therapy' and 'post-therapy' folders.",
    )
    parser.add_argument(
        "-d", "--task_id", required=False, type=int, default=300, help="nnU-Net Dataset ID, default: 300"
    )
    args = parser.parse_args()
    print("Converting...")
    convert_esophagus_1(args.input_folder, args.task_id)
    print("Done!")
