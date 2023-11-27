# https://www.kaggle.com/datasets/kuantinglai/ntut-4k-drone-photo-dataset-for-human-detection

import os, csv, ast
from collections import defaultdict
import supervisely as sly
from supervisely.io.fs import get_file_name_with_ext, get_file_name, get_file_ext
from dotenv import load_dotenv

import supervisely as sly
import os
from dataset_tools.convert import unpack_if_archive
import src.settings as s
from urllib.parse import unquote, urlparse
from supervisely.io.fs import get_file_name, get_file_size
import shutil

from tqdm import tqdm


def download_dataset(teamfiles_dir: str) -> str:
    """Use it for large datasets to convert them on the instance"""
    api = sly.Api.from_env()
    team_id = sly.env.team_id()
    storage_dir = sly.app.get_data_dir()

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, str):
        parsed_url = urlparse(s.DOWNLOAD_ORIGINAL_URL)
        file_name_with_ext = os.path.basename(parsed_url.path)
        file_name_with_ext = unquote(file_name_with_ext)

        sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
        local_path = os.path.join(storage_dir, file_name_with_ext)
        teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

        fsize = api.file.get_directory_size(team_id, teamfiles_dir)
        with tqdm(
            desc=f"Downloading '{file_name_with_ext}' to buffer...",
            total=fsize,
            unit="B",
            unit_scale=True,
        ) as pbar:
            api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)
        dataset_path = unpack_if_archive(local_path)

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, dict):
        for file_name_with_ext, url in s.DOWNLOAD_ORIGINAL_URL.items():
            local_path = os.path.join(storage_dir, file_name_with_ext)
            teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

            if not os.path.exists(get_file_name(local_path)):
                fsize = api.file.get_directory_size(team_id, teamfiles_dir)
                with tqdm(
                    desc=f"Downloading '{file_name_with_ext}' to buffer...",
                    total=fsize,
                    unit="B",
                    unit_scale=True,
                ) as pbar:
                    api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)

                sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
                unpack_if_archive(local_path)
            else:
                sly.logger.info(
                    f"Archive '{file_name_with_ext}' was already unpacked to '{os.path.join(storage_dir, get_file_name(file_name_with_ext))}'. Skipping..."
                )

        dataset_path = storage_dir
    return dataset_path


def count_files(path, extension):
    count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                count += 1
    return count


def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    # project_name = "NTUT 4K Drone Photo"
    train_path = "/home/grokhi/rawdata/ntut-4k-drone-photo/ntut_drone_train/ntut_drone_train"
    test_path = "/home/grokhi/rawdata/ntut-4k-drone-photo/ntut_drone_test/ntut_drone_test"
    vott_folder = "vott-csv-export"
    images_ext = ".jpg"
    anns_ext = "-export.csv"
    batch_size = 30

    ds_name_to_data = {"train": train_path, "test": test_path}

    def create_ann(image_path):
        labels = []

        # image_np = sly.imaging.image.read(image_path)[:, :, 0]
        img_height = 2160  # image_np.shape[0]
        img_wight = 3840  # image_np.shape[1]

        sequence_tag = sly.Tag(sequence_meta, sequence)

        ann_data = name_to_data[get_file_name(image_path)]
        if ann_data == []:
            ann_data = name_to_data[get_file_name_with_ext(image_path)]

        for curr_ann_data in ann_data:
            tags = []
            # tag_meta = meta.get_tag_meta(curr_ann_data[0])
            # if tag_meta is not None:
            #     tag = sly.Tag(tag_meta, value=curr_ann_data[0])
            #     tags.append(tag)

            if "id_" in curr_ann_data[0]:
                val = curr_ann_data[0].split("_")[1]
                tag = sly.Tag(id_meta, value=val)
                tags.append(tag)
                obj_class = [obj_class for obj_class in obj_classes if obj_class.name == "person"][
                    0
                ]

            else:
                obj_class = [
                    obj_class for obj_class in obj_classes if obj_class.name == curr_ann_data[0]
                ][0]

            left = int(curr_ann_data[1][0])
            top = int(curr_ann_data[1][1])
            right = int(curr_ann_data[1][2])
            bottom = int(curr_ann_data[1][3])
            rect = sly.Rectangle(left=left, top=top, right=right, bottom=bottom)
            label = sly.Label(rect, obj_class, tags=tags)
            labels.append(label)

        return sly.Annotation(
            img_size=(img_height, img_wight), labels=labels, img_tags=[sequence_tag]
        )

    # obj_class = sly.ObjClass("human", sly.Rectangle)

    class_names = [
        "stand",
        "soccer",
        "baseball",
        "sit",
        "watchphone",
        "riding",
        "push",
        "walk",
        "block25",
        "block50",
        "block75",
        "person",
    ]

    obj_classes = [sly.ObjClass(name, sly.Rectangle) for name in class_names]

    sequence_meta = sly.TagMeta("sequence", sly.TagValueType.ANY_STRING)
    id_meta = sly.TagMeta("id", sly.TagValueType.ANY_STRING)

    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)
    meta = sly.ProjectMeta(
        obj_classes=obj_classes,
        tag_metas=[sequence_meta, id_meta],
    )
    api.project.update_meta(project.id, meta.to_json())

    for ds_name, ds_data in ds_name_to_data.items():
        dataset = api.dataset.create(project.id, ds_name, change_name_if_conflict=True)

        for sequence in os.listdir(ds_data):
            curr_data_path = os.path.join(ds_data, sequence, vott_folder)

            ann_path = os.path.join(curr_data_path, sequence + anns_ext)

            name_to_data = defaultdict(list)

            with open(ann_path, "r") as file:
                csvreader = csv.reader(file)
                for idx, row in enumerate(csvreader):
                    if idx == 0:
                        continue
                    name_to_data[row[0]].append([row[-1], list(map(float, row[1:-1]))])

            images_names = [
                im_name
                for im_name in os.listdir(curr_data_path)
                if get_file_ext(im_name) == images_ext
            ]

            progress = sly.Progress("Create dataset {}".format(ds_name), len(images_names))

            for images_names_batch in sly.batched(images_names, batch_size=batch_size):
                img_pathes_batch = [
                    os.path.join(curr_data_path, image_name) for image_name in images_names_batch
                ]

                img_infos = api.image.upload_paths(dataset.id, images_names_batch, img_pathes_batch)
                img_ids = [im_info.id for im_info in img_infos]

                anns = [create_ann(image_path) for image_path in img_pathes_batch]
                api.annotation.upload_anns(img_ids, anns)

                progress.iters_done_report(len(images_names_batch))
    return project
