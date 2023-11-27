from typing import Dict, List, Optional, Union

from dataset_tools.templates import (
    AnnotationType,
    Category,
    CVTask,
    Domain,
    Industry,
    License,
    Research,
)

##################################
# * Before uploading to instance #
##################################
PROJECT_NAME: str = "NTUT 4K Drone Photo"
PROJECT_NAME_FULL: str = "NTUT 4K Drone Photo Dataset for Human Detection"
HIDE_DATASET = False  # set False when 100% sure about repo quality

##################################
# * After uploading to instance ##
##################################
LICENSE: License = License.Unknown()
APPLICATIONS: List[Union[Industry, Domain, Research]] = [Domain.DroneInspection(is_used=False)]
CATEGORY: Category = Category.Drones(extra=Category.Aerial())

CV_TASKS: List[CVTask] = [CVTask.ObjectDetection(), CVTask.Identification()]
ANNOTATION_TYPES: List[AnnotationType] = [AnnotationType.ObjectDetection()]

RELEASE_DATE: Optional[str] = None  # e.g. "YYYY-MM-DD"
if RELEASE_DATE is None:
    RELEASE_YEAR: int = 2022

HOMEPAGE_URL: str = "https://www.kaggle.com/datasets/kuantinglai/ntut-4k-drone-photo-dataset-for-human-detection/data"
# e.g. "https://some.com/dataset/homepage"

PREVIEW_IMAGE_ID: int = 9193535
# This should be filled AFTER uploading images to instance, just ID of any image.

GITHUB_URL: str = "https://github.com/dataset-ninja/ntut-4k-drone-photo"
# URL to GitHub repo on dataset ninja (e.g. "https://github.com/dataset-ninja/some-dataset")

##################################
### * Optional after uploading ###
##################################
DOWNLOAD_ORIGINAL_URL: Optional[
    Union[str, dict]
] = "https://www.kaggle.com/datasets/kuantinglai/ntut-4k-drone-photo-dataset-for-human-detection/download?datasetVersionNumber=1"
# Optional link for downloading original dataset (e.g. "https://some.com/dataset/download")

CLASS2COLOR: Optional[Dict[str, List[str]]] = {
    "stand": [230, 25, 75],
    "soccer": [60, 180, 75],
    "baseball": [255, 225, 25],
    "sit": [0, 130, 200],
    "watchphone": [245, 130, 48],
    "riding": [145, 30, 180],
    "push": [70, 240, 240],
    "walk": [240, 50, 230],
    "block25": [210, 245, 60],
    "block50": [250, 190, 212],
    "block75": [0, 128, 128],
    "recognizable": [220, 190, 255],
}
# If specific colors for classes are needed, fill this dict (e.g. {"class1": [255, 0, 0], "class2": [0, 255, 0]})

# If you have more than the one paper, put the most relatable link as the first element of the list
# Use dict key to specify name for a button
PAPER: Optional[Union[str, List[str], Dict[str, str]]] = None
BLOGPOST: Optional[
    Union[str, List[str], Dict[str, str]]
] = "http://www.aiotlab.org/projects/drone_actions.html"
REPOSITORY: Optional[Union[str, List[str], Dict[str, str]]] = {
    "Competition": "https://www.kaggle.com/competitions/111-1-ntut-dl-app-hw4/overview"
}

CITATION_URL: Optional[str] = None
AUTHORS: Optional[List[str]] = ["Kuan-Ting (K. T.) Lai"]
AUTHORS_CONTACTS: Optional[List[str]] = ["kuantinglai@gmail.com"]

ORGANIZATION_NAME: Optional[Union[str, List[str]]] = ["Taipei Tech AIoT Lab, Taiwan"]
ORGANIZATION_URL: Optional[Union[str, List[str]]] = ["http://www.aiotlab.org/"]

# Set '__PRETEXT__' or '__POSTTEXT__' as a key with string value to add custom text. e.g. SLYTAGSPLIT = {'__POSTTEXT__':'some text}
SLYTAGSPLIT: Optional[Dict[str, Union[List[str], str]]] = {
    "__PRETEXT__": "Additionally, every *recognizable* class has its own ***id*** tag"
}
TAGS: Optional[List[str]] = None


SECTION_EXPLORE_CUSTOM_DATASETS: Optional[List[str]] = None

##################################
###### ? Checks. Do not edit #####
##################################


def check_names():
    fields_before_upload = [PROJECT_NAME]  # PROJECT_NAME_FULL
    if any([field is None for field in fields_before_upload]):
        raise ValueError("Please fill all fields in settings.py before uploading to instance.")


def get_settings():
    if RELEASE_DATE is not None:
        global RELEASE_YEAR
        RELEASE_YEAR = int(RELEASE_DATE.split("-")[0])

    settings = {
        "project_name": PROJECT_NAME,
        "project_name_full": PROJECT_NAME_FULL or PROJECT_NAME,
        "hide_dataset": HIDE_DATASET,
        "license": LICENSE,
        "applications": APPLICATIONS,
        "category": CATEGORY,
        "cv_tasks": CV_TASKS,
        "annotation_types": ANNOTATION_TYPES,
        "release_year": RELEASE_YEAR,
        "homepage_url": HOMEPAGE_URL,
        "preview_image_id": PREVIEW_IMAGE_ID,
        "github_url": GITHUB_URL,
    }

    if any([field is None for field in settings.values()]):
        raise ValueError("Please fill all fields in settings.py after uploading to instance.")

    settings["release_date"] = RELEASE_DATE
    settings["download_original_url"] = DOWNLOAD_ORIGINAL_URL
    settings["class2color"] = CLASS2COLOR
    settings["paper"] = PAPER
    settings["blog"] = BLOGPOST
    settings["repository"] = REPOSITORY
    settings["citation_url"] = CITATION_URL
    settings["authors"] = AUTHORS
    settings["authors_contacts"] = AUTHORS_CONTACTS
    settings["organization_name"] = ORGANIZATION_NAME
    settings["organization_url"] = ORGANIZATION_URL
    settings["slytagsplit"] = SLYTAGSPLIT
    settings["tags"] = TAGS

    settings["explore_datasets"] = SECTION_EXPLORE_CUSTOM_DATASETS

    return settings
