import warnings
from dataclasses import dataclass, field


@dataclass
class Dataset:
    dataset_name: str
    dataset_type: str = field(default="torch")
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    meta_path: str = field(default=None, metadata={"help": "Path to the meta data for webdataset."})
    image_path: str = field(default=None, metadata={"help": "Path to the training image data."})
    depth_path: str = field(default=None, metadata={"help": "Path to the training depth data."})
    caption_choice: str = field(default=None, metadata={"help": "Path to the caption directory for recaption."})
    description: str = field(
        default=None,
        metadata={
            "help": "Detailed desciption of where the data is from, how it is labelled, intended use case and the size of the dataset."
        },
    )
    test_script: str = (None,)
    maintainer: str = (None,)
    ############## ############## ############## ############## ############## ##############
    caption_choice: str = field(default=None, metadata={"help": "Path to the captions for webdataset."})
    caption_choice_2: str = field(default=None, metadata={"help": "Path to the captions for webdataset."})
    start_idx: float = field(default=-1, metadata={"help": "Start index of the dataset."})
    end_idx: float = field(default=-1, metadata={"help": "Start index of the dataset."})


DATASETS = {}


def add_dataset(dataset):
    if dataset.dataset_name in DATASETS:
        # make sure the data_name is unique
        warnings.warn(f"{dataset.dataset_name} already existed in DATASETS. Make sure the name is unique.")
    assert "+" not in dataset.dataset_name, "Dataset name cannot include symbol '+'."
    DATASETS.update({dataset.dataset_name: dataset})
    print(f"---> Register {dataset.dataset_name} to DATASET dict")


def register_datasets_mixtures():

    llava_1_5_mm_align = Dataset(
        dataset_name="llava_1_5_mm_align",
        dataset_type="torch",
        data_path="/PATH/LLaVA-CC3M-Pretrain-595K/chat.json",
        image_path="/PATH/LLaVA-CC3M-Pretrain-595K/images",
    )
    add_dataset(llava_1_5_mm_align)

    llava_1_5_sft = Dataset(
        dataset_name="llava_1_5_sft",
        dataset_type="torch",
        data_path="/PATH/llava_v1_5_mix665k.json",
        image_path="/PATH/data",
    )
    add_dataset(llava_1_5_sft)

    spatialrgpt_ft = Dataset(
        dataset_name="spatialrgpt_ft",
        dataset_type="spatialrgpt",
        data_path="/PATH/result_10_depth_convs.json",
        image_path="/PATH/Openimages/train",
        depth_path="/PATH/relative_depth/raw",
        description="900K SFT data by SpatialRGPT (submission) w/ depth (template+LLaMa rephrased).",
    )
    add_dataset(spatialrgpt_ft)
    
    # ========== Added dataset for AI City Challenge ==============
    
    PSIW_sft_train = Dataset(
        dataset_name="PSIW_sft_train",
        dataset_type="spatial_warehouse",
        data_path="datasets/PhysicalAI-Spatial-Intelligence-Warehouse/train_data_added_region_label_formatted.jsonl",
        image_path="datasets/PhysicalAI-Spatial-Intelligence-Warehouse/train/images",
        depth_path="datasets/PhysicalAI-Spatial-Intelligence-Warehouse/train/depths",
        description="This is the Dataset of -> training <- data for Warehouse Spatial Intelligence"
    )
    add_dataset(PSIW_sft_train)
    
    PSIW_sft_val = Dataset(
        dataset_name="PSIW_sft_val",
        dataset_type="spatial_warehouse",
        data_path="datasets/PhysicalAI-Spatial-Intelligence-Warehouse/formatted_dataset/val_aicity_srgpt.jsonl",
        image_path="datasets/PhysicalAI-Spatial-Intelligence-Warehouse/val/images",
        depth_path="datasets/PhysicalAI-Spatial-Intelligence-Warehouse/val/depths",
        description="This is the Dataset of -> Validation <- data for Warehouse Spatial Intelligence"
    )
    add_dataset(PSIW_sft_val)
    
    PSIW_sft_test = Dataset(
        dataset_name="PSIW_sft_test",
        dataset_type="spatial_warehouse",
        data_path="datasets/PhysicalAI-Spatial-Intelligence-Warehouse/formatted_dataset/test_aicity_srgpt.jsonl",
        image_path="datasets/PhysicalAI-Spatial-Intelligence-Warehouse/test/images",
        depth_path="datasets/PhysicalAI-Spatial-Intelligence-Warehouse/test/depths",
        description="This is the Dataset of -> Test <- data for Warehouse Spatial Intelligence"
    )
    
    add_dataset(PSIW_sft_test)
