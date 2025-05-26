import re
from pathlib import Path
import numpy as np
import pandas as pd
from fire import Fire
from natsort import natsorted
from loguru import logger
# from datasets.preprocessing.base_preprocessing import BasePreprocessing
from datasets.preprocessing.scannet_preprocessing import ScannetPreprocessing
from utils.point_cloud_utils import load_ply_with_normals # I had to put it in this folder to make it works

from datasets.scannet200.scannet200_constants import VALID_CLASS_IDS_200, SCANNET_COLOR_MAP_200, CLASS_LABELS_200 #copied scanned 200 inside

from rich.traceback import install
install()

def test_preprocessing_single_scene():
    data_dir = "scans"
    save_dir = "out"
    git_repo_dir = "openbench3d/ScanNet"
    
    mode = "train"
    scene_id = "scene0000_00"
    scene_path = Path(data_dir) / scene_id
    point_cloud_path = scene_path / (scene_id + "_vh_clean_2.ply")
    save_path = Path(save_dir)  / scene_id
    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)
        
    preprocessor = ScannetPreprocessing(data_dir=data_dir, save_dir=save_dir, git_repo=git_repo_dir, scannet200=True)
    filebase = preprocessor.process_file(point_cloud_path, mode)
    print(filebase)
    
if __name__ == "__main__":
    # Fire(ScannetPreprocessing)
    test_preprocessing_single_scene()