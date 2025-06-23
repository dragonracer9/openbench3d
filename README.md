# OpenBench3D: Enhanced OpenMask3D Pipeline

## Overview
This repository builds on top of the original [OpenMask3D](https://github.com/EPFL-VILAB/OpenMask3D) pipeline, providing improved 3D mask inpainting, enhanced debugging/visualization, and a real-time interactive similarity visualization demo. It is designed for research on 3D semantic instance segmentation and open-vocabulary 3D scene understanding.

**For the original OpenMask3D documentation and setup, see [`openmask3d/README_openmask3d.md`](README_openmask3d.md).**

---

## 1. Installation

**Recommended:**
- Follow the official setup instructions in [`setup.md`](setup.md) and [`openmask3d/README_openmask3d.md`](openmask3d/README_openmask3d.md) for all dependencies, dataset preparation, and environment configuration.

**Alternative:**
- We provide a [`requirements.txt`](requirements.txt) and [`environment.yaml`](environment.yaml) reflecting our current working environment. You may try:
  ```bash
  conda env create -f environment.yaml
  # or
  pip install -r requirements.txt
  ```
  *Note: We cannot guarantee these will work on all systems due to complex dependencies (CUDA, Open3D, etc).*

---

## 2. Data Preparation

- Download the ScanNet200 dataset and precomputed features using the provided script:
  ```bash
  bash download_dataset.sh
  ```
- This will populate the `data/` directory with the required scans and metadata.
- **Paths:** You may need to adjust paths in scripts/configs to match your local setup.

---

## 3. Pipeline Usage

### Main Pipeline
- The main pipeline is run via:
  ```bash
  bash run_openmask3d_scannet200_eval.sh
  ```
- This will execute the full feature extraction, mask inpainting, and evaluation pipeline on ScanNet200.
- **Note:** You must adjust paths in the script and configs to match your local directory structure.

### Main Changes to OpenMask3D
- **Data Preprocessing Scripts:** Added scripts to convert data from the OpenScene format to the format required by OpenMask3D. These scripts help with the data preparation process.
- **Occlusion Inpainting:** Mask inpainting using 3D occlusion checks.
- **Debugging/Visualization:** Extensive debug outputs, including mask overlays and inpainted images, are saved for each mask/view pair if so desired.
- **Refactored Code:** See `openmask3d/mask_features_computation/features_extractor.py` and related files for the main logic changes. Due to a change in the nested loop structure in order to save on SAM calls, our implementation runs around 3x faster than the original (with inpainting disabled) on our machine (RTX 2080).
- **Evaluation Script:** Significant adjustments were also made to the evaluation process in `run_eval_close_vocab_inst_seg.py`. This script evaluates closed-vocabulary instance segmentation results. Here you can find our changes regarding synonym/hierarchical evaluation. 

---

## 4. Interactive Visualization Demo

### Queryable Similarity Visualization
- We provide an interactive demo for real-time text-to-3D similarity visualization:
  - **Script:** [`openmask3d/visualization/viz_sim_score_export.py`](openmask3d/visualization/viz_sim_score_export.py)
  - **Requirements:**
    - Python 3.8+
    - `open3d` (with GUI support), `torch`, `matplotlib`, `clip`, `numpy`
  - **Required Files:**
    - Point cloud: `data/scans/<scene>/<scene>_vh_clean_2.ply`
    - Predicted masks: `output/clip/masks/<scene>_masks.pt`
    - Mask features: `output/clip/mask_features_final/<scene>_openmask3d_features.npy`
  - **Usage:**
    ```bash
    python openmask3d/visualization/viz_sim_score_export.py <scene_name>
    ```
    - Enter a text query in the terminal to update the similarity heatmap in real time.
    - Example: `python openmask3d/visualization/viz_sim_score_export.py scene0030_02`

---

## 5. Citing and Acknowledgements

This project is built on top of the excellent [OpenMask3D](https://github.com/EPFL-VILAB/OpenMask3D). If you use this code or data, **please cite their original work**:

> @inproceedings{takmaz2023openmask3d,
  title={{OpenMask3D: Open-Vocabulary 3D Instance Segmentation}},
  author={Takmaz, Ay{\c{c}}a and Fedele, Elisabetta and Sumner, Robert W. and Pollefeys, Marc and Tombari, Federico and Engelmann, Francis},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
year={2023}
}

See [`openmask3d/README_openmask3d.md`](openmask3d/README_openmask3d.md) for the original documentation and more details.