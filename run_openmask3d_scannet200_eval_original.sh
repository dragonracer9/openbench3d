# #!/bin/bash
# export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine
# set -e

# # OPENMASK3D SCANNET200 EVALUATION SCRIPT
# # This script performs the following in order to evaluate OpenMask3D predictions on the ScanNet200 validation set
# # 1. Compute class agnostic masks and save them
# # 2. Compute mask features for each mask and save them
# # 3. Evaluate for closed-set 3D semantic instance segmentation

# # --------
# # NOTE: SET THESE PARAMETERS!
# SCANS_PATH="/home/ninol/data/preprocessed/scans"
# SCANNET_PROCESSED_DIR="$(pwd)/openmask3d/class_agnostic_mask_computation/datasets/processed/scannet"
# # model ckpt paths
# MASK_MODULE_CKPT_PATH="$(pwd)/resources/scannet200_val.ckpt"
# SAM_CKPT_PATH="$(pwd)/resources/sam_vit_h_4b8939.pth"
# # output directories to save masks and mask features
# EXPERIMENT_NAME="scannet200"
# OUTPUT_DIRECTORY="$(pwd)/output"
# TIMESTAMP=$(date +"%Y-%m-%d-%H-%M-%S")
# OUTPUT_FOLDER_DIRECTORY="${OUTPUT_DIRECTORY}/${TIMESTAMP}-${EXPERIMENT_NAME}"
# MASK_SAVE_DIR="${OUTPUT_FOLDER_DIRECTORY}/masks"
# MASK_FEATURE_SAVE_DIR="${OUTPUT_FOLDER_DIRECTORY}/mask_features"
# SAVE_VISUALIZATIONS=false #if set to true, saves pyviz3d visualizations

# # Paremeters below are AUTOMATICALLY set based on the parameters above:
# SCANNET_LABEL_DB_PATH="${SCANNET_PROCESSED_DIR%/}/label_database.yaml"
# SCANNET_INSTANCE_GT_DIR="${SCANNET_PROCESSED_DIR%/}/instance_gt/validation"
# # gpu optimization
# OPTIMIZE_GPU_USAGE=false

# cd openmask3d

# # 1.Compute class agnostic masks and save them
# python class_agnostic_mask_computation/get_masks_scannet200.py \
# general.experiment_name=${EXPERIMENT_NAME} \
# general.project_name="scannet200" \
# general.checkpoint=${MASK_MODULE_CKPT_PATH} \
# general.train_mode=false \
# model.num_queries=150 \
# general.use_dbscan=true \
# general.dbscan_eps=0.95 \
# general.save_visualizations=${SAVE_VISUALIZATIONS} \
# data.test_dataset.data_dir=${SCANNET_PROCESSED_DIR}  \
# data.validation_dataset.data_dir=${SCANNET_PROCESSED_DIR} \
# data.train_dataset.data_dir=${SCANNET_PROCESSED_DIR} \
# data.train_dataset.label_db_filepath=${SCANNET_LABEL_DB_PATH} \
# data.validation_dataset.label_db_filepath=${SCANNET_LABEL_DB_PATH} \
# data.test_dataset.label_db_filepath=${SCANNET_LABEL_DB_PATH}  \
# general.mask_save_dir=${MASK_SAVE_DIR} \
# # data.test_dataloader.batch_size=1 \
# # data.test_dataloader.num_workers=8 \
# hydra.run.dir="${OUTPUT_FOLDER_DIRECTORY}/hydra_outputs/class_agnostic_mask_computation"
# echo "[INFO] Mask computation done!"
# # get the path of the saved masks
# echo "[INFO] Masks saved to ${MASK_SAVE_DIR}."

# # 2. Compute mask features
# echo "[INFO] Computing mask features..."
# python compute_features_scannet200.py \
# data.scans_path=${SCANS_PATH} \
# data.masks.masks_path=${MASK_SAVE_DIR} \
# output.output_directory=${MASK_FEATURE_SAVE_DIR} \
# output.experiment_name=${EXPERIMENT_NAME} \
# external.sam_checkpoint=${SAM_CKPT_PATH} \
# gpu.optimize_gpu_usage=${OPTIMIZE_GPU_USAGE} \
# hydra.run.dir="${OUTPUT_FOLDER_DIRECTORY}/hydra_outputs/mask_features_computation"
# echo "[INFO] Feature computation done!"

# # 3. Evaluate for closed-set 3D semantic instance segmentation
# python evaluation/run_eval_close_vocab_inst_seg.py \
# --gt_dir=${SCANNET_INSTANCE_GT_DIR} \
# --mask_pred_dir=${MASK_SAVE_DIR} \
# --mask_features_dir=${MASK_FEATURE_SAVE_DIR} \

#!/bin/bash
export OMP_NUM_THREADS=3
set -e

# --------
# Parameters (edit these!)
SCANS_PATH="$(pwd)/datasets/data/scans"
SCANNET_PROCESSED_DIR="$(pwd)/openmask3d/class_agnostic_mask_computation/data/processed/scannet"
MASK_MODULE_CKPT_PATH="$(pwd)/resources/scannet200_val.ckpt"
SAM_CKPT_PATH="$(pwd)/resources/sam_vit_h_4b8939.pth"
EXPERIMENT_NAME="scannet200"
OUTPUT_DIRECTORY="$(pwd)/output"
SAVE_VISUALIZATIONS=false
OPTIMIZE_GPU_USAGE=false


# Set resume point (options: "", "step1", "step2")
RESUME_FROM=$1

if [[ "$RESUME_FROM" == "step1" || "$RESUME_FROM" == "step2" ]]; then
    # RESUME FROM PREVIOUS TIMESTAMP
    PREVIOUS_RUN="2025-05-20-01-00-59-scannet200-Copy_newCLIP50masks"
    OUTPUT_FOLDER_DIRECTORY="${OUTPUT_DIRECTORY}/${PREVIOUS_RUN}"
    MASK_SAVE_DIR="${OUTPUT_FOLDER_DIRECTORY}/masks"
    MASK_FEATURE_SAVE_DIR="${OUTPUT_FOLDER_DIRECTORY}/mask_features"
else
    # FRESH RUN
    TIMESTAMP=$(date +"%Y-%m-%d-%H-%M-%S")
    OUTPUT_FOLDER_DIRECTORY="${OUTPUT_DIRECTORY}/${TIMESTAMP}-${EXPERIMENT_NAME}"
    MASK_SAVE_DIR="${OUTPUT_FOLDER_DIRECTORY}/masks"
    MASK_FEATURE_SAVE_DIR="${OUTPUT_FOLDER_DIRECTORY}/mask_features"
fi

SCANNET_LABEL_DB_PATH="${SCANNET_PROCESSED_DIR%/}/label_database.yaml"
SCANNET_INSTANCE_GT_DIR="${SCANNET_PROCESSED_DIR%/}/instance_gt/validation"

cd openmask3d

if [[ "$RESUME_FROM" != "step1" && "$RESUME_FROM" != "step2" ]]; then
    echo "[INFO] Running STEP 1: Mask Computation"
    python class_agnostic_mask_computation/get_masks_scannet200.py \
        general.experiment_name=${EXPERIMENT_NAME} \
        general.project_name="scannet200" \
        general.checkpoint=${MASK_MODULE_CKPT_PATH} \
        general.train_mode=false \
        model.num_queries=150 \
        general.use_dbscan=true \
        general.dbscan_eps=0.95 \
        general.save_visualizations=${SAVE_VISUALIZATIONS} \
        data.test_dataset.data_dir=${SCANNET_PROCESSED_DIR}  \
        data.validation_dataset.data_dir=${SCANNET_PROCESSED_DIR} \
        data.train_dataset.data_dir=${SCANNET_PROCESSED_DIR} \
        data.train_dataset.label_db_filepath=${SCANNET_LABEL_DB_PATH} \
        data.validation_dataset.label_db_filepath=${SCANNET_LABEL_DB_PATH} \
        data.test_dataset.label_db_filepath=${SCANNET_LABEL_DB_PATH}  \
        general.mask_save_dir=${MASK_SAVE_DIR} \
        hydra.run.dir="${OUTPUT_FOLDER_DIRECTORY}/hydra_outputs/class_agnostic_mask_computation"
    echo "[INFO] Step 1 complete: Masks saved to ${MASK_SAVE_DIR}"
else
    echo "[INFO] Skipping STEP 1 (Mask Computation)"
fi

if [[ "$RESUME_FROM" != "step2" ]]; then
    echo "[INFO] Running STEP 2: Feature Computation"
    python compute_features_scannet200.py \
        data.scans_path=${SCANS_PATH} \
        data.masks.masks_path=${MASK_SAVE_DIR} \
        output.output_directory=${MASK_FEATURE_SAVE_DIR} \
        output.experiment_name=${EXPERIMENT_NAME} \
        external.sam_checkpoint=${SAM_CKPT_PATH} \
        gpu.optimize_gpu_usage=${OPTIMIZE_GPU_USAGE} \
        hydra.run.dir="${OUTPUT_FOLDER_DIRECTORY}/hydra_outputs/mask_features_computation"
    echo "[INFO] Step 2 complete: Features saved to ${MASK_FEATURE_SAVE_DIR}"
else
    echo "[INFO] Skipping STEP 2 (Feature Computation)"
fi

echo "[INFO] Running STEP 3: Evaluation"
python evaluation/run_eval_close_vocab_inst_seg.py \
    --gt_dir=${SCANNET_INSTANCE_GT_DIR} \
    --mask_pred_dir=${MASK_SAVE_DIR} \
    --mask_features_dir=${MASK_FEATURE_SAVE_DIR}
echo "[INFO] Evaluation complete"