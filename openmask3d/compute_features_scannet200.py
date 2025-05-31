import hydra
from omegaconf import DictConfig
import numpy as np
from openmask3d.data.load import Camera, InstanceMasks3D, Images, PointCloud, get_number_of_images
from openmask3d.utils import get_free_gpu, create_out_folder
from openmask3d.mask_features_computation.features_extractor import FeaturesExtractor
from openmask3d.mask_features_computation.features_extractor_siglip import FeaturesExtractorSiglip
import torch
import os
import time
from glob import glob

# TIP: add version_base=None to the arguments if you encounter some error  
@hydra.main(config_path="configs", config_name="openmask3d_scannet200_eval")
def main(ctx: DictConfig):
    device = "cpu"  # "mps" if torch.backends.mps.is_available() else "cpu"
    device = get_free_gpu(7000) if torch.cuda.is_available() else device
    print(f"[INFO] Using device: {device}")
    out_folder = ctx.output.output_directory
    os.chdir(hydra.utils.get_original_cwd())
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    print(f"[INFO] Saving feature results to {out_folder}")

    # Check already computed features
    existing_features = glob(os.path.join(out_folder, 'scene*_openmask3d_features.npy'))
    # If features already exist, skip computation for those scenes
    existing_scenes = [os.path.basename(f).replace('_openmask3d_features.npy', '').replace('scene', '') for f in existing_features]
    print(f"[INFO] Existing features for scenes: {existing_scenes}")
    
    # Selected subset of masks to evaluate
    # selected_masks = ['0671_01', '0300_01', '0231_00', '0063_00', '0553_01', '0095_01', '0655_00', '0329_02', '0549_00', '0217_00', '0334_02', '0702_00', '0355_01', '0164_03', '0606_00', '0474_05', '0389_00', '0684_01', '0670_01', '0256_00', '0100_01', '0084_02', '0580_00', '0488_00', '0050_02', '0426_00', '0651_02', '0663_00', '0559_01', '0353_00', '0684_00', '0583_01', '0552_01', '0357_01', '0599_01', '0334_00', '0139_00', '0575_01', '0277_02', '0629_02', '0353_02', '0607_00', '0432_00', '0458_01', '0406_02', '0030_02', '0088_01', '0559_00', '0435_03', '0643_00']
    selected_masks = ['0011_00']
    all_masks_paths = sorted(glob(os.path.join(ctx.data.masks.masks_path, ctx.data.masks.masks_suffix)))
    
    # Filter masks_paths to only include selected scenes
    masks_paths = []
    for mask_path in all_masks_paths:
        scene_id = mask_path.split('/')[-1].replace('scene', '').replace('_masks.pt', '')
        if scene_id in existing_scenes:
            print(f"[INFO] Skipping already computed scene {scene_id} from {mask_path}")
            continue
        if scene_id in selected_masks:
            masks_paths.append(mask_path)
    
    print(f"[INFO] Selected masks: {[p.split('/')[-1].replace('scene', '').replace('_masks.pt', '') for p in masks_paths]}")
    print(masks_paths)
    
    for masks_path in masks_paths:
        # Start timing for this scene
        scene_start_time = time.time()

        print(f"[INFO] Processing masks from {masks_path}")

        scene_num_str = masks_path.split('/')[-1][5:12]
        path = os.path.join(ctx.data.scans_path, 'scene'+ scene_num_str)
        poses_path = os.path.join(path,ctx.data.camera.poses_path)
        point_cloud_path = glob(os.path.join(path, '*vh_clean_2.ply'))[0]
        intrinsic_path = os.path.join(path, ctx.data.camera.intrinsic_path)
        images_path = os.path.join(path, ctx.data.images.images_path)
        depths_path = os.path.join(path, ctx.data.depths.depths_path)
        
        # 1. Load the masks
        masks = InstanceMasks3D(masks_path) 

        # 2. Load the images
        indices = np.arange(0, get_number_of_images(poses_path), step = ctx.openmask3d.frequency)

        images = Images(images_path=images_path, 
                        extension=ctx.data.images.images_ext, 
                        indices=indices)

        # 3. Load the pointcloud
        pointcloud = PointCloud(point_cloud_path, ply=True)

        # 4. Load the camera configurations
        camera = Camera(intrinsic_path=intrinsic_path, 
                        intrinsic_resolution=ctx.data.camera.intrinsic_resolution, 
                        poses_path=poses_path, 
                        depths_path=depths_path, 
                        extension_depth=ctx.data.depths.depths_ext, 
                        depth_scale=ctx.data.depths.depth_scale)

        # 5. Run extractor
        if ctx.external.feature_extractor_type == 'clip':
            features_extractor = FeaturesExtractor(camera=camera, 
                                                    clip_model=ctx.external.clip_model, 
                                                    images=images, 
                                                    masks=masks,
                                                    pointcloud=pointcloud, 
                                                    sam_model_type=ctx.external.sam_model_type,
                                                    sam_checkpoint=ctx.external.sam_checkpoint,
                                                    vis_threshold=ctx.openmask3d.vis_threshold,
                                                    device=device)
        elif ctx.external.feature_extractor_type == 'siglip':
            features_extractor = FeaturesExtractorSiglip(camera=camera, 
                                                    siglip_model=ctx.external.siglip_model, 
                                                    images=images, 
                                                    masks=masks,
                                                    pointcloud=pointcloud, 
                                                    sam_model_type=ctx.external.sam_model_type,
                                                    sam_checkpoint=ctx.external.sam_checkpoint,
                                                    vis_threshold=ctx.openmask3d.vis_threshold,
                                                    device=device)
        else:
            raise ValueError(f"Unknown feature extractor type: {ctx.external.feature_extractor_type}. Supported types: 'clip', 'siglip'")
    
        features = features_extractor.extract_features(topk=ctx.openmask3d.top_k, 
                                                        multi_level_expansion_ratio = ctx.openmask3d.multi_level_expansion_ratio,
                                                        num_levels=ctx.openmask3d.num_of_levels, 
                                                        num_random_rounds=ctx.openmask3d.num_random_rounds,
                                                        num_selected_points=ctx.openmask3d.num_selected_points,
                                                        save_crops=ctx.output.save_crops,
                                                        # save_crops=True,
                                                        out_folder=out_folder,
                                                        optimize_gpu_usage=ctx.gpu.optimize_gpu_usage)
        # 6. Save features
        filename = f"scene{scene_num_str}_openmask3d_features.npy"
        output_path = os.path.join(out_folder, filename)
        np.save(output_path, features)
        
        # Calculate and print scene processing time
        scene_end_time = time.time()
        scene_total_time = scene_end_time - scene_start_time
        print(f"[INFO] Mask features for scene {scene_num_str} saved to {output_path}.")
        print(f"[TIMING] Scene {scene_num_str} completed in {scene_total_time:.1f}s ({scene_total_time/60:.1f} minutes)")


        # # 7. Debugging
        # print(f"[INFO] Features shape: {features.shape}")
        # print(f"[INFO] Feature norms (first 10): {[np.linalg.norm(features[i]) for i in range(min(10, len(features)))]}")
        # print(f"[INFO] Feature value ranges (first 3 masks):")
        # for i in range(min(3, len(features))):
        #     feat = features[i]
            # print(f"  Mask {i}: {feat.min():.6f} - {feat.max():.6f}")
    

        # print(f"[INFO] Finished processing scene {scene_num_str}.")
        # break
    
if __name__ == "__main__":
    main()