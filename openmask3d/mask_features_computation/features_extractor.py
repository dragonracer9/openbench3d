
import clip
# import lama_cleaner.model
# import lama_cleaner.model.lama
import iopaint.model
import iopaint.model_manager
import iopaint.schema
from iopaint.schema import InpaintRequest
import numpy as np
import imageio
import torch
from tqdm import tqdm
import os
from openmask3d.data.load import Camera, InstanceMasks3D, Images, PointCloud, get_number_of_images
from openmask3d.mask_features_computation.utils import initialize_sam_model, mask2box_multi_level, run_sam
# import lama_cleaner
# from lama_cleaner import LaMa
import iopaint
from simple_lama_inpainting import SimpleLama

class PointProjector:
    def __init__(self, camera: Camera, 
                 point_cloud: PointCloud, 
                 masks: InstanceMasks3D, 
                 vis_threshold, 
                 indices):
        self.vis_threshold = vis_threshold
        self.indices = indices
        self.camera = camera
        self.point_cloud = point_cloud
        self.masks = masks
        self.visible_points_in_view_in_mask, self.visible_points_view, self.projected_points, self.resolution = self.get_visible_points_in_view_in_mask()
        
        
    def get_visible_points_view(self):
        # Initialization
        vis_threshold = self.vis_threshold
        indices = self.indices
        depth_scale = self.camera.depth_scale
        poses = self.camera.load_poses(indices)
        X = self.point_cloud.get_homogeneous_coordinates()
        n_points = self.point_cloud.num_points
        depths_path = self.camera.depths_path        
        resolution = imageio.imread(os.path.join(depths_path, '0.png')).shape
        height = resolution[0]
        width = resolution[1]
        intrinsic = self.camera.get_adapted_intrinsic(resolution)
        
        projected_points = np.zeros((len(indices), n_points, 2), dtype = int)
        visible_points_view = np.zeros((len(indices), n_points), dtype = bool)
        print(f"[INFO] Computing the visible points in each view.")
        
        for i, idx in tqdm(enumerate(indices)): # for each view
            # *******************************************************************************************************************
            # STEP 1: get the projected points
            # Get the coordinates of the projected points in the i-th view (i.e. the view with index idx)
            projected_points_not_norm = (intrinsic @ poses[i] @ X.T).T
            # Get the mask of the points which have a non-null third coordinate to avoid division by zero
            mask = (projected_points_not_norm[:, 2] != 0) # don't do the division for point with the third coord equal to zero
            # Get non homogeneous coordinates of valid points (2D in the image)
            projected_points[i][mask] = np.column_stack([[projected_points_not_norm[:, 0][mask]/projected_points_not_norm[:, 2][mask], 
                    projected_points_not_norm[:, 1][mask]/projected_points_not_norm[:, 2][mask]]]).T
            
            # *******************************************************************************************************************
            # STEP 2: occlusions computation
            # Load the depth from the sensor
            depth_path = os.path.join(depths_path, str(idx) + '.png')
            sensor_depth = imageio.imread(depth_path) / depth_scale
            inside_mask = (projected_points[i,:,0] >= 0) * (projected_points[i,:,1] >= 0) \
                                * (projected_points[i,:,0] < width) \
                                * (projected_points[i,:,1] < height)
            pi = projected_points[i].T
            # Depth of the points of the pointcloud, projected in the i-th view, computed using the projection matrices
            point_depth = projected_points_not_norm[:,2]
            # Compute the visibility mask, true for all the points which are visible from the i-th view
            visibility_mask = (np.abs(sensor_depth[pi[1][inside_mask], pi[0][inside_mask]]
                                        - point_depth[inside_mask]) <= \
                                        vis_threshold).astype(bool)
            inside_mask[inside_mask == True] = visibility_mask
            visible_points_view[i] = inside_mask
        return visible_points_view, projected_points, resolution
    
    def get_bbox(self, mask, view):
        if(self.visible_points_in_view_in_mask[view][mask].sum()!=0):
            true_values = np.where(self.visible_points_in_view_in_mask[view, mask])
            valid = True
            t, b, l, r = true_values[0].min(), true_values[0].max()+1, true_values[1].min(), true_values[1].max()+1 
        else:
            valid = False
            t, b, l, r = (0,0,0,0)
        return valid, (t, b, l, r)
    
    def get_visible_points_in_view_in_mask(self):
        masks = self.masks
        num_view = len(self.indices)
        visible_points_view, projected_points, resolution = self.get_visible_points_view()
        visible_points_in_view_in_mask = np.zeros((num_view, masks.num_masks, resolution[0], resolution[1]), dtype=bool)
        print(f"[INFO] Computing the visible points in each view in each mask.")
        for i in tqdm(range(num_view)):
            for j in range(masks.num_masks):
                visible_masks_points = (masks.masks[:,j] * visible_points_view[i]) > 0
                proj_points = projected_points[i][visible_masks_points]
                if(len(proj_points) != 0):
                    visible_points_in_view_in_mask[i][j][proj_points[:,1], proj_points[:,0]] = True
        self.visible_points_in_view_in_mask = visible_points_in_view_in_mask
        self.visible_points_view = visible_points_view
        self.projected_points = projected_points
        self.resolution = resolution
        return visible_points_in_view_in_mask, visible_points_view, projected_points, resolution
    
    def get_top_k_indices_per_mask(self, k):
        num_points_in_view_in_mask = self.visible_points_in_view_in_mask.sum(axis=2).sum(axis=2)
        topk_indices_per_mask = np.argsort(-num_points_in_view_in_mask, axis=0)[:k,:].T
        return topk_indices_per_mask
    
    def debug_occlusion_for_mask_in_view(self, target_mask_idx, view_idx_in_projector_list, images_obj, num_points_to_debug=5):
        import imageio # For loading depth
        import os
        import matplotlib.pyplot as plt
        import numpy as np # Ensure numpy is imported

        print(f"\n--- Debugging Occlusion for Mask {target_mask_idx} in View (Projector Index {view_idx_in_projector_list}) ---")

        # Get the original image file index and pose for this view
        original_file_idx = self.indices[view_idx_in_projector_list]
        
        # Ensure poses are loaded and accessible. Assuming self.camera.poses is populated correctly by Camera class
        # and corresponds to self.indices used by PointProjector.
        # if self.camera.poses is None or len(self.camera.poses) <= view_idx_in_projector_list:
        #     print(f"ERROR: Poses not available or insufficient for view_idx_in_projector_list {view_idx_in_projector_list}")
        #     # Attempt to load poses for the specific indices if not already done broadly
        #     # This might be redundant if Camera class loads them all based on self.indices
        #     try:
        #         self.camera.load_poses(self.indices) # Ensure poses for all relevant views are loaded
        #         if len(self.camera.poses) <= view_idx_in_projector_list: # Check again
        #              print("ERROR: Still poses not available after attempting load.")
        #              return
        #     except Exception as e:
        #         print(f"ERROR: Failed to load poses: {e}")
        #         return
        
        # pose = self.camera.poses[view_idx_in_projector_list]

        # Get 3D points for the target mask
        # self.masks.masks should be (num_total_points, num_masks)
        mask_3d_point_indices_in_cloud = np.where(self.masks.masks[:, target_mask_idx] > 0)[0]

        if len(mask_3d_point_indices_in_cloud) == 0:
            print(f"Mask {target_mask_idx} has no 3D points. Cannot debug occlusion.")
            return

        # Select a few points from the mask to debug
        points_to_debug_indices_in_cloud = mask_3d_point_indices_in_cloud[:num_points_to_debug]
        selected_3d_points = self.point_cloud.points[points_to_debug_indices_in_cloud]
        # self.point_cloud.X should be (num_total_points, 4)
        selected_3d_points_homogeneous = self.point_cloud.X[points_to_debug_indices_in_cloud, :]

        print(f"Debugging {len(selected_3d_points)} points from mask {target_mask_idx}.")
        print(f"Point indices in cloud: {points_to_debug_indices_in_cloud}")


        # Load sensor depth for this view
        depth_path = os.path.join(self.camera.depths_path, str(original_file_idx) + self.camera.extension_depth) # Use camera's depth extension
        if not os.path.exists(depth_path):
            print(f"ERROR: Depth image not found at {depth_path}")
            return
        
        try:
            sensor_depth_map_raw = imageio.imread(depth_path)
        except Exception as e:
            print(f"ERROR: Could not read depth image {depth_path}: {e}")
            return
            
        sensor_depth_map = sensor_depth_map_raw / self.camera.depth_scale
        height, width = sensor_depth_map.shape
        intrinsic_matrix = self.camera.get_adapted_intrinsic((height, width))

        # Load the corresponding color image for visualization
        color_image_pil = None
        if images_obj and view_idx_in_projector_list < len(images_obj.images):
            color_image_pil = images_obj.images[view_idx_in_projector_list]
            plt.figure(figsize=(10, 8))
            plt.imshow(color_image_pil)
            plt.title(f"Debug View: Mask {target_mask_idx}, Proj. Idx {view_idx_in_projector_list} (Orig File Idx {original_file_idx})")
            projected_debug_points_x = []
            projected_debug_points_y = []
            colors_for_debug_points = []
            labels_for_debug_points = []
        else:
            print("Color image not available for this view index or Images object not provided.")


        for i, point_idx_in_cloud in enumerate(points_to_debug_indices_in_cloud):
            point_3d = selected_3d_points[i]
            point_3d_homogeneous = selected_3d_points_homogeneous[i]

            print(f"\n  Point {i+1} (Cloud Index: {point_idx_in_cloud}): 3D Coords {point_3d}")

            # Project point
            projected_homogeneous = (intrinsic_matrix @ pose @ point_3d_homogeneous.T).T # Should be (4,) then (3,)
            
            calculated_depth_from_projection = projected_homogeneous[2]
            print(f"    Projected Homogeneous (after Intrinsic@Pose@X.T): {projected_homogeneous}")
            print(f"    Calculated Depth from Projection (Z_cam): {calculated_depth_from_projection:.4f}")

            if calculated_depth_from_projection == 0:
                print("    Point projects to camera center (depth 0) or behind. Skipping further checks for this point.")
                if color_image_pil:
                    labels_for_debug_points.append(f"P{i+1}\nDepth=0")
                    # Add a placeholder if you want to plot it, e.g., at (0,0)
                    projected_debug_points_x.append(0) 
                    projected_debug_points_y.append(0)
                    colors_for_debug_points.append('blue')
                continue

            u_coord = projected_homogeneous[0] / calculated_depth_from_projection
            v_coord = projected_homogeneous[1] / calculated_depth_from_projection
            print(f"    Projected 2D Coords (u,v) in pixels: ({u_coord:.2f}, {v_coord:.2f})")

            # Check if inside image bounds
            if 0 <= u_coord < width and 0 <= v_coord < height:
                u_int, v_int = int(u_coord), int(v_coord)
                sensor_depth_at_uv = sensor_depth_map[v_int, u_int]
                depth_diff = abs(sensor_depth_at_uv - calculated_depth_from_projection)
                is_visible_occlusion_check = depth_diff <= self.vis_threshold

                print(f"    Sensor Depth at ({u_int},{v_int}): {sensor_depth_at_uv:.4f}")
                print(f"    Depth Difference: {depth_diff:.4f} (Threshold: {self.vis_threshold})")
                print(f"    Point Considered Visible (Occlusion Check): {is_visible_occlusion_check}")
                
                if color_image_pil:
                    projected_debug_points_x.append(u_int)
                    projected_debug_points_y.append(v_int)
                    colors_for_debug_points.append('lime' if is_visible_occlusion_check else 'magenta')
                    labels_for_debug_points.append(f"P{i+1}\nVis:{is_visible_occlusion_check}\nS_D:{sensor_depth_at_uv:.2f}\nP_D:{calculated_depth_from_projection:.2f}")

            else:
                print("    Point projects outside image bounds.")
                if color_image_pil: # Mark it outside
                    projected_debug_points_x.append(u_coord) # Plot actual coords even if outside
                    projected_debug_points_y.append(v_coord)
                    colors_for_debug_points.append('cyan')
                    labels_for_debug_points.append(f"P{i+1}\nOut")

        if color_image_pil and projected_debug_points_x:
            plt.scatter(projected_debug_points_x, projected_debug_points_y, c=colors_for_debug_points, s=60, edgecolors='black', zorder=10)
            for j, txt in enumerate(labels_for_debug_points):
                plt.text(projected_debug_points_x[j]+5, projected_debug_points_y[j]+5, txt, fontsize=7, color='white', bbox=dict(facecolor='black', alpha=0.5))
            plt.xlim(0, width)
            plt.ylim(height, 0) # Standard image coordinates
            plt.show()
        elif color_image_pil:
             plt.show() # Show empty image if no points made it to plotting stage
    
class FeaturesExtractor:
    def __init__(self, 
                 camera, 
                 clip_model, 
                 images, 
                 masks,
                 pointcloud,
                 sam_model_type,
                 sam_checkpoint,
                 vis_threshold,
                 device):
        self.camera = camera
        self.images = images
        self.device = device
        self.point_projector = PointProjector(camera, pointcloud, masks, vis_threshold, images.indices)
        self.predictor_sam = initialize_sam_model(device, sam_model_type, sam_checkpoint)
        self.clip_model, self.clip_preprocess = clip.load(clip_model, device)
        self.inpainting_model = SimpleLama()
        # iopaint.model.LaMa(device=device)
        
    
    def extract_features(self, topk, multi_level_expansion_ratio, num_levels, num_random_rounds, num_selected_points, save_crops, out_folder, optimize_gpu_usage=False):
        if(save_crops):
            out_folder = os.path.join(out_folder, "crops")
            os.makedirs(out_folder, exist_ok=True)
                            
        topk_indices_per_mask = self.point_projector.get_top_k_indices_per_mask(topk)
        
        num_masks = self.point_projector.masks.num_masks
        mask_clip = np.zeros((num_masks, 768)) #initialize mask clip
        
        np_images = self.images.get_as_np_list()
        for mask in tqdm(range(num_masks)): # for each mask 
            images_crops = []
            if(optimize_gpu_usage):
                self.clip_model.to(torch.device('cpu'))
                self.predictor_sam.model.cuda()
            for view_count, view in enumerate(topk_indices_per_mask[mask]): # for each view
                if(optimize_gpu_usage):
                    torch.cuda.empty_cache()
                
                # Get original mask points coordinates in 2d images
                point_coords = np.transpose(np.where(self.point_projector.visible_points_in_view_in_mask[view][mask] == True))
                if (point_coords.shape[0] > 0):
                    self.predictor_sam.set_image(np_images[view])
                    
                    # SAM
                    best_mask = run_sam(image_size=np_images[view],
                                        num_random_rounds=num_random_rounds,
                                        num_selected_points=num_selected_points,
                                        point_coords=point_coords,
                                        predictor_sam=self.predictor_sam,)
                    
                    mask = np.logical_and(
                        self.point_projector.visible_points_in_view_in_mask[view][mask],
                        np.logical_not(self.point_projector.visible_points_view[view][mask])
                    )

                    # Save all three masks for inspection
                    mask_dir = os.path.join(out_folder, f"masks_{mask}_{view}")
                    os.makedirs(mask_dir, exist_ok=True)
                    imageio.imwrite(os.path.join(mask_dir, "visible_points_in_view_in_mask.png"),
                                    self.point_projector.visible_points_in_view_in_mask[view][mask].astype(np.uint8) * 255)
                    imageio.imwrite(os.path.join(mask_dir, "visible_points_view.png"),
                                    self.point_projector.visible_points_view[view][mask].astype(np.uint8) * 255)
                    imageio.imwrite(os.path.join(mask_dir, "best_mask.png"),
                                    best_mask.astype(np.uint8) * 255)
                    imageio.imwrite(os.path.join(mask_dir, "final_mask.png"),
                                    mask.astype(np.uint8) * 255)
                    image_height, image_width = np_images[view].shape[:2]

                    # Save the original image next to the inpainted mask image
                    orig_img_path = os.path.join(out_folder, f"orig{mask}_{view}.png")
                    imageio.imwrite(orig_img_path, np_images[view])

                 
                    mask_size = min(image_height, image_width) // 4  # Size of the square (adjust as needed)
                    center_y, center_x = image_height // 2, image_width // 2
                    half_size = mask_size // 2

                    inpaintin_mask = np.zeros((image_height, image_width), dtype=np.uint8)
                    y1 = max(center_y - half_size, 0)
                    y2 = min(center_y + half_size, image_height)
                    x1 = max(center_x - half_size, 0)
                    x2 = min(center_x + half_size, image_width)
                    inpaintin_mask[y1:y2, x1:x2] = 1

                    # Save the mask used for inpainting
                    mask_path = os.path.join(out_folder, f"mask{mask}_{view}_mask.png")
                    imageio.imwrite(mask_path, inpaintin_mask * 255)

                    # Create InpaintRequest config for LaMa model
                    # config = iopaint.schema.InpaintRequest()
                    result = self.inpainting_model(np_images[view], inpaintin_mask)
                    # result = self.inpainting_model(np_images[view], inpaintin_mask, config)
                    
                    # Convert result to proper format for saving
                    if isinstance(result, np.ndarray):
                        # Convert from float to uint8 if needed
                        if result.dtype == np.float64 or result.dtype == np.float32:
                            result = (result * 255).astype(np.uint8)
                        result_to_save = result
                    else:
                        # If it's a PIL Image, convert to numpy array
                        result_to_save = np.array(result)
                    
                    print(f"Mask {mask} for view {view} processed. Result shape: {result_to_save.shape}, dtype: {result_to_save.dtype}")
                    # Save result
                    imageio.imwrite(os.path.join(out_folder, f"mask{mask}_{view}.png"), result_to_save)
                    
                    # MULTI LEVEL CROPS
                    for level in range(num_levels):
                        # get the bbox and corresponding crops
                        x1, y1, x2, y2 = mask2box_multi_level(torch.from_numpy(best_mask), level, multi_level_expansion_ratio)    

                        print("Uncropped size:", self.images.images[view].size)

                        cropped_img = self.images.images[view].crop((x1, y1, x2, y2))
                        
                        print("Crop size:", cropped_img.size)
                        
                        if(save_crops):
                            cropped_img.save(os.path.join(out_folder, f"crop{mask}_{view}_{level}.png"))
                            
                        # I compute the CLIP feature using the standard clip model
                        cropped_img_processed = self.clip_preprocess(cropped_img)

                        print("Processed crop size:", cropped_img_processed.size())
                        
                        images_crops.append(cropped_img_processed)
            
            if(optimize_gpu_usage):
                self.predictor_sam.model.cpu()
                self.clip_model.to(torch.device('cuda'))                
            if(len(images_crops) > 0):
                image_input = torch.tensor(np.stack(images_crops))
                with torch.no_grad():
                    image_features = self.clip_model.encode_image(image_input.to(self.device)).float()
                    image_features /= image_features.norm(dim=-1, keepdim=True) #normalize
                
                mask_clip[mask] = image_features.mean(axis=0).cpu().numpy()
                    
        return mask_clip
        
    def debug_mask_features(self, target_mask_idx, topk, 
                            multi_level_expansion_ratio, num_levels, 
                            num_random_rounds, num_selected_points, 
                            display_point_cloud=True):
        import matplotlib.pyplot as plt
        import open3d as o3d

        from PIL import ImageDraw # For drawing on images
        import numpy as np # Ensure numpy is imported
        import torch # Ensure torch is imported

        print(f"--- Debugging Mask Index: {target_mask_idx} ---")

         # Get the 3D points for the target mask
        # self.point_projector.masks.masks is (num_total_points, num_masks) boolean or float
        # self.point_projector.point_cloud.points is (num_total_points, 3)
        mask_point_indices_in_cloud = np.where(self.point_projector.masks.masks[:, target_mask_idx] > 0)[0]
        num_points_in_mask = len(mask_point_indices_in_cloud)
        print(f"Number of 3D points in target mask {target_mask_idx}: {num_points_in_mask}")

        if num_points_in_mask > 0:
            mask_points_3d = self.point_projector.point_cloud.points[mask_point_indices_in_cloud]
            
            # Visualize the isolated 3D mask
            if display_point_cloud:
                isolated_mask_pcd = o3d.geometry.PointCloud()
                isolated_mask_pcd.points = o3d.utility.Vector3dVector(mask_points_3d)
                isolated_mask_pcd.paint_uniform_color([0.0, 1.0, 0.0]) # Green for isolated mask
                print(f"Visualizing isolated 3D mask {target_mask_idx} (GREEN). Close window to continue.")
                o3d.visualization.draw_geometries([isolated_mask_pcd], 
                                                  window_name=f"Isolated Mask {target_mask_idx} ({num_points_in_mask} points)")
        else:
            print(f"Target mask {target_mask_idx} is empty in 3D (contains no points).")
            mask_points_3d = np.array([]) # Ensure it's an empty array for consistency if needed later

        # 0. Optionally display the full point cloud with the target mask highlighted
        if display_point_cloud:
            pcd_original_o3d = o3d.geometry.PointCloud()
            pcd_original_o3d.points = o3d.utility.Vector3dVector(self.point_projector.point_cloud.points)
            
            # mask_points_3d is already defined above
            geometries_to_draw = [pcd_original_o3d]
            if num_points_in_mask > 0: # Use num_points_in_mask check
                mask_pcd_highlight = o3d.geometry.PointCloud()
                mask_pcd_highlight.points = o3d.utility.Vector3dVector(mask_points_3d)
                mask_pcd_highlight.paint_uniform_color([1.0, 0.0, 0.0]) # Red for highlight in full scene
                geometries_to_draw.append(mask_pcd_highlight)
                print(f"Visualizing full point cloud. Target mask {target_mask_idx} (if points exist) is in RED.")
            else:
                print(f"Target mask {target_mask_idx} has no points in 3D to highlight in the full scene. Visualizing only full point cloud.")
            o3d.visualization.draw_geometries(geometries_to_draw, 
                                              window_name=f"Full Scene - Target Mask {target_mask_idx}")

        all_topk_indices = self.point_projector.get_top_k_indices_per_mask(topk)
        
        if target_mask_idx >= all_topk_indices.shape[0]:
            print(f"ERROR: target_mask_idx {target_mask_idx} is out of bounds for available masks ({all_topk_indices.shape[0]}).")
            return

        selected_views_for_target_mask = all_topk_indices[target_mask_idx]
        
        print(f"Top {topk} view indices (in projector's list) for mask {target_mask_idx}: {selected_views_for_target_mask}")
        
        original_image_indices_for_topk_views = [self.images.indices[view_idx_in_projector] for view_idx_in_projector in selected_views_for_target_mask]
        print(f"Corresponding original image file indices: {original_image_indices_for_topk_views}")

        # self.images.images is a list of PIL.Image objects
        # self.images.get_as_np_list() can be used if needed, but SAM takes np array
        
        # Pre-load numpy versions of images if not already done by get_as_np_list or if it's not stored
        # For safety, let's get them if SAM needs them directly.
        # The Images class already loads them as PIL images in self.images.
        # We can convert PIL to numpy for SAM.

        for i, view_idx_in_projector in enumerate(selected_views_for_target_mask):
            original_image_file_idx = self.images.indices[view_idx_in_projector]
            current_image_pil = self.images.images[view_idx_in_projector] # This is a PIL.Image
            current_image_np = np.array(current_image_pil) # Convert PIL to NumPy for SAM

            print(f"\n--- Processing View {i+1}/{topk} for Mask {target_mask_idx} ---")
            print(f"Projector View Index: {view_idx_in_projector}, Original Image File Index: {original_image_file_idx}")
            
            # Call the new occlusion debug method for this mask and view
            # Pass self.images so it can access the color image for visualization
            # self.point_projector.debug_occlusion_for_mask_in_view(
            #     target_mask_idx=target_mask_idx,
            #     view_idx_in_projector_list=view_idx_in_projector,
            #     images_obj=self.images, 
            #     num_points_to_debug=3 # You can change how many points to inspect
            # )

            plt.figure(figsize=(10, 7))
            plt.imshow(current_image_pil)
            plt.title(f"Mask {target_mask_idx} - View {i+1} (Orig File Idx: {original_image_file_idx}) - Full Image")
            plt.axis('off')
            plt.show()

            visible_mask_in_view_2d = self.point_projector.visible_points_in_view_in_mask[view_idx_in_projector, target_mask_idx]
            point_coords_2d_yx = np.transpose(np.where(visible_mask_in_view_2d == True)) # (row, col) i.e. (y,x)

            num_visible_points_in_mask_view = point_coords_2d_yx.shape[0]
            print(f"Number of visible points for mask {target_mask_idx} in this view: {num_visible_points_in_mask_view}")

            if num_visible_points_in_mask_view > 0:
                plt.figure(figsize=(10, 7))
                plt.imshow(current_image_pil)
                if point_coords_2d_yx.size > 0:
                    plt.scatter(point_coords_2d_yx[:, 1], point_coords_2d_yx[:, 0], s=10, c='red', marker='.') # Scatter takes (x,y)
                plt.title(f"Mask {target_mask_idx} - View {i+1} - Projected 3D Mask Points (Red)")
                plt.axis('off')
                plt.show()

                self.predictor_sam.set_image(current_image_np) # SAM expects H, W, C numpy array
                
                # run_sam expects point_coords as (N,2) with (y,x)
                best_sam_mask = run_sam(image_size=current_image_np.shape[:2], # (H,W)
                                        num_random_rounds=num_random_rounds,
                                        num_selected_points=num_selected_points,
                                        point_coords=point_coords_2d_yx, 
                                        predictor_sam=self.predictor_sam)

                plt.figure(figsize=(10, 7))
                plt.imshow(current_image_pil)
                plt.imshow(best_sam_mask, alpha=0.6, cmap='viridis')
                plt.title(f"Mask {target_mask_idx} - View {i+1} - SAM Mask Overlay")
                plt.axis('off')
                plt.show()

                print("Multi-level crops from SAM mask:")
                fig_crops, axes_crops = plt.subplots(1, num_levels, figsize=(num_levels * 4, 4))
                if num_levels == 1: axes_crops = [axes_crops] # Make it iterable

                for level in range(num_levels):
                    x1, y1, x2, y2 = mask2box_multi_level(torch.from_numpy(best_sam_mask), level, multi_level_expansion_ratio)
                    
                    img_width, img_height = current_image_pil.size
                    x1_c = max(0, int(x1))
                    y1_c = max(0, int(y1))
                    x2_c = min(img_width, int(x2))
                    y2_c = min(img_height, int(y2))

                    if x1_c < x2_c and y1_c < y2_c:
                        cropped_img_pil = current_image_pil.crop((x1_c, y1_c, x2_c, y2_c))
                        axes_crops[level].imshow(cropped_img_pil)
                        axes_crops[level].set_title(f"L{level}\nBox:({x1_c},{y1_c})-({x2_c},{y2_c})")
                        axes_crops[level].axis('off')
                    else:
                        print(f"  Level {level}: Invalid crop box ({x1_c},{y1_c})-({x2_c},{y2_c}). Original: ({x1},{y1})-({x2},{y2})")
                        axes_crops[level].text(0.5, 0.5, 'Invalid Crop', ha='center', va='center')
                        axes_crops[level].axis('off')
                plt.suptitle(f"Mask {target_mask_idx} - View {i+1} - Crops", fontsize=14)
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                plt.show()
            else:
                print(f"Skipping SAM and cropping for view (orig file idx {original_image_file_idx}) as no points of mask {target_mask_idx} are visible.")