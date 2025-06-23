from transformers import SiglipProcessor, SiglipModel
from tqdm import tqdm
from openmask3d.data.load import Camera, InstanceMasks3D, Images, PointCloud, get_number_of_images
from openmask3d.mask_features_computation.utils import initialize_sam_model, mask2box_multi_level, run_sam
from PIL import Image
from PIL import ImageOps
from torchvision.transforms import Resize, CenterCrop, Compose
from torchvision.transforms import InterpolationMode
import numpy as np
import imageio
import torch
import os
import gc
import psutil

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
    
class FeaturesExtractorSiglip:
    def __init__(self, 
                 camera, 
                 siglip_model, # google/siglip-base-patch16-224
                 images, 
                 masks,
                 pointcloud,
                 sam_model_type,
                 sam_checkpoint,
                 vis_threshold,
                 device):
        # Set random seeds for deterministic behavior
        import random
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        import os
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        self.camera = camera
        self.images = images
        self.device = device
        self.point_projector = PointProjector(camera, pointcloud, masks, vis_threshold, images.indices)
        self.predictor_sam = initialize_sam_model(device, sam_model_type, sam_checkpoint)
        
        # ——— HuggingFace SigLIP setup ———
        self.processor = SiglipProcessor.from_pretrained(siglip_model)
        self.model = SiglipModel.from_pretrained(siglip_model).to(device)

        # run a dummy forward to infer embedding dim
        dummy_img = Image.new('RGB', (224, 224), color=0)
        dummy_inputs = self.processor(images=dummy_img, padding="max_length", return_tensors="pt")
        px = dummy_inputs["pixel_values"].to(device)
        w, h = px.shape[-2], px.shape[-1]
        if w != h:
            print("[WARNING] The input image after passing through preprocessing is not square.")
        with torch.no_grad():
            dummy_feat = self.model.get_image_features(px)
        
        self.feature_dim = dummy_feat.shape[-1]
        
        # Clean up dummy computation
        del dummy_img, dummy_inputs, px, dummy_feat
        torch.cuda.empty_cache()
        gc.collect()

    def _clear_memory(self, force_gc=False):
        """Comprehensive memory cleanup"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        if force_gc:
            gc.collect()
    
    def _log_memory_usage(self, stage=""):
        """Log current memory usage for debugging"""
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
            gpu_cached = torch.cuda.memory_reserved() / 1024**3   # GB
            print(f"[MEMORY] {stage}: GPU allocated: {gpu_memory:.2f}GB, reserved: {gpu_cached:.2f}GB")
        
        process = psutil.Process(os.getpid())
        cpu_memory = process.memory_info().rss / 1024**3  # GB
        print(f"[MEMORY] {stage}: CPU memory: {cpu_memory:.2f}GB")

    def extract_features(self, topk, multi_level_expansion_ratio, num_levels, num_random_rounds, num_selected_points, save_crops, out_folder, optimize_gpu_usage=False):
        """Memory-optimized feature extraction"""
        
        # Set random seeds again for consistent behavior across runs
        import random
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        print("[INFO] Starting memory-optimized feature extraction")
        self._log_memory_usage("Start")
        
        if save_crops:
            out_folder = os.path.join(out_folder, "crops")
            os.makedirs(out_folder, exist_ok=True)
                            
        topk_indices_per_mask = self.point_projector.get_top_k_indices_per_mask(topk)
        
        num_masks = self.point_projector.masks.num_masks
        mask_siglip = np.zeros((num_masks, self.feature_dim), dtype=np.float32)
        
        np_images = self.images.get_as_np_list()
        
        # Process each mask with aggressive memory management
        for mask in tqdm(range(num_masks), desc="Processing masks"):
            self._process_single_mask(
                mask, topk_indices_per_mask, np_images, num_levels,
                multi_level_expansion_ratio, num_random_rounds, num_selected_points,
                save_crops, out_folder, optimize_gpu_usage, mask_siglip
            )
            
            # Aggressive memory cleanup every few masks
            if mask % 10 == 0:
                self._clear_memory(force_gc=True)
                self._log_memory_usage(f"After mask {mask}")
        
        print("[INFO] Feature extraction completed")
        self._log_memory_usage("End")
        return mask_siglip

    def _process_single_mask(self, mask, topk_indices_per_mask, np_images, num_levels,
                           multi_level_expansion_ratio, num_random_rounds, num_selected_points,
                           save_crops, out_folder, optimize_gpu_usage, mask_siglip):
        """Process a single mask with memory management"""
        
        # Clear any leftover GPU memory from previous mask
        self._clear_memory()
        
        # Collect crops in batches to avoid memory accumulation
        all_crops = []
        batch_size = 32  # Process crops in smaller batches
        
        if optimize_gpu_usage:
            self.model.to(torch.device('cpu'))
            self.predictor_sam.model.cuda()
        
        for view_count, view in enumerate(topk_indices_per_mask[mask]):
            if optimize_gpu_usage:
                self._clear_memory()
            
            # Get original mask points coordinates in 2d images
            point_coords = np.transpose(np.where(self.point_projector.visible_points_in_view_in_mask[view][mask] == True))
            
            if point_coords.shape[0] > 0:
                # Clear any previous SAM state
                self.predictor_sam.reset_image()
                self.predictor_sam.set_image(np_images[view])
                
                # SAM processing
                best_mask = run_sam(
                    image_size=np_images[view],
                    num_random_rounds=num_random_rounds,
                    num_selected_points=num_selected_points,
                    point_coords=point_coords,
                    predictor_sam=self.predictor_sam,
                )
                
                # Multi-level crops
                view_crops = []
                for level in range(num_levels):
                    x1, y1, x2, y2 = mask2box_multi_level(
                        torch.from_numpy(best_mask), level, multi_level_expansion_ratio
                    )
                    
                    cropped_img = self.images.images[view].crop((x1, y1, x2, y2))
                    
                    if save_crops:
                        cropped_img.save(os.path.join(out_folder, f"crop{mask}_{view}_{level}.png"))
                    
                    # Convert to RGB if needed and ensure consistent format
                    if cropped_img.mode != 'RGB':
                        cropped_img = cropped_img.convert('RGB')
                    
                    view_crops.append(cropped_img)
                
                all_crops.extend(view_crops)
                
                # Process crops in batches if we have too many
                if len(all_crops) >= batch_size:
                    self._process_crop_batch(all_crops[:batch_size], mask_siglip, mask, optimize_gpu_usage)
                    all_crops = all_crops[batch_size:]
                    self._clear_memory()
                
                # Clean up intermediate variables
                del best_mask, view_crops
                
        # Process remaining crops
        if all_crops:
            self._process_crop_batch(all_crops, mask_siglip, mask, optimize_gpu_usage)
        
        # Clean up all crops from memory
        del all_crops
        self._clear_memory()

    def _process_crop_batch(self, crops, mask_siglip, mask, optimize_gpu_usage):
        """Process a batch of crops to extract features"""
        if not crops:
            return
        
        if optimize_gpu_usage:
            self.predictor_sam.model.cpu()
            self.model.to(torch.device('cuda'))
        
        try:
            # Process crops with SigLIP
            inputs = self.processor(images=crops, padding="max_length", return_tensors="pt")
            
            with torch.no_grad():
                # Extract and normalize features
                image_features = self.model.get_image_features(inputs["pixel_values"].to(self.device))
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # Average the normalized features
                mean_feat = image_features.mean(axis=0)
                mean_feat = mean_feat / mean_feat.norm()
                
                # Update mask features (accumulate if this is not the first batch for this mask)
                current_feat = mask_siglip[mask]
                if np.linalg.norm(current_feat) == 0:
                    # First batch for this mask
                    mask_siglip[mask] = mean_feat.cpu().numpy()
                else:
                    # Accumulate with previous batches (weighted average)
                    prev_feat = torch.from_numpy(current_feat).to(self.device)
                    prev_feat = prev_feat / prev_feat.norm()
                    
                    # Simple average (could be weighted by number of crops)
                    combined_feat = (prev_feat + mean_feat) / 2
                    combined_feat = combined_feat / combined_feat.norm()
                    mask_siglip[mask] = combined_feat.cpu().numpy()
                
                # Clean up tensors
                del image_features, mean_feat, inputs
                
        except Exception as e:
            print(f"[ERROR] Processing crop batch for mask {mask}: {e}")
        
        finally:
            # Clean up crops from memory
            for crop in crops:
                if hasattr(crop, 'close'):
                    crop.close()
            del crops
            self._clear_memory()
