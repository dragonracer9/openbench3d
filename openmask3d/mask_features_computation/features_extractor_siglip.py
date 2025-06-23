# from transformers import SiglipProcessor, SiglipModel
# import numpy as np
# import imageio
# import torch
# from tqdm import tqdm
# import os
# from openmask3d.data.load import Camera, InstanceMasks3D, Images, PointCloud, get_number_of_images
# from openmask3d.mask_features_computation.utils import initialize_sam_model, mask2box_multi_level, run_sam
# from PIL import Image
# from PIL import ImageOps
# from torchvision.transforms import Resize, CenterCrop, Compose
# from torchvision.transforms import InterpolationMode

# class PointProjector:
#     def __init__(self, camera: Camera, 
#                  point_cloud: PointCloud, 
#                  masks: InstanceMasks3D, 
#                  vis_threshold, 
#                  indices):
#         self.vis_threshold = vis_threshold
#         self.indices = indices
#         self.camera = camera
#         self.point_cloud = point_cloud
#         self.masks = masks
#         self.visible_points_in_view_in_mask, self.visible_points_view, self.projected_points, self.resolution = self.get_visible_points_in_view_in_mask()
        
        
#     def get_visible_points_view(self):
#         # Initialization
#         vis_threshold = self.vis_threshold
#         indices = self.indices
#         depth_scale = self.camera.depth_scale
#         poses = self.camera.load_poses(indices)
#         X = self.point_cloud.get_homogeneous_coordinates()
#         n_points = self.point_cloud.num_points
#         depths_path = self.camera.depths_path        
#         resolution = imageio.imread(os.path.join(depths_path, '0.png')).shape
#         height = resolution[0]
#         width = resolution[1]
#         intrinsic = self.camera.get_adapted_intrinsic(resolution)
        
#         projected_points = np.zeros((len(indices), n_points, 2), dtype = int)
#         visible_points_view = np.zeros((len(indices), n_points), dtype = bool)
#         print(f"[INFO] Computing the visible points in each view.")
        
#         for i, idx in tqdm(enumerate(indices)): # for each view
#             # *******************************************************************************************************************
#             # STEP 1: get the projected points
#             # Get the coordinates of the projected points in the i-th view (i.e. the view with index idx)
#             projected_points_not_norm = (intrinsic @ poses[i] @ X.T).T
#             # Get the mask of the points which have a non-null third coordinate to avoid division by zero
#             mask = (projected_points_not_norm[:, 2] != 0) # don't do the division for point with the third coord equal to zero
#             # Get non homogeneous coordinates of valid points (2D in the image)
#             projected_points[i][mask] = np.column_stack([[projected_points_not_norm[:, 0][mask]/projected_points_not_norm[:, 2][mask], 
#                     projected_points_not_norm[:, 1][mask]/projected_points_not_norm[:, 2][mask]]]).T
            
#             # *******************************************************************************************************************
#             # STEP 2: occlusions computation
#             # Load the depth from the sensor
#             depth_path = os.path.join(depths_path, str(idx) + '.png')
#             sensor_depth = imageio.imread(depth_path) / depth_scale
#             inside_mask = (projected_points[i,:,0] >= 0) * (projected_points[i,:,1] >= 0) \
#                                 * (projected_points[i,:,0] < width) \
#                                 * (projected_points[i,:,1] < height)
#             pi = projected_points[i].T
#             # Depth of the points of the pointcloud, projected in the i-th view, computed using the projection matrices
#             point_depth = projected_points_not_norm[:,2]
#             # Compute the visibility mask, true for all the points which are visible from the i-th view
#             visibility_mask = (np.abs(sensor_depth[pi[1][inside_mask], pi[0][inside_mask]]
#                                         - point_depth[inside_mask]) <= \
#                                         vis_threshold).astype(bool)
#             inside_mask[inside_mask == True] = visibility_mask
#             visible_points_view[i] = inside_mask
#         return visible_points_view, projected_points, resolution
    
#     def get_bbox(self, mask, view):
#         if(self.visible_points_in_view_in_mask[view][mask].sum()!=0):
#             true_values = np.where(self.visible_points_in_view_in_mask[view, mask])
#             valid = True
#             t, b, l, r = true_values[0].min(), true_values[0].max()+1, true_values[1].min(), true_values[1].max()+1 
#         else:
#             valid = False
#             t, b, l, r = (0,0,0,0)
#         return valid, (t, b, l, r)
    
#     def get_visible_points_in_view_in_mask(self):
#         masks = self.masks
#         num_view = len(self.indices)
#         visible_points_view, projected_points, resolution = self.get_visible_points_view()
#         visible_points_in_view_in_mask = np.zeros((num_view, masks.num_masks, resolution[0], resolution[1]), dtype=bool)
#         print(f"[INFO] Computing the visible points in each view in each mask.")
#         for i in tqdm(range(num_view)):
#             for j in range(masks.num_masks):
#                 visible_masks_points = (masks.masks[:,j] * visible_points_view[i]) > 0
#                 proj_points = projected_points[i][visible_masks_points]
#                 if(len(proj_points) != 0):
#                     visible_points_in_view_in_mask[i][j][proj_points[:,1], proj_points[:,0]] = True
#         self.visible_points_in_view_in_mask = visible_points_in_view_in_mask
#         self.visible_points_view = visible_points_view
#         self.projected_points = projected_points
#         self.resolution = resolution
#         return visible_points_in_view_in_mask, visible_points_view, projected_points, resolution
    
#     def get_top_k_indices_per_mask(self, k):
#         num_points_in_view_in_mask = self.visible_points_in_view_in_mask.sum(axis=2).sum(axis=2)
#         topk_indices_per_mask = np.argsort(-num_points_in_view_in_mask, axis=0)[:k,:].T
#         return topk_indices_per_mask
    
# class FeaturesExtractorSiglip:
#     def __init__(self, 
#                  camera, 
#                  siglip_model, # google/siglip-base-patch16-224
#                  images, 
#                  masks,
#                  pointcloud,
#                  sam_model_type,
#                  sam_checkpoint,
#                  vis_threshold,
#                  device):
#         self.camera = camera
#         self.images = images
#         self.device = device
#         self.point_projector = PointProjector(camera, pointcloud, masks, vis_threshold, images.indices)
#         self.predictor_sam = initialize_sam_model(device, sam_model_type, sam_checkpoint)
#         # self.clip_model, self.clip_preprocess = clip.load(clip_model, device)
#         # ——— HuggingFace SigLIP setup ———
#         self.processor = SiglipProcessor.from_pretrained(siglip_model)
#         self.model = SiglipModel.from_pretrained(siglip_model).to(device)

#         # run a dummy forward to infer embedding dim
#         # infer image dimension from images
#         dummy_img = Image.new('RGB', (224, 224), color=0)  # Create a dummy image
#         # Use the same approach as our successful test
#         dummy_inputs = self.processor(images=dummy_img, padding="max_length", return_tensors="pt")
#         px = dummy_inputs["pixel_values"].to(device)
#         # Infer image dimension from the dummy image
#         w, h = px.shape[-2], px.shape[-1]
#         if w != h:
#             print("[WARNING] The input image after passing through preprocessing is not square.")
#         with torch.no_grad():
#             dummy_feat = self.model.get_image_features(px)
        
#         self.feature_dim = dummy_feat.shape[-1]
#         # print(f"[INFO] SigLIP feature dimension: {self.feature_dim}")
#         # print(f"[INFO] SigLIP model and processor initialized successfully")
#         # BICUBIC = InterpolationMode.BICUBIC
#         # self.transform = Compose([
#         #                             Resize(w, interpolation=BICUBIC),
#         #                             CenterCrop(w)
#         #                         ])
    
#     def extract_features(self, topk, multi_level_expansion_ratio, num_levels, num_random_rounds, num_selected_points, save_crops, out_folder, optimize_gpu_usage=False):
#         if(save_crops):
#             out_folder = os.path.join(out_folder, "crops")
#             os.makedirs(out_folder, exist_ok=True)
                            
#         topk_indices_per_mask = self.point_projector.get_top_k_indices_per_mask(topk)
        
#         num_masks = self.point_projector.masks.num_masks
#         mask_siglip = np.zeros((num_masks, self.feature_dim))#), dtype=np.float32) #initialize mask clip
        
#         np_images = self.images.get_as_np_list()
#         for mask in tqdm(range(num_masks)): # for each mask 
#             images_crops = []
#             if(optimize_gpu_usage):
#                 self.model.to(torch.device('cpu'))
#                 self.predictor_sam.model.cuda()
#             for view_count, view in enumerate(topk_indices_per_mask[mask]): # for each view
#                 if(optimize_gpu_usage):
#                     torch.cuda.empty_cache()
                
#                 # Get original mask points coordinates in 2d images
#                 point_coords = np.transpose(np.where(self.point_projector.visible_points_in_view_in_mask[view][mask] == True))
#                 if (point_coords.shape[0] > 0):
#                     self.predictor_sam.set_image(np_images[view])
                    
#                     # SAM
#                     best_mask = run_sam(image_size=np_images[view],
#                                         num_random_rounds=num_random_rounds,
#                                         num_selected_points=num_selected_points,
#                                         point_coords=point_coords,
#                                         predictor_sam=self.predictor_sam,)
                    
#                     # MULTI LEVEL CROPS
#                     for level in range(num_levels):
#                         # get the bbox and corresponding crops
#                         x1, y1, x2, y2 = mask2box_multi_level(torch.from_numpy(best_mask), level, multi_level_expansion_ratio)
#                         # cropped_img = self.images.images[view].crop((x1, y1, x2, y2))
                        
#                         # print("Uncropped size:", self.images.images[view].size)

#                         cropped_img = self.images.images[view].crop((x1, y1, x2, y2))
                        
#                         # print("Crop size:", cropped_img.size)
                        
#                         if(save_crops):
#                             cropped_img.save(os.path.join(out_folder, f"crop{mask}_{view}_{level}.png"))
                        
#                         # Save some sample crops for debugging (first 3 masks, up to 6 crops each)
#                         if mask < 3 and len(images_crops) < 6:
#                             debug_folder = os.path.join(out_folder, "../debug_crops")
#                             os.makedirs(debug_folder, exist_ok=True)
#                             cropped_img.save(os.path.join(debug_folder, f"mask{mask:03d}_view{view:03d}_level{level}_crop.png"))
                            
#                         images_crops.append(cropped_img)
#                         # I compute the CLIP feature using the standard clip model
#                         # cropped_img_processed = self.clip_preprocess(cropped_img)
#                         # images_crops.append(cropped_img_processed)
                        
#                         # cropped_img = self.transform(cropped_img)
                        
#                         # print("Padded crop size:", cropped_img.size)
                        
#                         # px = self.processor.image_processor.preprocess(images=cropped_img, return_tensors="pt")["pixel_values"].squeeze(0)
#                         # print(type(self.processor))
#                         # print("Processed crop shape:", px.shape, "sample values:", px.flatten()[:10])
                
                        
#                         # print("Processed crop size:", px.size())
                        
#                         # images_crops.append(px)
#                         # print(f"[INFO] Crop {level} for mask {mask} in view {view} is of size: {cropped_img.size}")
#                         # print(f"[INFO] First few pixels: {np.array(cropped_img).flatten()[:10]}")
#             # print(f"[INFO] I'm here")
#             if(optimize_gpu_usage):
#                 self.predictor_sam.model.cpu()
#                 self.model.to(torch.device('cuda'))                
#             if(len(images_crops) > 0):
#                 # print(f"[INFO] Number of crops for mask {mask} is {len(images_crops)}")
#                 # CRITICAL: Use padding="max_length" as proven in our test
#                 inputs = self.processor(images=images_crops, padding="max_length", return_tensors="pt")
#                 # print(f"[INFO] Image input keys: {inputs.keys()}")
#                 # if "pixel_values" in inputs:
#                 #     print(f"[INFO] Pixel values shape: {inputs['pixel_values'].shape}")
                
#                 with torch.no_grad():
#                     # Extract image features using the same approach as our successful test
#                     image_features = self.model.get_image_features(inputs["pixel_values"].to(self.device))
#                     # print(f"[INFO] Image features shape: {image_features.shape}, dtype: {image_features.dtype}")
#                     # print(f"[INFO] Raw image features range: {image_features.min():.3f} - {image_features.max():.3f}")
                    
#                     # Normalize features (critical for cosine similarity)
#                     image_features = image_features / image_features.norm(dim=-1, keepdim=True)
#                     # print(f"[INFO] Normalized features range: {image_features.min():.3f} - {image_features.max():.3f}")
#                     # print(f"[INFO] Feature norms: {image_features.norm(dim=-1)}")
                    
#                     # Debug: Check if crops are actually different
#                     # if image_features.shape[0] > 1:
#                     #     # Compute pairwise similarities between crops
#                     #     crop_similarities = torch.mm(image_features, image_features.T)
#                     #     # print(f"[INFO] Crop similarities for mask {mask}: min={crop_similarities.min():.3f}, max={crop_similarities.max():.3f}, mean={crop_similarities.mean():.3f}")
#                     #     # Show off-diagonal similarities (excluding self-similarity)
#                     #     off_diag = crop_similarities[torch.triu(torch.ones_like(crop_similarities), diagonal=1) == 1]
#                     #     if len(off_diag) > 0:
#                     #         print(f"[INFO] Off-diagonal similarities: {off_diag[:5].tolist()}")
                    
#                     # Average the normalized features (this preserves the normalization)
#                     mean_feat = image_features.mean(axis=0)
#                     # Re-normalize after averaging (important!)
#                     mean_feat = mean_feat / mean_feat.norm()
                    
#                 # print(f"[INFO] Final mean feature for mask {mask} shape: {mean_feat.shape}, norm: {mean_feat.norm():.3f}")
#                 # print(f"[INFO] Mean feature range: {mean_feat.min():.3f} - {mean_feat.max():.3f}")
#                 mask_siglip[mask] = mean_feat.cpu().numpy()
#                 # print(f"[INFO] Mask features are of shape: {mask_siglip.shape} and with values in range {mask_siglip.min()} - {mask_siglip.max()}")
#                 # print(f"[INFO] Sample values for mask {mask}: {mask_siglip[mask][:10]}")
                
#                 # Debug: Check similarity with previous masks
#                 # if mask > 0:
#                 #     prev_mask_idx = mask - 1
#                 #     while prev_mask_idx >= 0 and np.linalg.norm(mask_siglip[prev_mask_idx]) == 0:
#                 #         prev_mask_idx -= 1
#                 #     if prev_mask_idx >= 0:
#                 #         similarity = np.dot(mask_siglip[mask], mask_siglip[prev_mask_idx])
#                 #         print(f"[INFO] Similarity between mask {mask} and mask {prev_mask_idx}: {similarity:.4f}")
                        
#         # Final analysis
#         # print(f"\n[INFO] === FINAL FEATURE ANALYSIS ===")
#         # non_zero_masks = [i for i in range(num_masks) if np.linalg.norm(mask_siglip[i]) > 0]
#         # print(f"[INFO] Non-zero masks: {len(non_zero_masks)}/{num_masks}")
        
#         # if len(non_zero_masks) > 1:
#         #     # Compute pairwise similarities between all mask features
#         #     similarities = []
#         #     for i in range(len(non_zero_masks)):
#         #         for j in range(i+1, len(non_zero_masks)):
#         #             sim = np.dot(mask_siglip[non_zero_masks[i]], mask_siglip[non_zero_masks[j]])
#         #             similarities.append(sim)
#         #     similarities = np.array(similarities)
#         #     print(f"[INFO] Mask-to-mask similarities: min={similarities.min():.4f}, max={similarities.max():.4f}, mean={similarities.mean():.4f}")
#         #     print(f"[INFO] Sample similarities: {similarities[:10].tolist()}")
                    
#         return mask_siglip


from transformers import SiglipProcessor, SiglipModel
import numpy as np
import imageio
import torch
from tqdm import tqdm
import os
from openmask3d.data.load import Camera, InstanceMasks3D, Images, PointCloud, get_number_of_images
from openmask3d.mask_features_computation.utils import initialize_sam_model, mask2box_multi_level, run_sam
from PIL import Image
from PIL import ImageOps
from torchvision.transforms import Resize, CenterCrop, Compose
from torchvision.transforms import InterpolationMode
import time

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
        # self.clip_model, self.clip_preprocess = clip.load(clip_model, device)
        # ——— HuggingFace SigLIP setup ———
        self.processor = SiglipProcessor.from_pretrained(siglip_model)
        self.model = SiglipModel.from_pretrained(siglip_model).to(device)

        # run a dummy forward to infer embedding dim
        # infer image dimension from images
        dummy_img = Image.new('RGB', (224, 224), color=0)  # Create a dummy image
        # Use the same approach as our successful test
        dummy_inputs = self.processor(images=dummy_img, padding="max_length", return_tensors="pt")
        px = dummy_inputs["pixel_values"].to(device)
        # Infer image dimension from the dummy image
        w, h = px.shape[-2], px.shape[-1]
        if w != h:
            print("[WARNING] The input image after passing through preprocessing is not square.")
        with torch.no_grad():
            dummy_feat = self.model.get_image_features(px)
        
        self.feature_dim = dummy_feat.shape[-1]
        # print(f"[INFO] SigLIP feature dimension: {self.feature_dim}")
        # print(f"[INFO] SigLIP model and processor initialized successfully")
        # BICUBIC = InterpolationMode.BICUBIC
        # self.transform = Compose([
        #                             Resize(w, interpolation=BICUBIC),
        #                             CenterCrop(w)
        #                         ])
    
    def extract_features(self, topk, multi_level_expansion_ratio, num_levels, num_random_rounds, num_selected_points, save_crops, out_folder, optimize_gpu_usage=False):
        # Set random seeds again for consistent behavior across runs
        import random
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Add timing for the entire function
        total_start_time = time.time()
        
        if(save_crops):
            out_folder = os.path.join(out_folder, "crops")
            os.makedirs(out_folder, exist_ok=True)
                            
        topk_indices_per_mask = self.point_projector.get_top_k_indices_per_mask(topk)
        
        num_masks = self.point_projector.masks.num_masks
        mask_siglip = np.zeros((num_masks, self.feature_dim))
        
        np_images = self.images.get_as_np_list()
        
        print(f"[TIMING] Starting feature extraction for {num_masks} masks")
        print(f"[TIMING] SAM parameters: num_random_rounds={num_random_rounds}, num_selected_points={num_selected_points}")
        
        # Reorganize processing to be view-centric for better SAM encoding efficiency
        view_to_masks = {}  # view -> list of (mask, mask_crops_storage)
        for mask in range(num_masks):
            for view in topk_indices_per_mask[mask]:
                if view not in view_to_masks:
                    view_to_masks[view] = []
                view_to_masks[view].append(mask)
        
        print(f"[TIMING] Processing {len(view_to_masks)} unique views for {num_masks} masks")
        
        # Storage for mask crops - will be populated during view-centric processing
        mask_crops = {mask: [] for mask in range(num_masks)}
        
        sam_total_time = 0
        crop_total_time = 0
        siglip_total_time = 0
        
        # Process by view to minimize SAM re-encoding
        with tqdm(total=len(view_to_masks), desc="SAM View Processing") as pbar:
            for view_idx, (view, masks_using_view) in enumerate(view_to_masks.items()):
                pbar.set_description(f"SAM View {view} ({len(masks_using_view)} masks)")
                
                # SAM image encoding - only once per view
                set_image_start = time.time()
                
                if(optimize_gpu_usage):
                    # Move SigLIP to CPU, SAM to GPU for this view
                    # Note: This will be moved back when we do SigLIP processing
                    pass  # We'll handle GPU optimization per mask later
                    
                self.predictor_sam.set_image(np_images[view])
                set_image_time = time.time() - set_image_start
                
                # Process all masks that use this view
                for mask in masks_using_view:
                    # Get original mask points coordinates in 2d images
                    point_coords = np.transpose(np.where(self.point_projector.visible_points_in_view_in_mask[view][mask] == True))
                    if (point_coords.shape[0] > 0):
                        # Time SAM processing (image already encoded)
                        sam_start = time.time()
                        best_mask = run_sam(image_size=np_images[view],
                                            num_random_rounds=num_random_rounds,
                                            num_selected_points=num_selected_points,
                                            point_coords=point_coords,
                                            predictor_sam=self.predictor_sam,
                                            mask_id=mask,
                                            view_id=view)
                        sam_time = time.time() - sam_start
                        sam_total_time += sam_time
                        
                        # Time crop processing
                        crop_start = time.time()
                        # MULTI LEVEL CROPS
                        for level in range(num_levels):
                            # get the bbox and corresponding crops
                            x1, y1, x2, y2 = mask2box_multi_level(torch.from_numpy(best_mask), level, multi_level_expansion_ratio)

                            cropped_img = self.images.images[view].crop((x1, y1, x2, y2))
                            
                            if(save_crops):
                                cropped_img.save(os.path.join(out_folder, f"crop{mask}_{view}_{level}.png"))
                                
                            mask_crops[mask].append(cropped_img)
                        crop_time = time.time() - crop_start
                        crop_total_time += crop_time
                
                # Add encoding time distributed across masks that used this view
                sam_total_time += set_image_time
                pbar.update(1)
            
        # Now process SigLIP features for each mask
        print(f"[TIMING] Processing SigLIP features for all masks")
        
        if(optimize_gpu_usage):
            # Move SAM to CPU, SigLIP to GPU for feature extraction
            self.predictor_sam.model.cpu()
            self.model.to(torch.device('cuda'))
        
        for mask in tqdm(range(num_masks), desc="SigLIP Processing"):
            if len(mask_crops[mask]) > 0:
                # Time SigLIP processing
                siglip_start = time.time()
                
                # CRITICAL: Use padding="max_length" as proven in our test
                inputs = self.processor(images=mask_crops[mask], padding="max_length", return_tensors="pt")
                
                with torch.no_grad():
                    # Extract image features using the same approach as our successful test
                    image_features = self.model.get_image_features(inputs["pixel_values"].to(self.device))
                    
                    # Normalize features (critical for cosine similarity)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    
                    # Average the normalized features (this preserves the normalization)
                    mean_feat = image_features.mean(axis=0)
                    # Re-normalize after averaging (important!)
                    mean_feat = mean_feat / mean_feat.norm()
                    
                mask_siglip[mask] = mean_feat.cpu().numpy()
                
                siglip_time = time.time() - siglip_start
                siglip_total_time += siglip_time
                
        total_time = time.time() - total_start_time
        
        # Calculate SAM encoding efficiency  
        total_views_processed = sum(len(topk_indices_per_mask[mask]) for mask in range(num_masks))
        unique_views_encoded = len(view_to_masks)
        
        print(f"\n[TIMING] === FEATURE EXTRACTION COMPLETE ===")
        print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"SAM time: {sam_total_time:.1f}s ({sam_total_time/total_time*100:.1f}%)")
        print(f"Cropping time: {crop_total_time:.1f}s ({crop_total_time/total_time*100:.1f}%)")
        print(f"SigLIP time: {siglip_total_time:.1f}s ({siglip_total_time/total_time*100:.1f}%)")
        print(f"Average time per mask: {total_time/num_masks:.1f}s")
        print(f"SAM encoding efficiency: {unique_views_encoded} unique views encoded vs {total_views_processed} total view processings")
        print(f"SAM encoding savings: {((total_views_processed - unique_views_encoded) / total_views_processed * 100):.1f}% fewer encodings")
                    
        return mask_siglip