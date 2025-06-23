import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import clip
import numpy as np
import imageio
import torch
import cv2
from tqdm import tqdm
from openmask3d.data.load import Camera, InstanceMasks3D, Images, PointCloud, get_number_of_images
from openmask3d.mask_features_computation.utils import initialize_sam_model, mask2box_multi_level, run_sam, set_global_seeds
from simple_lama_inpainting import SimpleLama
from PIL import Image

from transformers import (
    CLIPProcessor,
    CLIPModel,
    BlipProcessor,
    BlipModel
)

class PointProjector:
    def __init__(self,
                 camera: "Camera",
                 point_cloud: "PointCloud",
                 masks: "InstanceMasks3D",
                 vis_threshold: float,
                 indices: list):
        """
        camera:        an object that knows how to load intrinsics, depth images, and poses for each index
        point_cloud:   an object exposing .num_points and .get_homogeneous_coordinates() → (N×4)
        masks:         an object with .num_masks and .masks (shape = [N, num_masks]) labeling which 3D points belong to which mask
        vis_threshold: allowable depth difference (in camera‐space units) between projected point and sensor depth
        indices:       list of integer frame‐indices (e.g. [0, 5, 10]) for which to precompute visibility
        """
        self.vis_threshold = vis_threshold
        self.indices       = indices
        self.camera        = camera
        self.point_cloud   = point_cloud
        self.masks         = masks

        # After calling get_visible_points_view(), we will have:
        #   visible_points_view[i, p]  = True  if point p is visible in view i
        #   projected_uv[i, p, :]     = (u, v) pixel coords of point p in view i (integers)
        #   projected_depths[i, p]    = z_cam  (camera‐space depth) of point p in view i
        #   resolution                = (height, width) of each depth image
        self.visible_points_view = None
        self.projected_uv        = None
        self.projected_depths    = None
        self.resolution          = None

        # Compute the above arrays once (for all requested indices)
        (self.visible_points_view,
         self.projected_uv,
         self.projected_depths,
         self.resolution) = self.get_visible_points_view()

        # Now build a per‐view, per‐mask 2D binary image marking exactly where each
        # mask’s VISIBLE 3D points land in pixel‐space.
        self.visible_points_in_view_in_mask = self.get_visible_points_in_view_in_mask()


    def get_visible_points_view(self):
        """
        Projects all 3D points into each selected view, computes camera‐space depth,
        and compares against the sensor depth image to determine visibility.

        Returns:
            visible_points_view:  np.bool array [num_views, num_points]
            projected_uv:         np.int   array [num_views, num_points, 2]
            projected_depths:     np.float array [num_views, num_points]
            resolution:           tuple (height, width)
        """

        vis_threshold = self.vis_threshold
        indices       = self.indices
        depth_scale   = self.camera.depth_scale
        poses         = self.camera.load_poses(indices)            # list of 4×4 pose matrices
        X_hom         = self.point_cloud.get_homogeneous_coordinates()  # (num_points × 4)
        n_points      = self.point_cloud.num_points
        depths_path   = self.camera.depths_path

        # Read one depth image just to figure out (height, width)
        sample_depth   = imageio.imread(os.path.join(depths_path, f"{indices[0]}.png"))
        height, width  = sample_depth.shape[:2]
        resolution     = (height, width)

        intrinsic = self.camera.get_adapted_intrinsic(resolution)  # 3×3 camera matrix

        # Allocate:
        #   - projected_uv      : (num_views, num_points, 2)
        #   - projected_depths  : (num_views, num_points)
        #   - visible_points_view: (num_views, num_points) boolean
        num_views = len(indices)
        projected_uv       = np.zeros((num_views, n_points, 2), dtype=np.int32)
        projected_depths   = np.zeros((num_views, n_points), dtype=np.float32)
        visible_points_view = np.zeros((num_views, n_points), dtype=bool)

        print("[INFO] Computing projected_uv, projected_depths, and visibility for each view.")
        for i, idx in enumerate(indices):
            # 1) Project every 3D point into view #i:
            P = intrinsic @ poses[i]       # (3×4) projection matrix
            H = (P @ X_hom.T).T            # shape = (num_points, 3); row = [x_cam* , y_cam* , z_cam]
            cam_z = H[:, 2].copy()         # camera‐space depth of each 3D point
            projected_depths[i] = cam_z

            # Avoid division by zero (z_cam <= 0 means “behind camera” or invalid):
            valid = cam_z > 1e-6

            # Compute pixel coords (u, v) = (x_cam / z_cam, y_cam / z_cam), then round→int
            uv = np.zeros((n_points, 2), dtype=np.int32)
            uv[valid, 0] = np.round(H[valid, 0] / cam_z[valid]).astype(np.int32)
            uv[valid, 1] = np.round(H[valid, 1] / cam_z[valid]).astype(np.int32)
            projected_uv[i] = uv

            # 2) Determine which of these projected points are actually “visible”
            depth_image = imageio.imread(os.path.join(depths_path, f"{idx}.png")) / depth_scale
            # Create a mask of points falling inside the image bounds:
            inside = (
                (uv[:, 0] >= 0) & (uv[:, 0] < width) &
                (uv[:, 1] >= 0) & (uv[:, 1] < height) &
                valid
            )

            # For each “inside” point, compare cam_z[p] vs. depth_image[v,u]
            # If |cam_z[p] − depth_image[v,u]| ≤ vis_threshold, then point is visible.
            cam_measured = np.zeros_like(cam_z)
            cam_measured[inside] = depth_image[uv[inside, 1], uv[inside, 0]]
            depth_diff = np.abs(cam_z[inside] - cam_measured[inside])
            visible = depth_diff <= vis_threshold

            # Mark only those indices as visible:
            inside_indices = np.where(inside)[0]
            visible_points_view[i, inside_indices[visible]] = True

        return visible_points_view, projected_uv, projected_depths, resolution


    def get_visible_points_in_view_in_mask(self):
        """
        Builds a tiny H×W Boolean “raster” for each (view, mask) pair, marking exactly
        where each mask’s VISIBLE 3D points appear in pixel‐space.

        Returns:
            visible_points_in_view_in_mask: np.bool array [num_views, num_masks, H, W]
        """
        num_views = len(self.indices)
        num_masks = self.masks.num_masks
        height, width = self.resolution

        vpivim = np.zeros((num_views, num_masks, height, width), dtype=bool)

        print("[INFO] Computing visible‐points‐in‐each‐mask for each view.")
        for i in range(num_views):
            for m in range(num_masks):
                # Which 3D points belong to mask #m?
                mask3d = self.masks.masks[:, m].astype(bool)
                # Of those, which are visible in view i?
                vis3d  = self.visible_points_view[i]
                both   = mask3d & vis3d
                if not both.any():
                    # no visible 3D points of this mask in view i → stays all‐False
                    continue

                pts_uv = self.projected_uv[i][both]  # shape = (k, 2)
                # Clip to ensure we stay in [0, width−1], [0, height−1]
                us = pts_uv[:, 0].clip(0, width - 1)
                vs = pts_uv[:, 1].clip(0, height - 1)
                vpivim[i, m, vs, us] = True

        return vpivim


    def get_top_k_indices_per_mask(self, k: int) -> np.ndarray:
        """
        Return the top‐k views (by number of visible points) for each mask.
        num_points_in_view_in_mask[i, m] = (# of pixels in visible_points_in_view_in_mask[i, m, :, :])
        """
        counts = self.visible_points_in_view_in_mask.sum(axis=2).sum(axis=2)  
        # counts.shape = (num_views, num_masks)
        # We want, for each mask m, the top‐k view‐indices (in descending order).
        topk = np.argsort(-counts, axis=0)[:k, :].T  # final shape = (num_masks, k)
        return topk


    def find_occluding_masks_for_target(self, target_mask_idx: int, view_idx: int) -> list:
        """
        Given a target mask index and a view index, return a list of OTHER mask‐indices
        that *truly* occlude the target (i.e. lie in front of it) in that view.

        Method:
          - Iterate over every 3D point in the target mask.
          - Project it into (u, v) and look up the sensor depth at (u, v).
          - If target_point_depth_cam > sensor_depth_at_uv + vis_threshold, that point is occluded.
          - We then scan other masks’ 3D points that project near (u, v) to see which mask is strictly
            in front (camera‐space depth smaller by at least vis_threshold). Any such mask is added
            to the occluder list.

        Returns:
          List of integer mask‐indices that truly occlude the target in view #view_idx.
        """
        # 1) which 3D indices belong to the target mask?
        target_pts = np.where(self.masks.masks[:, target_mask_idx] > 0)[0]
        if target_pts.size == 0:
            return []

        # 2) For this view, grab (u,v) and camera‐space depth for *all* points:
        uv_all    = self.projected_uv[view_idx]      # shape = (num_points, 2)
        depth_all = self.projected_depths[view_idx]  # shape = (num_points,)

        # 3) Load the sensor depth image for this view:
        original_idx = self.indices[view_idx]
        depth_path   = os.path.join(self.camera.depths_path, f"{original_idx}.png")
        try:
            sensor_depth_img = imageio.imread(depth_path) / self.camera.depth_scale
        except FileNotFoundError:
            # cannot load depth → no occluders
            return []

        height, width = sensor_depth_img.shape

        occluding_masks = set()

        for pid in target_pts:
            u, v = uv_all[pid]
            if not (0 <= u < width and 0 <= v < height):
                # this point fell outside the image → skip
                continue

            pt_cam_z       = depth_all[pid]
            sensor_depth_uv = sensor_depth_img[v, u]

            # If the projected 3D point sits behind the sensor reading by > vis_threshold,
            # it must be occluded by something else.
            if pt_cam_z > sensor_depth_uv + self.vis_threshold:
                # find which other mask has a 3D point in front at this pixel (within ±2 px)
                for other_mask_idx in range(self.masks.num_masks):
                    if other_mask_idx == target_mask_idx:
                        continue

                    # gather all 3D points of “other_mask_idx”
                    other_pts = np.where(self.masks.masks[:, other_mask_idx] > 0)[0]
                    for oid in other_pts:
                        ou, ov = uv_all[oid]
                        if not (0 <= ou < width and 0 <= ov < height):
                            continue

                        # if that “other” 3D‐point projects within 2 pixels of (u,v)
                        if abs(ou - u) <= 2 and abs(ov - v) <= 2:
                            o_cam_z = depth_all[oid]
                            # if that other point really is in front of the target point
                            if o_cam_z < pt_cam_z - self.vis_threshold:
                                occluding_masks.add(other_mask_idx)
                                # no need to check more points of this same mask,
                                # move on to the next mask
                                break
                    if other_mask_idx in occluding_masks:
                        break

        return list(occluding_masks)


class FeaturesExtractor:
    def __init__(self,
                 camera: Camera,
                 clip_model: str,
                 images: Images,
                 masks: InstanceMasks3D,
                 pointcloud: PointCloud,
                 sam_model_type: str,
                 sam_checkpoint: str,
                 vis_threshold: float,
                 device: torch.device,
                 model_type: str = "clip",
                 inpainting: bool = True,
                 seed = None):
        if seed is not None:
            set_global_seeds(seed)

        self.camera = camera
        self.images = images
        self.device = device

        # Build PointProjector once (precomputes all projections, depths, and per‐view/per‐mask rasters)
        self.point_projector = PointProjector(
            camera        = camera,
            point_cloud   = pointcloud,
            masks         = masks,
            vis_threshold = vis_threshold,
            indices       = images.indices
        )

        # Initialize SAM and CLIP
        self.predictor_sam    = initialize_sam_model(device, sam_model_type, sam_checkpoint)

        
        # 3) Decide which “embedding model” to use:
        assert model_type in ("clip", "eva", "blip"), "Must choose 'clip', 'eva', or 'blip'."
        
        print(f"[INFO] Using {model_type.upper()} model for feature extraction.")
        
        self.model_type = model_type

        if model_type == "clip":
            # CLIP ViT-L/14@336
            import clip
            # note the "openai/clip-vit-large-patch14-336" identifier
            self.clip_model, self.clip_preprocess = clip.load(
                "ViT-L/14@336px",  # or use "openai/clip-vit-large-patch14-336"
                device
            )
            self.feature_dim = self.clip_model.visual.output_dim  # usually 768
        elif model_type == "eva":
            # now that eva_clip is installed, this will work:
            from eva_clip import create_model_and_transforms

            # pick whichever EVA-CLIP variant you want (e.g. EVA02-CLIP-L-14-336)
            checkpoint_name = "EVA02-CLIP-L-14-336"  
            model, _, preprocess = create_model_and_transforms(
                model_name=checkpoint_name,
                pretrained="eva_clip",            # or the path to the local .pt
                force_custom_clip=True
            )
            self.eva_model      = model.to(device).eval()
            self.eva_preprocess = preprocess
            self.feature_dim = self.eva_vision.config.hidden_size  # 1024 for ViT-L/14

        elif model_type == "blip":
            # Use BLIP’s “image-captioning-base" checkpoint for vision features
            self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")  # :contentReference[oaicite:1]{index=1}
            self.blip_model     = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
            self.blip_model.eval()
            self.feature_dim = self.blip_model.config.vision_config.hidden_size

        # LaMa inpainting
        self.inpainting = inpainting
        if inpainting:
            self.inpainting_model = SimpleLama(device=device)

        


    def extract_features(self,
                         topk: int,
                         multi_level_expansion_ratio: float,
                         num_levels: int,
                         num_random_rounds: int,
                         num_selected_points: int,
                         save_crops: bool,
                         out_folder: str,
                         optimize_gpu_usage: bool = False):
        """
        Significantly faster loop by iterating *only* over the union of top‐k views:
        
          1) Compute topk_indices_per_mask = self.point_projector.get_top_k_indices_per_mask(topk).
             This is shape (num_masks, topk).  Each row i is “the top‐k views where mask_i
             has the most visible 3D points.”
          
          2) Build view_to_masks: a dict mapping each view_idx → [list of mask_ids that chose it].
             The *union* of all top‐k views (across masks) is the only set of views we will visit.

          3) For each view in that union (in ascending numerical order, or any order):
               a) Call sam.set_image(image_np) exactly once.
               b) Loop over the small list of mask_ids = view_to_masks[view].
                  For each mask:
                    i)  Run run_sam on just that mask’s 2D points to get best_mask.
                    ii) Build the inpainting mask (using remove_occluding_objects_keep_target).
                    iii) If needed, do LaMa inpainting once for this view. Otherwise, skip.
                    iv)  From 'best_mask' and the (possibly inpainted) image, produce multi‐level crops.
                    v)   Preprocess those crops for CLIP, append to crops_by_mask[mask_id].

          4) After finishing all views, each mask_id will have a list of 0–(topk × num_levels) crops in
             crops_by_mask[mask_id].  We then batch‐encode each mask’s crops in a single CLIP forward pass,
             average to a 768-D vector, and store in mask_clip[mask_id].

        Returns:
            mask_clip: np.ndarray of shape (num_masks, 768)
        """

        num_masks   = self.point_projector.masks.num_masks

        # (1) Get top‐k views for each mask (shape = [num_masks, topk])
        topk_indices_per_mask = self.point_projector.get_top_k_indices_per_mask(topk)

        # (2) Build a mapping: view_idx → list of mask_ids that included that view in their top‐k
        view_to_masks = {}
        for mask_id in range(num_masks):
            for view_idx in topk_indices_per_mask[mask_id]:
                view_to_masks.setdefault(view_idx, []).append(mask_id)

        # Now 'view_to_masks' keys = the union of all views any mask wanted. Suppose that size is V_eff.
        # In practice, V_eff << total # of frames. 
        effective_views = sorted(view_to_masks.keys())

        # (A) Prepare the output array
        mask_clip = np.zeros((num_masks, self.feature_dim), dtype=np.float32)

        # (B) Prepare a per‐mask list of crop‐tensors
        crops_by_mask = {m: [] for m in range(num_masks)}

        # (C) If requested, create a folder for saving intermediate crops/masks
        if save_crops:
            crops_folder = os.path.join(out_folder, "crops")
            os.makedirs(crops_folder, exist_ok=True)
        else:
            crops_folder = None

        # (D) Optionally, keep CLIP on CPU until batching (to save GPU RAM)
        if optimize_gpu_usage:
            self.clip_model.to(torch.device("cpu"))

        # Pre‐load all images into memory as NumPy arrays (so we index by view_idx directly)
        np_images = self.images.get_as_np_list()
        # We also need to reference visible_points_in_view_in_mask by the same index ordering:
        #   visible_points_in_view_in_mask[vi] corresponds to view_idx = images.indices[vi].
        # We already have that aligned in PointProjector.

        # ------------------------------------------------------
        # 3) Iterate over each “effective” view
        # ------------------------------------------------------
        for view_idx in tqdm(effective_views, desc="Looping over effective views"):
            # 3.a) Load the RGB image (NumPy) for this view
            image_np = np_images[view_idx]  # shape = (H, W, 3)

            # 3.b) Let SAM work on this image exactly once
            self.predictor_sam.set_image(image_np)

            # 3.c) Find the “internal index” vi such that images.indices[vi] == view_idx
            #     Because PointProjector stored everything in the same order as images.indices,
            #     we need that vi to look up visible_points_in_view_in_mask[vi].
            if isinstance(self.images.indices, np.ndarray):
                vi_arr = np.where(self.images.indices == view_idx)[0]
                if vi_arr.size == 0:
                    continue  # view_idx not found
                vi = int(vi_arr[0])
            else:
                vi = self.images.indices.index(view_idx)

            # 3.d) Now loop only over the masks that explicitly chose this view (in top‐k)
            for mask_id in view_to_masks[view_idx]:
                # (i) Get the 2D coords (v,u) where mask_id is visible in view vi
                vis2d = self.point_projector.visible_points_in_view_in_mask[vi][mask_id]
                # Just sanity check (it should be non‐empty because mask_id chose this view in top‐k):
                coords_2d = np.transpose(np.where(vis2d))
                if coords_2d.shape[0] == 0:
                    # This would be surprising, but if it happens, skip.
                    continue

                # (ii) Run SAM on this mask's 2D points to get best_mask (H×W boolean array)
                best_mask = run_sam(
                    image_size          = image_np,
                    num_random_rounds   = num_random_rounds,
                    num_selected_points = num_selected_points,
                    point_coords        = coords_2d,
                    predictor_sam       = self.predictor_sam,
                )

                # --- Visualization 1: original image with SAM mask overlaid ---
                dbg_dir = os.path.join("debug_masks", f"mask{mask_id}_view{view_idx}")
                os.makedirs(dbg_dir, exist_ok=True)
                overlay_sam = image_np.copy()
                if overlay_sam.ndim == 3:
                    overlay_sam[best_mask > 0, 1] = 255  # green channel for SAM mask
                else:
                    overlay_sam = cv2.cvtColor(overlay_sam, cv2.COLOR_GRAY2RGB)
                    overlay_sam[best_mask > 0, 1] = 255
                imageio.imwrite(os.path.join(dbg_dir, "sam_mask_overlay.png"), overlay_sam)

                if self.inpainting:
                    # (iii) Build the inpainting mask for this (mask_id, view_idx)
                    inpaint_mask = self.remove_occluding_objects_keep_target(
                        mask_idx             = mask_id,
                        view_idx             = view_idx,
                        dilation_kernel_size = 15,
                        erosion_kernel_size  = 5,
                        use_sam_dense_masks  = True
                    )

                    # --- Visualization 2: inpainting mask overlaid on original image ---
                    overlay_inpaint = image_np.copy()
                    if overlay_inpaint.ndim == 3:
                        overlay_inpaint[inpaint_mask > 0, 0] = 255  # red channel for inpainting mask
                    else:
                        overlay_inpaint = cv2.cvtColor(overlay_inpaint, cv2.COLOR_GRAY2RGB)
                        overlay_inpaint[inpaint_mask > 0, 0] = 255
                    imageio.imwrite(os.path.join(dbg_dir, "inpainting_mask_overlay.png"), overlay_inpaint)

                    # (iv) If at least one pixel must be inpainted, run LaMa once for this view
                    if inpaint_mask.any():
                        inpainted = self.inpainting_model(image_np, inpaint_mask)
                        if isinstance(inpainted, np.ndarray) and inpainted.dtype in (np.float32, np.float64):
                            inpainted = (inpainted * 255).astype(np.uint8)
                        elif not isinstance(inpainted, np.ndarray):
                            inpainted = np.array(inpainted)
                        image_for_crop = inpainted
                        # --- Visualization 3: save the inpainted image ---
                        imageio.imwrite(os.path.join(dbg_dir, "inpainted_image.png"), image_for_crop)
                    else:
                        # No occluders → use original image for cropping
                        image_for_crop = image_np
                    # (v) Optionally save the inpainting mask (debug)
                    if crops_folder:
                        mask_outpath = os.path.join(
                            crops_folder, f"mask{mask_id}_view{view_idx}_inpaint.png"
                        )
                        imageio.imwrite(mask_outpath, (inpaint_mask * 255).astype(np.uint8))
                else:
                    image_for_crop = image_np  # Use the original image for cropping, no inpainting

                # (vi) Use 'best_mask' to produce multi‐level crops from image_for_crop
                for lvl in range(num_levels):
                    x1, y1, x2, y2 = mask2box_multi_level(
                        torch.from_numpy(best_mask),
                        lvl,
                        multi_level_expansion_ratio
                    )
                    cropped_pil = Image.fromarray(image_for_crop).crop((x1, y1, x2, y2))

                    if crops_folder:
                        crop_outpath = os.path.join(
                            crops_folder, f"crop_m{mask_id}_v{view_idx}_l{lvl}.png"
                        )
                        cropped_pil.save(crop_outpath)

                    crops_by_mask[mask_id].append(cropped_pil)

        # ------------------------------------------------------
        # 4) Batch‐encode each mask’s collected crops with CLIP
        # ------------------------------------------------------
        for mask_id in range(num_masks):
            crop_list = crops_by_mask[mask_id]
            if len(crop_list) == 0:
                # No crops were generated (e.g. mask never in any top‐k view), leave zeros
                continue
            if self.model_type == "clip":
                # -----------------------
                # (A) CLIP path (unchanged)
                # -----------------------
                if optimize_gpu_usage:
                    self.clip_model.to(self.device)

                batch_t = torch.stack([ self.clip_preprocess(crop) for crop in crop_list ], dim=0).to(self.device)
                with torch.no_grad():
                    feats = self.clip_model.encode_image(batch_t).float()
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                mask_clip[mask_id] = feats.mean(dim=0).cpu().numpy()

                if optimize_gpu_usage:
                    self.clip_model.to(torch.device("cpu"))

            elif self.model_type == "eva":
                raise NotImplementedError(
                    "EVA-CLIP support is not implemented yet in this version of FeaturesExtractor."
                )
                #  # -------------------------------------------------
                # # (B) EVA-CLIP encode_image + FP16 normalization
                # # -------------------------------------------------
                # # 1) `self.eva_preprocess` is identical to CLIP’s 336px transforms.
                # inputs = torch.stack(
                #     [ self.eva_preprocess(crop) for crop in crop_list ],
                #     dim=0
                # ).to(self.device).half()

                # with torch.no_grad():
                #     image_features = self.eva_vision.encode_image(inputs)  # returns (B, 768) in FP16
                #     image_features = image_features.float()  # cast to FP32 for more stable norm
                #     image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                # mask_clip[mask_id] = image_features.mean(dim=0).cpu().numpy()

            elif self.model_type == "blip":
                # -----------------------
                # (C) BLIP-Large path
                # -----------------------
                if optimize_gpu_usage:
                    self.blip_model.to(self.device)

                # The BLIP “vision_model” expects a `pixel_values` key in inputs:
                inputs = self.blip_processor(
                    images = crop_list,
                    padding="max_length",
                    return_tensors = "pt"
                ).to(self.device)
                # pixel_values = inputs["pixel_values"].to(self.device)

                # with torch.no_grad():
                #     image_outputs = self.blip_model.vision_model(**inputs)
                #     print(f"[INFO] BLIP model output keys: {image_outputs.keys()}")
                #     # “pooler_output” is the [CLS] embedding, shape (N_crops, 768)
                #     feats = image_outputs.pooler_output
                #     print(f"[INFO] BLIP features shape: {feats.shape}")
                #     print(f"[INFO] BLIP features content: {feats[:5]}")
                #     feats = feats / feats.norm(dim=-1, keepdim=True)

                
                with torch.no_grad():
                    # This method does: vision_model → projection → normalize
                    image_features = self.blip_model.get_image_features(inputs["pixel_values"])  
                    # shape = (num_crops, D_blip), where D_blip = 768 for BLIP-base
                    # Normalize (already unit-norm in most BLIP implementations, but re-normalize to be safe):
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                    # Average & re-normalize exactly as SigLIP does:
                    mean_feat = image_features.mean(dim=0)   # (D_blip,)
                    feats = mean_feat / mean_feat.norm() # final unit-length vector

                mask_clip[mask_id] = feats.cpu().numpy()

                # if optimize_gpu_usage:
                #     self.blip_model.to(torch.device("cpu"))
                # # Move CLIP to GPU if necessary
                # if optimize_gpu_usage:
                #     self.clip_model.to(self.device)

                # batch_tensor = torch.stack(crop_list, dim=0).to(self.device)
                # with torch.no_grad():
                #     feats = self.clip_model.encode_image(batch_tensor).float()
                #     feats = feats / feats.norm(dim=-1, keepdim=True)

                # mask_clip[mask_id] = feats.mean(dim=0).cpu().numpy()

                # # Return CLIP to CPU if optimizing VRAM
                # if optimize_gpu_usage:
                #     self.clip_model.to(torch.device("cpu"))

        return mask_clip


    # def extract_features(
    #     self,
    #     topk: int,
    #     multi_level_expansion_ratio: float,
    #     num_levels: int,
    #     num_random_rounds: int,
    #     num_selected_points: int,
    #     save_crops: bool,
    #     out_folder: str,
    #     optimize_gpu_usage: bool = False
    # ):
    #     """
    #     For each 3D instance mask, pick its top‐k best views, inpaint the occluders, 
    #     crop multi‐scale windows around the target, and encode them with CLIP.

    #     Returns:
    #         mask_clip:   (num_masks, 768) array of normalized CLIP features (mean over all crops).
    #     """

    #     num_masks = self.point_projector.masks.num_masks

    #     if save_crops:
    #         crops_folder = os.path.join(out_folder, "crops")
    #         os.makedirs(crops_folder, exist_ok=True)
    #     else:
    #         crops_folder = None

    #     # 1) Precompute, for each mask, the top‐k view‐indices by visibility count
    #     topk_indices_per_mask = self.point_projector.get_top_k_indices_per_mask(topk)
    #     np_images = self.images.get_as_np_list()  # list of H×W×3 numpy arrays

    #     # 2) Prepare output array
    #     mask_clip = np.zeros((num_masks, 768), dtype=np.float32)

    #     # 3) Optionally keep CLIP on CPU until the last moment, to save VRAM
    #     if optimize_gpu_usage:
    #         self.clip_model.to(torch.device("cpu"))

    #     # 4) Loop over each mask
    #     for mask_id in tqdm(range(num_masks), desc="Extracting features for each mask"):
    #         # We'll gather all crops for this mask across its top‐k views, then batch‐encode in one CLIP call.
    #         images_crops = []

    #         # If optimize_gpu_usage, we want SAM on GPU inside this mask‐loop, and CLIP on CPU. 
    #         if optimize_gpu_usage:
    #             self.predictor_sam.model.cuda()

    #         # 5) For each of the top‐k views of this mask:
    #         for view in topk_indices_per_mask[mask_id]:
    #             # 5.1) Quickly check if the mask even projects ANYWHERE in this view:
    #             vis2d = self.point_projector.visible_points_in_view_in_mask[view][mask_id]
    #             if not vis2d.any():
    #                 # No visible 2D points → skip entire view
    #                 continue

    #             # 5.2) Convert image to numpy (if not already)
    #             image_np = np_images[view]

    #             # 5.3) Initialize SAM once per (mask,view) so we can reuse it for (a) best_mask, (b) occluders
    #             self.predictor_sam.set_image(image_np)

    #             # 5.4) Run SAM on the target‐mask points to get best_mask (for cropping)
    #             target_coords = np.transpose(np.where(vis2d))
    #             best_mask = run_sam(
    #                 image_size          = image_np,
    #                 num_random_rounds   = num_random_rounds,
    #                 num_selected_points = num_selected_points,
    #                 point_coords        = target_coords,
    #                 predictor_sam       = self.predictor_sam
    #             )

    #             # 5.5) Generate the occlusion‐based inpainting mask. 
    #             # If no occluders exist, this will be all zeros and we can skip inpainting.
    #             inpainting_mask = self.remove_occluding_objects_keep_target(
    #                 mask_idx             = mask_id,
    #                 view_idx             = view,
    #                 dilation_kernel_size = 15,
    #                 erosion_kernel_size  = 5,
    #                 use_sam_dense_masks  = True
    #             )

    #             # 5.6) If there is at least one occluder pixel, run inpainting; else skip it
    #             if inpainting_mask.any():
    #                 # Run LaMa inpainting once
    #                 result = self.inpainting_model(image_np, inpainting_mask)
    #                 if isinstance(result, np.ndarray) and result.dtype in (np.float32, np.float64):
    #                     result = (result * 255).astype(np.uint8)
    #                 elif not isinstance(result, np.ndarray): 
    #                     result = np.array(result)
    #                 image_for_crops = result
    #             else:
    #                 # No occluder → just use the original image
    #                 image_for_crops = image_np

    #             # 5.7) Save the inpainting mask to disk (if requested)
    #             if crops_folder:
    #                 mask_path = os.path.join(crops_folder, f"mask{mask_id}_{view}_mask.png")
    #                 imageio.imwrite(mask_path, (inpainting_mask * 255).astype(np.uint8))

    #             # 5.8) From best_mask (binary H×W), compute multi‐scale crops & collect them
    #             for level in range(num_levels):
    #                 x1, y1, x2, y2 = mask2box_multi_level(
    #                     torch.from_numpy(best_mask), 
    #                     level, 
    #                     multi_level_expansion_ratio
    #                 )
    #                 # Crop from either inpainted or original image
    #                 cropped_img = Image.fromarray(image_for_crops).crop((x1, y1, x2, y2))

    #                 if crops_folder:
    #                     cropped_img.save(os.path.join(crops_folder, f"crop{mask_id}_{view}_{level}.png"))

    #                 # Preprocess for CLIP
    #                 crop_tensor = self.clip_preprocess(cropped_img)
    #                 images_crops.append(crop_tensor)

    #             # 5.9) End of per‐view processing



    #         # 6) After looping over all top‐k views for this mask, we have a list of 0–(topk×num_levels) crops.
    #         #    If we actually collected anything, run them through CLIP in one batch.
    #         if images_crops:
    #             # If optimize_gpu_usage, move CLIP back to GPU now
    #             if optimize_gpu_usage:
    #                 self.clip_model.to(self.device)

    #             batch = torch.stack(images_crops, dim=0).to(self.device)
    #             with torch.no_grad():
    #                 feats = self.clip_model.encode_image(batch).float()
    #                 feats = feats / feats.norm(dim=-1, keepdim=True)
    #             # Average over all crops → one 768‐dim vector per mask
    #             mask_clip[mask_id] = feats.mean(dim=0).cpu().numpy()

    #             # Return CLIP to CPU if we’re saving VRAM
    #             if optimize_gpu_usage:
    #                 self.clip_model.to(torch.device("cpu"))

    #         # 7) After finishing one mask, move SAM back to CPU if desired
    #         if optimize_gpu_usage:
    #             self.predictor_sam.model.cpu()

    #     return mask_clip
    
    def remove_occluding_objects_keep_target(
        self,
        mask_idx: int,
        view_idx: int,
        dilation_kernel_size: int = 15,
        erosion_kernel_size: int = 5,
        use_sam_dense_masks: bool = True
    ) -> np.ndarray:
        """
        Build a binary “inpainting mask” that covers only the true occluders
        (in front of `mask_idx`) in view #view_idx.  Runs in roughly
        O(N_target_pts + Num_occluded_pixels×Num_masks) time.

        Returns an H×W uint8 array (0 or 1).  You can cast to bool if you prefer.
        """

        # --------------------------------------------------
        # 1) Load the RGB image and get its dimensions
        # --------------------------------------------------
        image_pil = self.images.images[view_idx]
        image_np  = np.array(image_pil)
        H, W      = image_np.shape[:2]

        # --------------------------------------------------
        # 2) Initialize the “final inpainting mask” to zeros
        # --------------------------------------------------
        final_mask = np.zeros((H, W), dtype=np.uint8)

        # --------------------------------------------------
        # 3) Build a debug image for the target mask itself
        # --------------------------------------------------
        #    We want to save an image that shows exactly which pixels
        #    (u,v) in this view belong to the target object (mask_idx).
        target_sparse = np.zeros((H, W), dtype=np.uint8)
        # The visible 2D coords of the target mask in this view:
        coords_target = np.transpose(
            np.where(self.point_projector.visible_points_in_view_in_mask[view_idx][mask_idx])
        )
        if coords_target.shape[0] > 0:
            ys_t = coords_target[:, 0].clip(0, H - 1)
            xs_t = coords_target[:, 1].clip(0, W - 1)
            target_sparse[ys_t, xs_t] = 255  # mark target pixels in white

        # --------------------------------------------------
        # 4) VECTORIZE “TRUE OCCLUDER” DETECTION
        # --------------------------------------------------

        # 4.1) All 3D‐point indices that belong to the target mask
        target_pts = np.where(self.point_projector.masks.masks[:, mask_idx] > 0)[0]
        if target_pts.size == 0:
            # If the target mask has no 3D points at all, nothing to occlude.
            # Save debug images and return all‐zero
            dbg_dir = os.path.join("debug_masks", f"mask{mask_idx}_view{view_idx}")
            os.makedirs(dbg_dir, exist_ok=True)
            imageio.imwrite(os.path.join(dbg_dir, "00_original.png"), image_np)
            imageio.imwrite(os.path.join(dbg_dir, "01_target_mask_sparse.png"), target_sparse)
            return final_mask

        # 4.2) Grab their projected (u,v) and camera‐space depth for this view
        if self.point_projector.projected_uv is None or self.point_projector.projected_depths is None:
            raise ValueError("PointProjector must have projected_uv and projected_depths initialized.")
        uv_all    = self.point_projector.projected_uv[view_idx]      # shape = (num_points, 2)
        depth_all = self.point_projector.projected_depths[view_idx]  # shape = (num_points,)

        # Extract only the target’s 3D points, then split into u,v arrays
        uv_target = uv_all[target_pts]       # shape = (N_t, 2)
        d_target  = depth_all[target_pts]    # shape = (N_t,)

        # Round (u,v) to integers and filter in‐bounds
        u = np.round(uv_target[:, 0]).astype(np.int32)
        v = np.round(uv_target[:, 1]).astype(np.int32)
        valid_mask = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if not valid_mask.all():
            u = u[valid_mask]
            v = v[valid_mask]
            d_target = d_target[valid_mask]

        if u.size == 0:
            # All projected points fell outside → nothing in‐frame to occlude
            dbg_dir = os.path.join("debug_masks", f"mask{mask_idx}_view{view_idx}")
            os.makedirs(dbg_dir, exist_ok=True)
            imageio.imwrite(os.path.join(dbg_dir, "00_original.png"), image_np)
            imageio.imwrite(os.path.join(dbg_dir, "01_target_mask_sparse.png"), target_sparse)
            return final_mask

        # 4.3) Compute a “min depth per pixel” for the target mask
        pixel_ids = v * W + u  # flatten (v,u) → unique pixel index
        unique_pixels, inverse_indices = np.unique(pixel_ids, return_inverse=True)
        min_depth_per_pixel = np.full(unique_pixels.shape, np.inf, dtype=np.float32)

        for i_pt, pix_idx in enumerate(inverse_indices):
            dval = d_target[i_pt]
            if dval < min_depth_per_pixel[pix_idx]:
                min_depth_per_pixel[pix_idx] = dval

        ups = unique_pixels % W
        vps = unique_pixels // W

        # 4.4) Load the sensor depth image for this view and compare
        depth_path = os.path.join(self.camera.depths_path, f"{self.point_projector.indices[view_idx]}.png")
        try:
            sensor_depth = imageio.imread(depth_path) / self.camera.depth_scale
        except FileNotFoundError:
            # If no depth available, save debug and return
            dbg_dir = os.path.join("debug_masks", f"mask{mask_idx}_view{view_idx}")
            os.makedirs(dbg_dir, exist_ok=True)
            imageio.imwrite(os.path.join(dbg_dir, "00_original.png"), image_np)
            imageio.imwrite(os.path.join(dbg_dir, "01_target_mask_sparse.png"), target_sparse)
            return final_mask

        sens_vals = sensor_depth[vps, ups]
        occluded_mask = min_depth_per_pixel > (sens_vals + self.point_projector.vis_threshold)
        if not occluded_mask.any():
            # Target is never behind anything → no occluders
            dbg_dir = os.path.join("debug_masks", f"mask{mask_idx}_view{view_idx}")
            os.makedirs(dbg_dir, exist_ok=True)
            imageio.imwrite(os.path.join(dbg_dir, "00_original.png"), image_np)
            imageio.imwrite(os.path.join(dbg_dir, "01_target_mask_sparse.png"), target_sparse)
            return final_mask

        occluded_pixels = unique_pixels[occluded_mask]
        oc_u = occluded_pixels % W
        oc_v = occluded_pixels // W

        # 4.5) Which masks occupy any of those occluded pixels?
        vis_matrix = self.point_projector.visible_points_in_view_in_mask[view_idx]  # (M, H, W)
        sub_vis = vis_matrix[:, oc_v, oc_u]  # shape = (num_masks, #occluded_pixels)
        occluder_bool = np.any(sub_vis, axis=1)
        occluder_bool[mask_idx] = False
        true_occluders = np.nonzero(occluder_bool)[0]

        if true_occluders.size == 0:
            # No actual occluders, just save debug and return
            dbg_dir = os.path.join("debug_masks", f"mask{mask_idx}_view{view_idx}")
            os.makedirs(dbg_dir, exist_ok=True)
            imageio.imwrite(os.path.join(dbg_dir, "00_original.png"), image_np)
            imageio.imwrite(os.path.join(dbg_dir, "01_target_mask_sparse.png"), target_sparse)
            return final_mask

        # --------------------------------------------------
        # 5) BUILD “FINAL_MASK” VIA SAM (or sparse fallback)
        # --------------------------------------------------

        # (A) For debug: save a grayscale “all sparse occluding points” image
        all_sparse = np.zeros((H, W), dtype=np.uint8)

        # (B) If we want SAM densification, set the image once
        if use_sam_dense_masks:
            self.predictor_sam.set_image(image_np)

        for other_m in true_occluders:
            coords = np.transpose(
                np.where(self.point_projector.visible_points_in_view_in_mask[view_idx][other_m])
            )  # shape = (N_pts_for_mask, 2)
            if coords.shape[0] == 0:
                continue

            ys = coords[:, 0].clip(0, H - 1)
            xs = coords[:, 1].clip(0, W - 1)
            all_sparse[ys, xs] = 255

            # Try SAM if requested
            if use_sam_dense_masks:
                try:
                    dense = run_sam(
                        image_size         = image_np,
                        num_random_rounds   = 10,
                        num_selected_points = 5,
                        point_coords        = coords,
                        predictor_sam       = self.predictor_sam,
                    )
                    if dense is not None:
                        final_mask |= dense.astype(np.uint8)
                        continue  # skip sparse fallback
                except Exception:
                    pass  # on any SAM failure, fall back to sparse points

            # Fallback: paint only the sparse points of this mask
            sparse_only = np.zeros((H, W), dtype=np.uint8)
            sparse_only[ys, xs] = 1
            final_mask |= sparse_only

        # --------------------------------------------------
        # 6) SAVE DEBUG IMAGES (INCLUDING “TARGET MASK”)
        # --------------------------------------------------
        dbg_dir = os.path.join("debug_masks", f"mask{mask_idx}_view{view_idx}")
        os.makedirs(dbg_dir, exist_ok=True)

        # 6.1) Original image
        imageio.imwrite(os.path.join(dbg_dir, "00_original.png"), image_np)

        # 6.2) Target mask sparse points (white on black)
        imageio.imwrite(os.path.join(dbg_dir, "01_target_mask_sparse.png"), target_sparse)

        # 6.3) All occluding‐masks’ sparse points (white on black)
        imageio.imwrite(os.path.join(dbg_dir, "02_all_occluding_sparse.png"), all_sparse)

        # 6.4) Raw occluder‐mask union before morphology (white on black)
        imageio.imwrite(os.path.join(dbg_dir, "03_raw_occluders_before_morph.png"), final_mask * 255)

        # --------------------------------------------------
        # 7) MORPHOLOGICAL CLEANUP: DILATE → ERODE → CLOSING
        # --------------------------------------------------
        if dilation_kernel_size > 0:
            kern = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (dilation_kernel_size, dilation_kernel_size)
            )
            final_mask = cv2.dilate(final_mask, kern, iterations=1)

        if erosion_kernel_size > 0:
            kern = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (erosion_kernel_size, erosion_kernel_size)
            )
            final_mask = cv2.erode(final_mask, kern, iterations=1)

        # Final 3×3 closing to fill small holes
        close_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, close_kern)

        # --------------------------------------------------
        # 8) SAVE FINAL MASK AND OVERLAY
        # --------------------------------------------------
        imageio.imwrite(
            os.path.join(dbg_dir, "04_final_inpainting_mask.png"),
            final_mask * 255
        )

        # Overlay the final_mask in red onto the original image (for easy visual check)
        overlay = image_np.copy()
        if overlay.ndim == 3:
            overlay[final_mask > 0, 0] = 255  # paint red channel where mask=1
        else:
            overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2RGB)
            overlay[final_mask > 0, 0] = 255
        imageio.imwrite(os.path.join(dbg_dir, "05_overlay.png"), overlay)

        return final_mask