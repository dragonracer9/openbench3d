#!/usr/bin/env python3
"""
Fixed SigLIP Feature Extractor for OpenMask3D

This implementation uses the proper SigLIP workflow:
- Image + Text -> Model -> Logits -> Sigmoid -> Probabilities

Instead of the incorrect CLIP-style approach:
- Image -> Features, Text -> Features -> Cosine Similarity
"""

import torch
import numpy as np
import os
from transformers import SiglipProcessor, SiglipModel
from tqdm import tqdm
from openmask3d.data.load import Camera, InstanceMasks3D, Images, PointCloud
from openmask3d.mask_features_computation.utils import initialize_sam_model, mask2box_multi_level, run_sam
from PIL import Image
import imageio

class SigLIPDirectEvaluator:
    """
    SigLIP evaluator that uses the proper direct logits approach
    Instead of storing features, this computes similarities on-demand
    """
    
    def __init__(self, 
                 siglip_model="google/siglip-base-patch16-384",
                 device="cuda"):
        self.device = device
        self.processor = SiglipProcessor.from_pretrained(siglip_model)
        self.model = SiglipModel.from_pretrained(siglip_model).to(device)
        
    def compute_image_text_similarities(self, images, texts):
        """
        Compute similarities between images and texts using proper SigLIP workflow
        
        Args:
            images: List of PIL Images or single PIL Image
            texts: List of strings or single string
            
        Returns:
            similarities: numpy array of shape [num_images, num_texts] with probabilities
        """
        if not isinstance(images, list):
            images = [images]
        if not isinstance(texts, list):
            texts = [texts]
            
        # Store results
        all_probs = []
        
        for img in images:
            # Process image with all texts at once (more efficient)
            inputs = self.processor(images=img, text=texts, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits_per_image  # Shape: [1, num_texts]
                probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()  # Convert to probabilities
                all_probs.append(probs)
        
        return np.array(all_probs)
    
    def evaluate_mask_crops(self, crop_images, query_texts):
        """
        Evaluate a set of crop images against query texts
        
        Args:
            crop_images: List of PIL Images (crops from a mask)
            query_texts: List of query strings
            
        Returns:
            best_class_idx: Index of best matching class
            best_probability: Probability of best match
            all_probabilities: All probabilities for this mask
        """
        if len(crop_images) == 0:
            return 0, 0.0, np.zeros(len(query_texts))
        
        # Compute probabilities for all crops and all texts
        similarities = self.compute_image_text_similarities(crop_images, query_texts)
        
        # Average probabilities across all crops for this mask
        avg_probabilities = similarities.mean(axis=0)
        
        # Find best match
        best_class_idx = np.argmax(avg_probabilities)
        best_probability = avg_probabilities[best_class_idx]
        
        return best_class_idx, best_probability, avg_probabilities

class FeaturesExtractorSiglipFixed:
    """
    Fixed SigLIP feature extractor that stores crop images instead of features
    This allows proper SigLIP evaluation during inference
    """
    
    def __init__(self, 
                 camera, 
                 siglip_model, 
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
        self.masks = masks
        self.vis_threshold = vis_threshold
        
        # Initialize SAM
        self.predictor_sam = initialize_sam_model(device, sam_model_type, sam_checkpoint)
        
        # Initialize SigLIP evaluator
        self.siglip_evaluator = SigLIPDirectEvaluator(siglip_model, device)
        
        # Initialize point projector (simplified version)
        self.point_projector = self._create_point_projector(camera, pointcloud, masks, vis_threshold, images.indices)
        
    def _create_point_projector(self, camera, pointcloud, masks, vis_threshold, indices):
        """Create a simplified point projector"""
        # This is a simplified version - you may need to adapt based on your exact needs
        from .features_extractor_siglip import PointProjector
        return PointProjector(camera, pointcloud, masks, vis_threshold, indices)
    
    def extract_and_evaluate(self, 
                           query_texts,
                           topk=5, 
                           multi_level_expansion_ratio=1.2, 
                           num_levels=3, 
                           num_random_rounds=5, 
                           num_selected_points=10,
                           save_crops=False, 
                           out_folder="./output"):
        """
        Extract crops and evaluate them directly using proper SigLIP approach
        
        Returns:
            predictions: List of (class_idx, confidence) for each mask
        """
        if save_crops:
            crop_folder = os.path.join(out_folder, "crops")
            os.makedirs(crop_folder, exist_ok=True)
        
        topk_indices_per_mask = self.point_projector.get_top_k_indices_per_mask(topk)
        num_masks = self.point_projector.masks.num_masks
        np_images = self.images.get_as_np_list()
        
        predictions = []
        
        print(f"[INFO] Processing {num_masks} masks with {len(query_texts)} query texts")
        
        for mask in tqdm(range(num_masks), desc="Processing masks"):
            crop_images = []
            
            # Extract crops for this mask
            for view_count, view in enumerate(topk_indices_per_mask[mask]):
                # Get mask points in this view
                point_coords = np.transpose(np.where(
                    self.point_projector.visible_points_in_view_in_mask[view][mask] == True
                ))
                
                if point_coords.shape[0] > 0:
                    self.predictor_sam.set_image(np_images[view])
                    
                    # Run SAM to get mask
                    best_mask = run_sam(
                        image_size=np_images[view],
                        num_random_rounds=num_random_rounds,
                        num_selected_points=num_selected_points,
                        point_coords=point_coords,
                        predictor_sam=self.predictor_sam
                    )
                    
                    # Extract multi-level crops
                    for level in range(num_levels):
                        x1, y1, x2, y2 = mask2box_multi_level(
                            torch.from_numpy(best_mask), 
                            level, 
                            multi_level_expansion_ratio
                        )
                        
                        cropped_img = self.images.images[view].crop((x1, y1, x2, y2))
                        crop_images.append(cropped_img)
                        
                        if save_crops:
                            crop_path = os.path.join(crop_folder, f"mask{mask:03d}_view{view:03d}_level{level}.png")
                            cropped_img.save(crop_path)
            
            # Evaluate crops using proper SigLIP approach
            if len(crop_images) > 0:
                best_class_idx, best_probability, all_probabilities = self.siglip_evaluator.evaluate_mask_crops(
                    crop_images, query_texts
                )
                predictions.append((best_class_idx, best_probability))
                
                if mask < 5:  # Debug first few masks
                    print(f"[DEBUG] Mask {mask}: {len(crop_images)} crops -> class {best_class_idx} (prob: {best_probability:.4f})")
                    print(f"[DEBUG] All probabilities: {all_probabilities}")
            else:
                # No crops for this mask
                predictions.append((0, 0.0))
                print(f"[WARNING] No crops extracted for mask {mask}")
        
        return predictions

def test_fixed_siglip_pipeline():
    """
    Test the fixed SigLIP pipeline with simple examples
    """
    print("=== Testing Fixed SigLIP Pipeline ===")
    
    # Create test evaluator
    evaluator = SigLIPDirectEvaluator()
    
    # Create test images (same as comprehensive test)
    test_images = [
        Image.new('RGB', (224, 224), color=(255, 0, 0)),    # Red
        Image.new('RGB', (224, 224), color=(0, 255, 0)),    # Green
        Image.new('RGB', (224, 224), color=(0, 0, 255)),    # Blue
    ]
    
    # Test texts (same format as comprehensive test)
    test_texts = ["red color", "green color", "blue color"]
    
    print(f"Testing with {len(test_images)} images and {len(test_texts)} texts")
    
    # Compute similarities
    similarities = evaluator.compute_image_text_similarities(test_images, test_texts)
    
    print(f"Similarities shape: {similarities.shape}")
    print(f"Similarities range: [{similarities.min():.6f}, {similarities.max():.6f}]")
    print(f"Similarities:\n{similarities}")
    
    # Check accuracy
    predictions = np.argmax(similarities, axis=1)
    expected = np.array([0, 1, 2])  # red->0, green->1, blue->2
    accuracy = (predictions == expected).mean()
    
    print(f"Predictions: {predictions}")
    print(f"Expected: {expected}")
    print(f"Accuracy: {accuracy:.2%}")
    
    # Show individual matches
    colors = ['red', 'green', 'blue']
    for i, color in enumerate(colors):
        pred_idx = predictions[i]
        pred_text = test_texts[pred_idx]
        score = similarities[i, pred_idx]
        print(f"{color} image -> '{pred_text}' (score: {score:.6f}) {'✓' if pred_idx == i else '✗'}")
    
    return accuracy == 1.0

if __name__ == "__main__":
    # Test the fixed pipeline
    success = test_fixed_siglip_pipeline()
    if success:
        print("✅ Fixed SigLIP pipeline works correctly!")
    else:
        print("❌ Fixed SigLIP pipeline has issues")
