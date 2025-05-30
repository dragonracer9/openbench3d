"""
Unified evaluation script for closed-vocabulary 3D semantic instance segmentation.

This script automatically detects the vision-language model used and applies
the appropriate evaluation procedure.
"""

import argparse
import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add the parent directory to Python path for imports
sys.path.append(str(Path(__file__).parent.parent))

from vision_language.unified_extractor import UnifiedVisionLanguageExtractor, VisionLanguageModelConfig
from evaluation.scannet_constants import SYNONYMS_SCANNET_200, VALID_CLASS_IDS_200, VALID_CLASS_IDS_20
from evaluation.eval_semantic_instance import evaluate


class UnifiedInstSegEvaluator:
    def __init__(self, dataset_type: str, model_key: str, sentence_structure: str = "a photo of {}."):
        self.dataset_type = dataset_type
        self.model_key = model_key
        self.sentence_structure = sentence_structure
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize the unified extractor
        self.extractor = UnifiedVisionLanguageExtractor(model_key=model_key, device=self.device)
        
        # Get model info for logging
        self.model_info = self.extractor.get_model_info()
        print(f"[INFO] Using {self.model_info['model_type'].upper()} model: {self.model_info['model_name']}")
        print(f"[INFO] Feature dimension: {self.model_info['feature_dim']}")
        
        # Set up label mappings and text queries
        self.set_label_and_color_mapper(dataset_type)
        self.query_sentences = self._get_query_sentences()
        self.text_query_embeddings = self._compute_text_embeddings()
        
        print(f"[INFO] Initialized evaluator for {dataset_type} with {len(self.query_sentences)} text queries")

    def _get_query_sentences(self):
        """Generate query sentences based on dataset"""
        if self.dataset_type == 'scannet':
            # Use the 20 base classes for ScanNet
            labels = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
                     'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator',
                     'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']
            return [self.sentence_structure.format(lbl) for lbl in labels]
            
        elif self.dataset_type == 'scannet200':
            # Use all synonyms for ScanNet200
            syns = list(SYNONYMS_SCANNET_200)
            return [self.sentence_structure.format(lbl) for group in syns for lbl in group]
        else:
            raise NotImplementedError(f"Unknown dataset: {self.dataset_type}")

    def _compute_text_embeddings(self):
        """Compute text embeddings for all query sentences"""
        print(f"[INFO] Computing text embeddings for {len(self.query_sentences)} queries...")
        
        # Process all texts at once for efficiency
        text_features = self.extractor.extract_text_features(self.query_sentences)
        
        print(f"[INFO] Text embeddings computed. Shape: {text_features.shape}")
        return text_features.cpu().numpy()

    def set_label_and_color_mapper(self, dataset_type):
        """Set up label mapping functions"""
        if dataset_type == 'scannet':
            self.label_mapper = np.vectorize({i: v for i, v in enumerate(VALID_CLASS_IDS_20)}.get)
        elif dataset_type == 'scannet200':
            self.label_mapper = np.vectorize({i: v for i, v in enumerate(VALID_CLASS_IDS_200)}.get)
        else:
            raise NotImplementedError(f"Unknown dataset: {dataset_type}")

    def compute_classes_per_mask(self, masks_path: str, features_path: str, keep_first=None):
        """Compute predicted classes for each mask using similarity"""
        # Load data
        masks = torch.load(masks_path)
        feats = np.load(features_path)
        
        print(f"[INFO] Loaded masks: {masks.shape}, features: {feats.shape}")
        print(f"[INFO] Feature norms (first 5): {np.linalg.norm(feats[:5], axis=1)}")
        
        if keep_first is not None:
            masks = masks[:, :keep_first]
            feats = feats[:keep_first]
        
        # Normalize features
        feats = feats / np.linalg.norm(feats, axis=1, keepdims=True)
        
        # Compute similarities
        sims = feats @ self.text_query_embeddings.T
        
        print(f"[INFO] Similarity matrix shape: {sims.shape}")
        print(f"[INFO] Similarity stats - min: {np.min(sims):.3f}, max: {np.max(sims):.3f}, mean: {np.mean(sims):.3f}")
        
        # Find best matches
        best = np.argmax(sims, axis=1)
        best_scores = np.max(sims, axis=1)
        
        # Convert similarities to confidence scores
        min_sim = np.min(sims)
        max_sim = np.max(sims)
        if max_sim > min_sim:
            confidence_scores = (best_scores - min_sim) / (max_sim - min_sim)
        else:
            confidence_scores = np.ones_like(best_scores) * 0.5
        
        print(f"[INFO] Best scores range: {np.min(best_scores):.3f} to {np.max(best_scores):.3f}")
        print(f"[INFO] Confidence range: {np.min(confidence_scores):.3f} to {np.max(confidence_scores):.3f}")
        
        # Check for zero features
        zero_feats = np.where(np.linalg.norm(feats, axis=1) == 0)[0]
        if len(zero_feats) > 0:
            print(f"[WARNING] Found {len(zero_feats)} masks with zero features")
        
        return masks, self.label_mapper(best), confidence_scores

    def evaluate_full(self, preds: dict, scene_gt_dir: str, dataset: str, output_file: str = 'temp_output.txt'):
        """Run full evaluation using the standard evaluation function"""
        return evaluate(preds, scene_gt_dir, output_file=output_file, dataset=dataset)


def test_pipeline_full_scannet200(mask_features_dir,
                                  gt_dir,
                                  pred_root_dir,
                                  sentence_structure,
                                  feature_file_template,
                                  dataset_type='scannet200',
                                  model_key='siglip_base_patch16_384',
                                  keep_first=None,
                                  scene_list_file='evaluation/val_scenes_scannet200.txt',
                                  masks_template='_masks.pt'):
    """
    Main evaluation pipeline that works with any supported vision-language model
    """
    
    evaluator = UnifiedInstSegEvaluator(dataset_type, model_key, sentence_structure)
    print(f"[INFO] Starting evaluation for {dataset_type} using {model_key}")
    print(f"[INFO] Model info: {evaluator.model_info}")

    with open(scene_list_file, 'r') as f:
        scene_names = f.read().splitlines()

    preds = {}
    
    for scene_name in scene_names:
        print(f"\n[INFO] Processing scene: {scene_name}")
        
        # Construct file paths
        masks_path = os.path.join(pred_root_dir, scene_name + masks_template)
        features_path = os.path.join(mask_features_dir, feature_file_template.format(scene_name=scene_name))
        
        if not os.path.exists(masks_path):
            print(f"[WARNING] Masks not found: {masks_path}")
            continue
        if not os.path.exists(features_path):
            print(f"[WARNING] Features not found: {features_path}")
            continue
        
        try:
            pred_masks, pred_classes, pred_confidence = evaluator.compute_classes_per_mask(
                masks_path, features_path, keep_first=keep_first
            )
            
            preds[scene_name] = {
                'pred_masks': pred_masks.numpy(),
                'pred_classes': pred_classes,
                'pred_scores': pred_confidence
            }
            
            print(f"[INFO] Scene {scene_name}: {len(pred_classes)} predictions")
            
        except Exception as e:
            print(f"[ERROR] Failed to process scene {scene_name}: {e}")
            continue

    if not preds:
        print("[ERROR] No predictions generated. Check your file paths.")
        return None

    print(f"\n[INFO] Starting evaluation for {len(preds)} scenes...")
    results = evaluator.evaluate_full(preds, gt_dir, dataset_type)
    return results


def main():
    parser = argparse.ArgumentParser(description='Unified OpenMask3D Evaluation')
    parser.add_argument('--gt_dir', required=True, help='Ground truth directory')
    parser.add_argument('--mask_pred_dir', required=True, help='Predicted masks directory')
    parser.add_argument('--mask_features_dir', required=True, help='Mask features directory')
    parser.add_argument('--model_key', default='siglip_base_patch16_384', 
                       help='Vision-language model key (e.g., siglip_base_patch16_384, clip_vit_l_14_336)')
    parser.add_argument('--dataset', default='scannet200', choices=['scannet', 'scannet200'])
    parser.add_argument('--sentence_structure', default='a photo of {}.')
    parser.add_argument('--keep_first', type=int, help='Keep only first N masks per scene')
    parser.add_argument('--scene_list_file', default='evaluation/val_scenes_scannet200.txt')
    
    args = parser.parse_args()
    
    # Validate model key
    try:
        config_manager = VisionLanguageModelConfig()
        config_manager.get_model_config(args.model_key)
    except ValueError as e:
        print(f"[ERROR] {e}")
        return
    
    print(f"[INFO] Starting unified evaluation with model: {args.model_key}")
    
    # Run evaluation
    results = test_pipeline_full_scannet200(
        mask_features_dir=args.mask_features_dir,
        gt_dir=args.gt_dir,
        pred_root_dir=args.mask_pred_dir,
        sentence_structure=args.sentence_structure,
        feature_file_template='scene{scene_name}_openmask3d_features.npy',
        dataset_type=args.dataset,
        model_key=args.model_key,
        keep_first=args.keep_first,
        scene_list_file=args.scene_list_file
    )
    
    if results:
        print(f"\n[INFO] Evaluation completed successfully!")
        print(f"[INFO] Results saved and displayed above.")
    else:
        print(f"\n[ERROR] Evaluation failed!")


if __name__ == "__main__":
    main()
