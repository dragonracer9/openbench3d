#!/usr/bin/env python3
"""
Evaluation script for OpenMask3D using SigLIP text embeddings only.
"""
import os
import numpy as np
import torch
from transformers import SiglipProcessor, SiglipModel
from eval_semantic_instance import evaluate
from scannet_constants import SCANNET_COLOR_MAP_20, VALID_CLASS_IDS_20, CLASS_LABELS_20, SCANNET_COLOR_MAP_200, VALID_CLASS_IDS_200, CLASS_LABELS_200
import tqdm
import argparse

class SigLIPInstSegEvaluator:
    def __init__(self, dataset_type: str, siglip_model: str, sentence_structure: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset_type = dataset_type
        self.siglip_model = siglip_model

        # Load SigLIP processor and full model
        self.processor = SiglipProcessor.from_pretrained(siglip_model)
        self.model = SiglipModel.from_pretrained(siglip_model)
        self.model = self.model.to(self.device)

        # Build queries and embeddings
        self.query_sentences = self.get_query_sentences(dataset_type, sentence_structure)
        # Encode queries to get embeddings and infer feature size
        text_feats = self.get_text_query_embeddings()

        # text_feats: Tensor [num_queries, dim]
        self.feature_size = text_feats.size(1)
        print(f"[INFO] Feature size for {siglip_model} is {self.feature_size}")
        self.text_query_embeddings = text_feats.cpu().numpy()
        
        print("Text embeddings shape:", self.text_query_embeddings.shape)
        print("Text embeddings norm (first 5):", np.linalg.norm(self.text_query_embeddings[:5], axis=1))

        self.set_label_and_color_mapper(dataset_type)

    def get_query_sentences(self, dataset_type, sentence_structure="a {} in a scene"):  # same as CLIP version
        if dataset_type == 'scannet':
            labels = list(CLASS_LABELS_20)
            labels[-1] = 'other'
        elif dataset_type == 'scannet200':
            labels = list(CLASS_LABELS_200)
        elif dataset_type == 'scannet200_synonyms':
            syns = list(SYNONYMS_SCANNET_200)
            return [sentence_structure.format(lbl) for group in syns for lbl in group]
        else:
            raise NotImplementedError(f"Unknown dataset: {dataset_type}")
        return [sentence_structure.format(lbl) for lbl in labels]

    def get_text_query_embeddings(self):
        # Compute embeddings with SigLIP's get_text_features, return Tensor [N, D]
        emb_list = []
        for sentence in self.query_sentences:
            text_inputs = self.processor(
                text=sentence,
                padding="max_length",
                return_tensors="pt"
            )
            text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
            with torch.no_grad():
                emb = self.model.get_text_features(**text_inputs)
            # emb shape: (1, D)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            emb_list.append(emb.squeeze(0))
        emb_stack = torch.stack(emb_list, dim=0)
        return emb_stack# shape: (num_queries, D)

    def set_label_and_color_mapper(self, dataset_type):
        if dataset_type == 'scannet':
            self.label_mapper = np.vectorize({i: v for i,v in enumerate(VALID_CLASS_IDS_20)}.get)
        elif dataset_type == 'scannet200':
            self.label_mapper = np.vectorize({i: v for i,v in enumerate(VALID_CLASS_IDS_200)}.get)
        else:
            raise NotImplementedError(f"Unknown dataset: {dataset_type}")

    def compute_classes_per_mask(self, masks_path: str, features_path: str, keep_first=None):
        masks = torch.load(masks_path)
        feats = np.load(features_path)
        print("Mask features shape:", feats.shape)
        print("Mask features norm (first 5):", np.linalg.norm(feats[:5], axis=1))
        if keep_first is not None:
            masks = masks[:, :keep_first]
            feats = feats[:keep_first]
        feats = feats / np.linalg.norm(feats, axis=1, keepdims=True)
        sims = feats @ self.text_query_embeddings.T

        print("Similarity matrix shape:", sims.shape)
        print("Similarity matrix stats: min", np.min(sims), "max", np.max(sims), "mean", np.mean(sims))
        print("First 5 similarities:", sims[:5])

        zero_feats = np.where(np.linalg.norm(feats, axis=1) == 0)[0]
        print(f"All-zero feature indices: {zero_feats[:10]} (total: {len(zero_feats)})")    

        best = np.argmax(sims, axis=1)
        best_scores = np.max(sims, axis=1)
        
        # Convert SigLIP similarities to positive confidence scores
        min_sim = np.min(sims)
        max_sim = np.max(sims)
        if max_sim > min_sim:
            confidence_scores = (best_scores - min_sim) / (max_sim - min_sim)
        else:
            confidence_scores = np.ones_like(best_scores) * 0.5
            
        print("Best scores range:", np.min(best_scores), "to", np.max(best_scores))
        print("Confidence scores range:", np.min(confidence_scores), "to", np.max(confidence_scores))
        
        return masks, self.label_mapper(best), confidence_scores

    def compute_classes_per_mask_diff_scores(self, masks_path: str, features_path: str, keep_first=None):
        pred_masks = torch.load(masks_path)

        print("Predicted masks shape:", pred_masks.shape)
        print("Nonzero mask elements (first 5 masks):", [np.count_nonzero(pred_masks[:,i]) for i in range(min(5, pred_masks.shape[1]))])

        feats = np.load(features_path)

        print("Mask features shape:", feats.shape)
        print("Mask features norm (first 5):", np.linalg.norm(feats[:5], axis=1))
        
        keep_mask = np.ones(pred_masks.shape[1], dtype=bool)
        if keep_first is not None:
            keep_mask[keep_first:] = False
        
        # Check for zero-norm features and handle them
        feature_norms = np.linalg.norm(feats, axis=1, keepdims=True)
        zero_norm_mask = feature_norms.squeeze() == 0
        if np.any(zero_norm_mask):
            print(f"[WARNING] Found {np.sum(zero_norm_mask)} features with zero norm")
            # Set zero-norm features to a small random vector to avoid division by zero
            feats[zero_norm_mask] = np.random.normal(0, 0.01, (np.sum(zero_norm_mask), feats.shape[1]))
            feature_norms = np.linalg.norm(feats, axis=1, keepdims=True)
        
        feats = feats / feature_norms
        feats[np.isnan(feats) | np.isinf(feats)] = 0.0
        sims = feats @ self.text_query_embeddings.T
        idxs = np.argmax(sims, axis=1)


        classes = self.label_mapper(idxs)
        
        # Get the best similarity scores for each mask
        best_scores = np.max(sims, axis=1)
        
        # Convert SigLIP similarities to positive confidence scores
        # SigLIP similarities are typically negative, so we need to transform them
        # We'll use a simple transformation: confidence = (similarity - min_sim) / (max_sim - min_sim)
        min_sim = np.min(sims)
        max_sim = np.max(sims)
        if max_sim > min_sim:
            confidence_scores = (best_scores - min_sim) / (max_sim - min_sim)
        else:
            confidence_scores = np.ones_like(best_scores) * 0.5
        
        print("Similarity matrix shape:", sims.shape)
        print("Similarity matrix stats: min", np.min(sims), "max", np.max(sims), "mean", np.mean(sims))
        print("First 5 similarities:", sims[:5])
        print("Best scores range:", np.min(best_scores), "to", np.max(best_scores))
        print("Confidence scores range:", np.min(confidence_scores), "to", np.max(confidence_scores))

        zero_feats = np.where(np.linalg.norm(feats, axis=1) == 0)[0]
        print(f"All-zero feature indices: {zero_feats[:10]} (total: {len(zero_feats)})")    

        
        print("Predicted class indices (first 10):", idxs[:10])
        print("Mapped class IDs (first 10):", classes[:10])
        print("Confidence scores (first 10):", confidence_scores[:10])

        return (
            pred_masks[:, keep_mask],
            classes[keep_mask],
            confidence_scores[keep_mask]
        )

    def evaluate_full(self, preds: dict, scene_gt_dir: str, dataset: str, output_file: str = 'temp_output.txt'):
        return evaluate(preds, scene_gt_dir, output_file=output_file, dataset=dataset)

def test_pipeline_full_scannet200(mask_features_dir,
                                    gt_dir,
                                    pred_root_dir,
                                    sentence_structure,
                                    feature_file_template,
                                    dataset_type='scannet200',
                                    model_type='google/siglip-base-patch16-384',
                                    keep_first = None,
                                    scene_list_file='evaluation/val_scenes_scannet200.txt',
                                    masks_template='_masks.pt'
                         ):
    
    evaluator = SigLIPInstSegEvaluator(dataset_type, model_type, sentence_structure)
    print(f"[INFO] {dataset_type} {model_type} {sentence_structure}")

    with open(scene_list_file, 'r') as f:
        scene_names = f.read().splitlines()

    preds = {}

    for scene_name in tqdm.tqdm(scene_names[:]):

        masks_path = os.path.join(pred_root_dir, scene_name + masks_template)
        scene_per_mask_feature_path = os.path.join(mask_features_dir, feature_file_template.format(scene_name))

        if not os.path.exists(scene_per_mask_feature_path):
            print('--- SKIPPING ---', scene_per_mask_feature_path)
            continue
        pred_masks, pred_classes, pred_scores = evaluator.compute_classes_per_mask_diff_scores(
            masks_path=masks_path, 
            features_path=scene_per_mask_feature_path,
            keep_first=keep_first
        )

        preds[scene_name] = {
            'pred_masks': pred_masks,
            'pred_scores': pred_scores,
            'pred_classes': pred_classes}

    inst_AP = evaluator.evaluate_full(preds, gt_dir, dataset=dataset_type)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Evaluate OpenMask3D with SigLIP embeddings")
    parser.add_argument('--gt_dir',            type=str, required=True)
    parser.add_argument('--mask_pred_dir',     type=str, required=True)
    parser.add_argument('--mask_features_dir', type=str, required=True)
    parser.add_argument('--model_type',      type=str, default='google/siglip-base-patch16-384')
    parser.add_argument('--feature_file_template', type=str, default='{}_openmask3d_features.npy')
    parser.add_argument('--sentence_structure',type=str, default='a {} in a scene')
    parser.add_argument('--scene_list_file',      type=str, default='evaluation/val_scenes_scannet200.txt')
    parser.add_argument('--masks_template',       type=str, default='_masks.pt')
    
    opt = parser.parse_args()

    test_pipeline_full_scannet200(opt.mask_features_dir,
                                  opt.gt_dir,
                                  opt.mask_pred_dir,
                                  opt.sentence_structure,
                                  opt.feature_file_template,
                                  dataset_type='scannet200',
                                  model_type=opt.model_type,
                                  keep_first=None,
                                  scene_list_file=opt.scene_list_file,
                                  masks_template=opt.masks_template)