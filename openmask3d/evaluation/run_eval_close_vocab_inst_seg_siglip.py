#!/usr/bin/env python3
"""
Evaluation script for OpenMask3D using SigLIP text embeddings only.
"""
import os
import numpy as np
import torch
from transformers import SiglipProcessor, SiglipModel
from eval_semantic_instance import evaluate
from scannet_constants import (
    SCANNET_COLOR_MAP_20, VALID_CLASS_IDS_20, CLASS_LABELS_20,
    SCANNET_COLOR_MAP_200, VALID_CLASS_IDS_200, CLASS_LABELS_200,
    SYNONYMS_SCANNET_200
)
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
        self.query_sentences = self._build_query_list(dataset_type, sentence_structure)
        # Encode queries to get embeddings and infer feature size
        text_feats = self.get_text_query_embeddings()

        # text_feats: Tensor [num_queries, dim]
        self.feature_size = text_feats.size(1)
        print(f"[INFO] Feature size for {siglip_model} is {self.feature_size}")
        self.text_query_embeddings = text_feats.cpu()
        
        print("Text embeddings shape:", self.text_query_embeddings.shape)
        # print("Text embeddings norm (first 5):", np.linalg.norm(self.text_query_embeddings[:5], axis=1))

        self.set_label_color_mapper(dataset_type)

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

    def set_label_color_mapper(self, dataset_type):
        if dataset_type == "scannet":
            idx_to_sc20 = {i: cid for i, cid in enumerate(VALID_CLASS_IDS_20)}
            self.label_mapper = np.vectorize(idx_to_sc20.get)
            self.color_mapper = np.vectorize(SCANNET_COLOR_MAP_20.get)

        elif dataset_type in ("scannet200", "scannet200_synonyms"):
            idx_to_sc200 = {i: cid for i, cid in enumerate(VALID_CLASS_IDS_200)}
            self.label_mapper = np.vectorize(idx_to_sc200.get)
            self.color_mapper = np.vectorize(SCANNET_COLOR_MAP_200.get)

        else:
            raise NotImplementedError(f"Unknown dataset_type={dataset_type}")
    
    def _build_query_list(self, dataset_type, sentence_structure):

        # print(f"[INFO] Building query sentences for dataset_type={dataset_type}...")
        if dataset_type == "scannet":
            labels = list(CLASS_LABELS_20)
            labels[-1] = "other"  # replace "otherfurniture" with "other"
            return [sentence_structure.format(lbl) for lbl in labels]

        elif dataset_type == "scannet200":
            labels = list(CLASS_LABELS_200)
            return [sentence_structure.format(lbl) for lbl in labels]

        elif dataset_type == "scannet200_synonyms":
            class_to_synonyms = SYNONYMS_SCANNET_200

            all_synonyms = []
            synonym_idx_to_class_idx = []
            for class_idx, cls_name in enumerate(CLASS_LABELS_200):
                syn_list = class_to_synonyms[cls_name]
                for _ in syn_list:
                    synonym_idx_to_class_idx.append(class_idx)
                all_synonyms.extend(syn_list)

            self.all_synonyms = all_synonyms
            self.synonym_idx_to_class_idx = torch.tensor(synonym_idx_to_class_idx, dtype=torch.long)

            # ← Return the flat list of S synonyms here:
            return all_synonyms
        else:
            raise NotImplementedError(f"Unknown dataset_type={dataset_type}")

    # def compute_classes_per_mask(self, masks_path: str, features_path: str, keep_first=None):
    #     masks = torch.load(masks_path)
    #     feats = np.load(features_path)
    #     print("Mask features shape:", feats.shape)
    #     print("Mask features norm (first 5):", np.linalg.norm(feats[:5], axis=1))
    #     if keep_first is not None:
    #         masks = masks[:, :keep_first]
    #         feats = feats[:keep_first]
    #     feats = feats / np.linalg.norm(feats, axis=1, keepdims=True)
    #     sims = feats @ self.text_query_embeddings.T

    #     print("Similarity matrix shape:", sims.shape)
    #     print("Similarity matrix stats: min", np.min(sims), "max", np.max(sims), "mean", np.mean(sims))
    #     print("First 5 similarities:", sims[:5])

    #     zero_feats = np.where(np.linalg.norm(feats, axis=1) == 0)[0]
    #     print(f"All-zero feature indices: {zero_feats[:10]} (total: {len(zero_feats)})")    

    #     best = np.argmax(sims, axis=1)
    #     best_scores = np.max(sims, axis=1)
        
    #     # Convert SigLIP similarities to positive confidence scores
    #     min_sim = np.min(sims)
    #     max_sim = np.max(sims)
    #     if max_sim > min_sim:
    #         confidence_scores = (best_scores - min_sim) / (max_sim - min_sim)
    #     else:
    #         confidence_scores = np.ones_like(best_scores) * 0.5
            
    #     print("Best scores range:", np.min(best_scores), "to", np.max(best_scores))
    #     print("Confidence scores range:", np.min(confidence_scores), "to", np.max(confidence_scores))
        
    #     return masks, self.label_mapper(best), confidence_scores
    
    def compute_classes_per_mask(self, masks_path, mask_features_path, keep_first=None):
        pred_masks = torch.load(masks_path, weights_only=False)  # (H,W,M) or (Npts, M)
        mask_feats = np.load(mask_features_path)                  # shape (M_total, D)
        
        keep_mask = np.asarray([True for el in range(pred_masks.shape[-1])])
        if keep_first:
            keep_mask[keep_first:] = False

        # --- debug prints ---
        print(f"[DEBUG] raw mask_feats.shape = {mask_feats.shape}")
        print(f"[DEBUG] raw mask_feats[0][:10] = {mask_feats[0][:10]} …")

        # 1) Normalize each mask‐feature (L2‐norm):
        # norms = np.linalg.norm(mask_feats, axis=1, keepdims=True)  # (M,1)
        # mask_feats = mask_feats / (norms + 1e-8)
        # mask_feats[np.isnan(mask_feats) | np.isinf(mask_feats)] = 0.0
        mask_feats = mask_feats / np.linalg.norm(mask_feats, axis=1)[..., None]
        mask_feats[np.isnan(mask_feats) | np.isinf(mask_feats)] = 0.0

        # --- debug after normalization ---
        normalized_norms = np.linalg.norm(mask_feats, axis=1)
        print(f"[DEBUG] first 5 normalized mask norms = {normalized_norms[:5].tolist()}")

        # 2) Load text embeddings (already unit‐normed) into a NumPy array:
        txt_embeds = self.text_query_embeddings.numpy()            # shape = (S, D) if synonyms, else (200, D)
        print(f"[DEBUG] text_query_embeddings.shape = {txt_embeds.shape}")
        print(f"[DEBUG] first text vector (first 10 dims) = {txt_embeds[0][:10]} …")
        text_norms = np.linalg.norm(txt_embeds, axis=1)
        print(f"[DEBUG] first 5 text norms = {text_norms[:5].tolist()}")

        # 3) If using synonyms, S = total number of synonyms across all 200 classes.
        #    Otherwise, S = 200.  In both cases we do:
        sims = mask_feats @ txt_embeds.T    # shape = (M, S)
        print(f"[DEBUG] sims.shape = {sims.shape}")
        print(f"[DEBUG] sims.min() = {sims.min():.6f}, sims.max() = {sims.max():.6f}")
        print(f"[DEBUG] sims[0, :5] = {sims[0, :5].tolist()}")

        # 4) If we’re not doing synonyms, just pick argmax over S=200 classes:
        if self.dataset_type in ("scannet", "scannet200"):
            max_inds = np.argmax(sims, axis=1)                       # (M,)
            remapped = self.label_mapper(max_inds)                   # local idx → SCANNET ID 
  
            pred_classes = remapped[keep_mask]
            max_scores = np.ones(pred_classes.shape)
            pred_scores  = max_scores

        else:
            # ---- synonyms path (dataset_type == "scannet200_synonyms") ----
            # We have:
            #   sims: shape (M, S)
            #   self.synonym_idx_to_class_idx: 1D tensor of length S,
            #       saying which of the 200 classes each synonym column belongs to.

            ###########################################################
            # This takes the average of all synonyms for each class.  #
            ###########################################################
        #         M, S = sims.shape
        #         C = len(CLASS_LABELS_200)   # 200

        #         # Build an (M, C) array of zeroes:
        #         aggregated_scores = np.zeros((M, C), dtype=np.float32)

        #         # ... inside compute_classes_per_mask, after sims is computed but before aggregation:
        #         if self.dataset_type == "scannet200_synonyms":
        #             print(f"[DEBUG] Before aggregation, sims.shape = {sims.shape}  # (M, S)")
        #             # Suppose M=5, S=600.  Then aggregated_scores should be (5, 200).

        #         # For each synonym index k in [0..S-1], find out its original class c:
        #         # synonym_idx_to_class_idx is a torch.LongTensor, so convert to NumPy:
        #         syn2class = self.synonym_idx_to_class_idx.numpy()  # shape (S,)

        #         # We want to average all columns of sims[:, k] that map to the same c.
        #         # A simple way is: for each class c, do
        #         #    aggregated_scores[:, c] = sims[:, syn_indices_for_class_c].mean(axis=1)

        #         # Build a dictionary: class_idx c → list of synonym‐indices
        #         class_to_list_of_syn_idxs = {c: [] for c in range(C)}
        #         for k in range(S):
        #             c = syn2class[k]
        #             class_to_list_of_syn_idxs[c].append(k)

        #         # Now average:
        #         for c, syn_idxs in class_to_list_of_syn_idxs.items():
        #             if len(syn_idxs) == 0:
        #                 # No synonyms for this class (should only happen if your dict was incomplete).
        #                 # aggregated_scores[:, c] stays zero, or you can assign a small default.
        #                 print(f"[WARNING] No synonyms for class {c}, skipping aggregation.")
        #                 continue
        #             # syn_idxs might be e.g. [0,1,2] for "chair"
        #             cols = sims[:, syn_idxs]            # shape (M, #synonyms_for_c)
        #             aggregated_scores[:, c] = cols.mean(axis=1)
                
        #         # After aggregation:
        #         print(f"[DEBUG] aggregated_scores.shape = {aggregated_scores.shape}  # should be (M, 200)")
        #         print(f"[DEBUG] First row of aggregated_scores (first 10 classes) = {aggregated_scores[0, :10].tolist()}")


        #         # Now we have aggregated_scores shape (M, 200).
        #         # Pick argmax along axis=1:
        #         max_inds = np.argmax(aggregated_scores, axis=1)                     # (M,)
        #         max_scores = aggregated_scores[np.arange(M), max_inds]              # (M,)

        #         # Map those local indices (0..199) to SCANNET class IDs:
        #         remapped = self.label_mapper(max_inds)       # still returns length‐200 ID
        #         pred_classes = remapped[keep_mask]

        # return pred_masks, pred_classes, max_scores

            # #######################################################
            # # This takes the max of all synonyms for each class.  #
            # #######################################################

            # # ---- synonyms path (dataset_type == "scannet200_synonyms") ----
            # # sims has shape (M, S), and syn2class[k] tells you “synonym k → class c”

            M, S = sims.shape
            syn2class = self.synonym_idx_to_class_idx.numpy()  # shape (S,)

            # For each mask i, find the index of the single best‐scoring synonym:
            best_syn                         = np.argmax(sims, axis=1)        # shape (M,)
            best_syn_scores                  = sims[np.arange(M), best_syn]   # shape (M,)

            # Now map that synonym index back to a 0..199 class index:
            best_class_idx_per_mask          = syn2class[best_syn]            # shape (M,)

            # Finally turn “local index → ScanNet class ID” via your label_mapper:
            remapped                         = self.label_mapper(best_class_idx_per_mask)
            pred_classes                     = remapped[keep_mask]
            pred_scores                      = best_syn_scores

        return pred_masks, pred_classes, pred_scores

    def compute_classes_per_mask_diff_scores(self, masks_path: str, features_path: str, keep_first=None):
        pred_masks = torch.load(masks_path)

        # print("Predicted masks shape:", pred_masks.shape)
        # print("Nonzero mask elements (first 5 masks):", [np.count_nonzero(pred_masks[:,i]) for i in range(min(5, pred_masks.shape[1]))])

        feats = np.load(features_path)

        # print("Mask features shape:", feats.shape)
        # print("Mask features norm (first 5):", np.linalg.norm(feats[:5], axis=1))
        
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
        
        # print("Similarity matrix shape:", sims.shape)
        # print("Similarity matrix stats: min", np.min(sims), "max", np.max(sims), "mean", np.mean(sims))
        # print("First 5 similarities:", sims[:5])
        # print("Best scores range:", np.min(best_scores), "to", np.max(best_scores))
        # print("Confidence scores range:", np.min(confidence_scores), "to", np.max(confidence_scores))

        # zero_feats = np.where(np.linalg.norm(feats, axis=1) == 0)[0]
        # print(f"All-zero feature indices: {zero_feats[:10]} (total: {len(zero_feats)})")    

        
        # print("Predicted class indices (first 10):", idxs[:10])
        # print("Mapped class IDs (first 10):", classes[:10])
        # print("Confidence scores (first 10):", confidence_scores[:10])

        return (
            pred_masks[:, keep_mask],
            classes[keep_mask],
            confidence_scores[keep_mask]
        )

    def evaluate_full(self, preds, scene_gt_dir, dataset, output_file="temp_output.txt"):
        """
        preds: dict mapping scene_name → {
                  "pred_masks":   (tensor of shape [H,W,M] or [N_pts,M]),
                  "pred_scores":  (array of length M),
                  "pred_classes": (array of length M)
               }
        scene_gt_dir: directory containing GT .txt files
        dataset: e.g. "scannet200"
        """
        inst_AP = evaluate(preds, scene_gt_dir, output_file=output_file, dataset=dataset)
        
        return inst_AP

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
        feat_path  = os.path.join(mask_features_dir, feature_file_template.format(scene_name))

        if not os.path.exists(feat_path):
            print(f"--- SKIPPING (no features): {feat_path}")
            continue
        pred_masks, pred_classes, pred_scores = evaluator.compute_classes_per_mask(
            masks_path=masks_path,
            mask_features_path=feat_path,
            keep_first=keep_first,
        )

        preds[scene_name] = {
            'pred_masks': pred_masks,
            'pred_scores': pred_scores,
            'pred_classes': pred_classes}

    inst_AP = evaluator.evaluate_full(preds, gt_dir, dataset='scannet200')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", type=str, default='scannet/scannet_example/instance_gt/validation', help="path to GT .txt files")
    parser.add_argument("--mask_pred_dir", type=str, default='clip/masks/masks', help="where class‐agnostic masks live")
    parser.add_argument("--mask_features_dir", type=str, default='clip/mask_features', help="where per‐mask features (.npy) live")
    parser.add_argument(
        "--feature_file_template",
        type=str,
        default="{}_openmask3d_features.npy",
        help="e.g. '{}_openmask3d_features.npy'",
    )
    parser.add_argument(
        "--sentence_structure",
        type=str,
        default="a {} in a scene",
        help="sentence template, e.g. 'a {} in a scene'",
    )
    parser.add_argument(
        "--scene_list_file",
        type=str,
        default="val_scenes_scannet200.txt",
        help="file listing one scene name per line (e.g. 'scene0000_00')",
    )
    parser.add_argument(
        "--masks_template",
        type=str,
        default="_masks.pt",
        help="suffix for your saved mask files",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="scannet200_synonyms",
        #default="scannet200",
        choices=["scannet", "scannet200", "scannet200_synonyms"],
        help="which dataset's labels/colors to use",
    )    
    parser.add_argument(
        '--model_type',
        type=str, 
        default='google/siglip-base-patch16-384',
    )
    parser.add_argument(
        "--keep_first",
        type=int,
        default=None,
        help="if set, only keep the first K masks from each scene",
    )
    args = parser.parse_args()

    test_pipeline_full_scannet200(
        mask_features_dir=args.mask_features_dir,
        gt_dir=args.gt_dir,
        pred_root_dir=args.mask_pred_dir,
        sentence_structure=args.sentence_structure,
        feature_file_template=args.feature_file_template,
        dataset_type=args.dataset_type,
        model_type=args.model_type,
        keep_first=args.keep_first,
        scene_list_file=args.scene_list_file,
        masks_template=args.masks_template,
    )