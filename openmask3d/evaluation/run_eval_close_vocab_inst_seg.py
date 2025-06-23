import os
import clip
import numpy as np
import torch
import clip
from transformers import (
    CLIPProcessor,
    CLIPTextModel,
    BlipProcessor,
    BlipModel,
)
from eval_semantic_instance import evaluate
from scannet_constants import (
    SCANNET_COLOR_MAP_20,
    VALID_CLASS_IDS_20,
    CLASS_LABELS_20,
    SCANNET_COLOR_MAP_200,
    VALID_CLASS_IDS_200,
    CLASS_LABELS_200,
    SYNONYMS_SCANNET_200,
)
import tqdm
import argparse


class InstSegEvaluator:
    def __init__(self, dataset_type, model_type, sentence_structure="a {} in a scene"):
        """
        dataset_type:       "scannet", "scannet200", or "scannet200_synonyms"
        model_type:         "clip", "eva", or "blip"
        sentence_structure: e.g. "a {} in a scene"
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.dataset_type = dataset_type
        self.dataset_type = dataset_type.lower()
        assert self.dataset_type in ("scannet", "scannet200", "scannet200_synonyms"), \
            "dataset_type must be one of ['scannet', 'scannet200', 'scannet200_synonyms']"
        self.model_type = model_type.lower()
        assert self.model_type in ("clip", "eva", "blip"), \
            "model_type must be one of ['clip', 'eva', 'blip']"

        # 1) Load the chosen backend's text encoder (and tokenizer/processor if needed).
        if self.model_type == "clip":
            # ───────────────────────────────────────────────────────────────────────────────
            # CLIP backend (openai/clip).  We allow user to pass any CLIP variant,
            # e.g. "ViT-B/32" or "ViT-L/14@336px".
            # clip.load(model_name, device) returns (model, preprocess_fn).
            self.clip_model, _ = clip.load("ViT-L/14@336px", device=self.device)
            self.clip_model.eval()

            # Query CLIP's actual text‐embedding size by doing a single dummy encode.
            with torch.no_grad():
                dummy_tokens = clip.tokenize(["dummy"]).to(self.device)  # [1, token_length]
                dummy_emb = self.clip_model.encode_text(dummy_tokens)   # [1, D_clip]
            self.feature_dim = dummy_emb.shape[-1]  # e.g. 512 for ViT-B/32, 768 for ViT-L/14, etc.

        elif self.model_type == "eva":
            # ───────────────────────────────────────────────────────────────────────────────
            # EVA-CLIP backend.  We only need its text encoder:
            #   1) CLIPProcessor (for tokenization)
            #   2) CLIPTextModel.from_pretrained("QuanSun/EVA-CLIP", subfolder=...)
            #
            # Here we assume the checkpoint subfolder is "EVA02-CLIP-L-336_psz14_s6B" (L/14@336).
            # Adjust subfolder if you want B/16 or E/14, etc.
            self.eva_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

            self.eva_text = CLIPTextModel.from_pretrained(
                "QuanSun/EVA-CLIP",
                subfolder="EVA02-CLIP-L-336_psz14_s6B",
                ignore_mismatched_sizes=True
            ).to(self.device)
            self.eva_text.eval()

            # Hidden size (1024 for EVA-CLIP-L/14@336).
            self.feature_dim = self.eva_text.config.hidden_size

        elif self.model_type == "blip":
            # ───────────────────────────────────────────────────────────────────────────────
            # BLIP backend (Salesforce/blip-image-captioning-large).  We only need text side.
            self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.blip_model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)
            self.blip_model.eval()

            # self.feature_dim = self.blip_model.config.text_config.hidden_size 

            # Query BLIP's text‐embedding size via a dummy pass:
            with torch.no_grad():
                dummy_tokens = self.blip_processor.tokenizer(
                    ["dummy"],
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.device)
                dummy_out = self.blip_model.text_model(**dummy_tokens)  # [1, D_blip]
                dummy_out = dummy_out.pooler_output  # shape = (1, 1024)
            self.feature_dim = dummy_out.shape[-1]  # typically 768
        else:
            raise NotImplementedError(f"Unknown model_type={self.model_type}")

        # 2) Build the list of query sentences for each class
        print(f"[INFO] Building query sentences for dataset_type={dataset_type} with model_type={model_type}")
        self.query_sentences = self._build_query_list(self.dataset_type, sentence_structure)

        if self.dataset_type == "scannet200_synonyms":
            missing = [lbl for lbl in CLASS_LABELS_200 if lbl not in SYNONYMS_SCANNET_200]
            if missing:
                print("These labels have no synonyms defined:", missing)
            else:
                print("All 200 labels are covered in SYNONYMS_SCANNET_200.")
            sentences_to_encode = self.all_synonyms      # length = S
        else:
            sentences_to_encode = self.query_sentences   # length = C = 200 (no synonyms)

        # print(self.query_sentences)

        # Now build (or load) your text‐embeddings file:
        embeddings_file = f"text_embeddings_{self.dataset_type}_{self.model_type}.npy"
        if os.path.exists(embeddings_file):
            self.text_query_embeddings = torch.tensor(
                np.load(embeddings_file), device="cpu"
            )  # shape (S, D) or (C, D)
        else:
            print(f"[INFO] Creating text embeddings for {len(sentences_to_encode)} sentences …")
            self.text_query_embeddings = self.encode_texts(sentences_to_encode)  # returns (N, D) on CPU
            np.save(embeddings_file, self.text_query_embeddings.numpy())

        # 4) Prepare label / color mappers
        self._set_label_color_mapper(dataset_type)

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

    def _set_label_color_mapper(self, dataset_type):
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

    def encode_texts(self, sentences):
        """
        Given a list of N sentences, returns a (N, feature_dim) FloatTensor
        of L2‐normalized text embeddings (on CPU).  Backend depends on self.model_type.
        """
        N = len(sentences)
        all_embeds = torch.zeros((N, self.feature_dim),
                                 dtype=torch.float32,
                                 device=self.device)

        if self.model_type == "clip":
            # ───────────────────────────────────────────────────────────────────────
            # CLIP backend: clip.tokenize + clip_model.encode_text
            for i, sent in enumerate(sentences):
                toks = clip.tokenize([sent]).to(self.device)     # shape [1, T]
                with torch.no_grad():
                    emb = self.clip_model.encode_text(toks)      # [1, D_clip]
                emb = emb / emb.norm(dim=-1, keepdim=True)
                all_embeds[i] = emb.squeeze(0)

        elif self.model_type == "eva":
            # ───────────────────────────────────────────────────────────────────────
            # EVA-CLIP backend: CLIPProcessor + CLIPTextModel
            encoded = self.eva_processor.text_processor(
                text=sentences,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)  # {"input_ids": ..., "attention_mask": ...}

            with torch.no_grad():
                out = self.eva_text(**encoded)
            emb = out.pooler_output                     # [N, D_eva]
            emb = emb / emb.norm(dim=-1, keepdim=True)  # normalize each row
            all_embeds[:] = emb

        elif self.model_type == "blip":
            print("[INFO] Using BLIP backend for text encoding...")
            # ───────────────────────────────────────────────────────────────────────
            # BLIP backend: BlipProcessor + BlipModel.get_text_features
            encoded = self.blip_processor.tokenizer(
                sentences,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                emb =  self.blip_model.text_model(**encoded)  # [N, D_blip]
                pooled      = emb.pooler_output    # shape = (num_classes, 1024)
            emb = pooled / pooled.norm(dim=-1, keepdim=True)
            all_embeds[:] = emb
        else:
            raise NotImplementedError(f"Unknown model_type={self.model_type}")

        all_embeds_cpu = all_embeds.cpu()
        # --- DEBUGGING: print a couple of norms to be sure ---
        norms = all_embeds_cpu.norm(dim=1)   # (N,)
        print(f"[DEBUG] First 5 text‐embedding norms ≈ 1? {norms[:5].tolist()}")

        return all_embeds_cpu  # return a CPU tensor

    # def compute_classes_per_mask(self, masks_path, mask_features_path, keep_first=None):
    #     """
    #     loads:
    #       - pred_masks = torch.load(masks_path)      # e.g. [H, W, M] or [N_pts, M]
    #       - mask_feats  = np.load(mask_features_path) # shape (M, feature_dim)
    #     then:
    #       - optionally keep only the first K masks
    #       - normalize each mask_feature
    #       - compute (M, C) = (M, D) @ (D, C_text)
    #       - pick max along classes → pred_classes, pred_scores
    #     returns: (pred_masks, pred_classes, pred_scores)
    #     """
    #     pred_masks = torch.load(masks_path, weights_only=False)
    #     mask_feats = np.load(mask_features_path)  # shape (M_total, D)

    #     if keep_first is not None:
    #         pred_masks = pred_masks[..., :keep_first]      # keep first K masks
    #         mask_feats = mask_feats[:keep_first, :]        # (K, D)

    #      # --- DEBUGGING: print raw mask_feats shape & a few rows BEFORE normalization ---
    #     print(f"[DEBUG] raw mask_feats.shape = {mask_feats.shape}")
    #     print(f"[DEBUG] raw mask_feats[0] (first row) = {mask_feats[0][:10]} …")  # first 10 dims


    #     # normalize each mask feature
    #     norms = np.linalg.norm(mask_feats, axis=1, keepdims=True)  # (M, 1)
    #     mask_feats = mask_feats / (norms + 1e-8)
    #     mask_feats[np.isnan(mask_feats) | np.isinf(mask_feats)] = 0.0

    #     # --- DEBUGGING: print a few normalized mask norms to verify unity ---
    #     normalized_norms = np.linalg.norm(mask_feats, axis=1)
    #     print(f"[DEBUG] first 5 normalized mask norms = {normalized_norms[:5].tolist()}")

    #     # At this point, mask_feats is (M, D).  Now load text embeddings:
    #     txt_embeds = self.text_query_embeddings.numpy()  # shape (C, D)
    #     print(f"[DEBUG] text_query_embeddings.shape = {txt_embeds.shape}")
    #     print(f"[DEBUG] first text vector (first 10 dims) = {txt_embeds[0][:10]} …")

    #     # --- DEBUGGING: print a few text norms too ---
    #     text_norms = np.linalg.norm(txt_embeds, axis=1)
    #     print(f"[DEBUG] first 5 text norms = {text_norms[:5].tolist()}")

    #     # Check that D matches on both sides:
    #     D_mask = mask_feats.shape[1]
    #     D_text = txt_embeds.shape[1]
    #     if D_mask != D_text:
    #         print(f"[ERROR] Dimension mismatch: mask_feats is D={D_mask}, text is D={D_text}")

    #     # Step 1: Create an index mapping for synonyms
    #     # if self.dataset_type == "scannet200_synonyms":
    #     #     print("[INFO] Creating index mapping for synonyms...")

    #     #     # Load synonym mapping
    #     #     synonyms_all = SYNONYMS_SCANNET_200  # {class: [synonyms]}
    #     #     class_to_synonyms = {key: values for key, values in synonyms_all.items()}

    #     #     # Create forward mapping: class index → synonym indices
    #     #     class_to_synonym_indices = {}
    #     #     synonym_index = 0
    #     #     for class_idx, (cls, synonyms) in enumerate(class_to_synonyms.items()):
    #     #         class_to_synonym_indices[cls] = (class_idx, list(range(synonym_index, synonym_index + len(synonyms))))
    #     #         synonym_index += len(synonyms)

    #     #     # Create reverse mapping: synonym index → class index
    #     #     synonym_to_class = {}
    #     #     for cls, (class_idx, synonym_indices) in class_to_synonym_indices.items():
    #     #         for synonym_idx in synonym_indices:
    #     #             synonym_to_class[synonym_idx] = (class_idx, cls)

    #     #     print(f"[INFO] Class to synonym indices: {class_to_synonym_indices}")
    #     #     print(f"[INFO] Synonym to class mapping: {synonym_to_class}")
    #     #     print(f"[INFO] Total synonyms: {len(synonym_to_class)}")

    #     #     # Step 2: Duplicate mask features to match text features
    #     #     print("[INFO] Duplicating mask features to match text features...")
    #     #     duplicated_mask_feats = np.zeros((len(synonym_to_class), mask_feats.shape[1]))
    #     #     for cls, (class_idx, synonym_indices) in class_to_synonym_indices.items():
    #     #         for synonym_idx in synonym_indices:
    #     #             # Ensure synonym_idx is a valid index
    #     #             if synonym_idx >= len(duplicated_mask_feats):
    #     #                 print(f"[WARNING] Synonym index {synonym_idx} is out of bounds for duplicated_mask_feats with size {duplicated_mask_feats.shape[0]}. Skipping.")
    #     #                 continue
                    
    #     #             duplicated_mask_feats[synonym_idx] = mask_feats[class_idx]

    #     #     mask_feats = duplicated_mask_feats

    #     #     print(f"[INFO] Duplicated mask features shape: {mask_feats.shape}")
    #     #     # --- DEBUGGING: print a few rows of duplicated mask features ---
    #     #     print(f"[DEBUG] First 5 duplicated mask features:\n{mask_feats[:5, :10]} ...")  # first 10 dims

    #     #     # Step 3: Calculate similarity scores (no change here)
    #     #     sims = mask_feats @ self.text_query_embeddings.numpy().T  # (M, C)

    #     #     # Step 4: Aggregate scores according to class
    #     #     print("[INFO] Aggregating similarity scores by class...")
    #     #     aggregated_scores = np.zeros((mask_feats.shape[0], len(class_to_synonyms)))
    #     #     for cls, (class_idx, synonym_indices) in class_to_synonym_indices.items():
    #     #         # Aggregate by class_idx: average all synonym columns belonging to this class
    #     #         synonym_indices = np.array(synonym_indices)
    #     #         aggregated_scores[:, class_idx] = sims[:, synonym_indices].mean(axis=1)

    #     #     # Update sims to use aggregated scores
    #     #     sims = aggregated_scores

    #     #     print(f"[INFO] Aggregated scores shape: {sims.shape}")
    #     # else:
    #     # Step 3: Calculate similarity scores (no change here)
    #     sims = mask_feats @ self.text_query_embeddings.numpy().T  # (M, C)

    #     max_inds = np.argmax(sims, axis=1)                         # (M,)
    #     max_scores = sims[np.arange(sims.shape[0]), max_inds]       # (M,)

    #     remapped = self.label_mapper(max_inds)  # map local index→SCANNET class ID
    #     pred_classes = remapped.astype(np.int64)

    #     return pred_masks, pred_classes, max_scores

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
                M, S = sims.shape
                C = len(CLASS_LABELS_200)   # 200

                # Build an (M, C) array of zeroes:
                aggregated_scores = np.zeros((M, C), dtype=np.float32)

                # ... inside compute_classes_per_mask, after sims is computed but before aggregation:
                if self.dataset_type == "scannet200_synonyms":
                    print(f"[DEBUG] Before aggregation, sims.shape = {sims.shape}  # (M, S)")
                    print(f"[DEBUG] synonym_idx_to_class_idx[:10] = {self.synonym_idx_to_class_idx[:10].tolist()}")
                    # Suppose M=5, S=600.  Then aggregated_scores should be (5, 200).

                # For each synonym index k in [0..S-1], find out its original class c:
                # synonym_idx_to_class_idx is a torch.LongTensor, so convert to NumPy:
                syn2class = self.synonym_idx_to_class_idx.numpy()  # shape (S,)

                # We want to average all columns of sims[:, k] that map to the same c.
                # A simple way is: for each class c, do
                #    aggregated_scores[:, c] = sims[:, syn_indices_for_class_c].mean(axis=1)

                # Build a dictionary: class_idx c → list of synonym‐indices
                class_to_list_of_syn_idxs = {c: [] for c in range(C)}
                for k in range(S):
                    c = syn2class[k]
                    class_to_list_of_syn_idxs[c].append(k)

                # Now average:
                for c, syn_idxs in class_to_list_of_syn_idxs.items():
                    if len(syn_idxs) == 0:
                        # No synonyms for this class (should only happen if your dict was incomplete).
                        # aggregated_scores[:, c] stays zero, or you can assign a small default.
                        print(f"[WARNING] No synonyms for class {c}, skipping aggregation.")
                        continue
                    # syn_idxs might be e.g. [0,1,2] for "chair"
                    cols = sims[:, syn_idxs]            # shape (M, #synonyms_for_c)
                    aggregated_scores[:, c] = cols.mean(axis=1)
                
                # After aggregation:
                print(f"[DEBUG] aggregated_scores.shape = {aggregated_scores.shape}  # should be (M, 200)")
                print(f"[DEBUG] First row of aggregated_scores (first 10 classes) = {aggregated_scores[0, :10].tolist()}")


                # Now we have aggregated_scores shape (M, 200).
                # Pick argmax along axis=1:
                max_inds = np.argmax(aggregated_scores, axis=1)                     # (M,)
                max_scores = aggregated_scores[np.arange(M), max_inds]              # (M,)

                # Map those local indices (0..199) to SCANNET class IDs:
                remapped = self.label_mapper(max_inds)       # still returns length‐200 ID
                pred_classes = remapped[keep_mask]

        return pred_masks, pred_classes, max_scores

            # #######################################################
            # # This takes the max of all synonyms for each class.  #
            # #######################################################

            # # ---- synonyms path (dataset_type == "scannet200_synonyms") ----
            # # sims has shape (M, S), and syn2class[k] tells you “synonym k → class c”

        #     M, S = sims.shape
        #     syn2class = self.synonym_idx_to_class_idx.numpy()  # shape (S,)

        #     # For each mask i, find the index of the single best‐scoring synonym:
        #     best_syn                         = np.argmax(sims, axis=1)        # shape (M,)
        #     best_syn_scores                  = sims[np.arange(M), best_syn]   # shape (M,)

        #     # Now map that synonym index back to a 0..199 class index:
        #     best_class_idx_per_mask          = syn2class[best_syn]            # shape (M,)

        #     # Finally turn “local index → ScanNet class ID” via your label_mapper:
        #     remapped                         = self.label_mapper(best_class_idx_per_mask)
        #     pred_classes                     = remapped[keep_mask]
        #     pred_scores                      = best_syn_scores

        # return pred_masks, pred_classes, pred_scores

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


def test_pipeline_full_scannet200(
    mask_features_dir,
    gt_dir,
    pred_root_dir,
    sentence_structure,
    feature_file_template,
    dataset_type="scannet200",
    model_type="clip",                     # "clip", "eva", or "blip"
    keep_first=None,
    scene_list_file="val_scenes_scannet200.txt",
    masks_template="_masks.pt",
):
    evaluator = InstSegEvaluator(dataset_type, model_type, sentence_structure)
    print(f"[INFO] dataset: {dataset_type}, model: {model_type}, template: {sentence_structure}")

    with open(scene_list_file, "r") as f:
        scene_names = f.read().splitlines()

    preds = {}
    for scene_name in tqdm.tqdm(scene_names):
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
            "pred_masks":   pred_masks,
            "pred_classes": pred_classes,
            "pred_scores":  pred_scores,
        }

    # inst_AP = evaluator.evaluate_full(preds, gt_dir, dataset=dataset_type)
    inst_AP = evaluator.evaluate_full(preds, gt_dir, dataset='scannet200')
    # print(f"→ Instance AP on {dataset_type} with {model_type}: {inst_AP:.4f}")


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
        "--model_type",
        type=str,
        default="clip",
        choices=["clip", "eva", "blip"],
        help="which text‐encoder backend to use",
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