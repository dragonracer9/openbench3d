import matplotlib.cm
import matplotlib
import os
import sys
import argparse
import numpy as np
import open3d as o3d
import torch
import clip
import matplotlib.pyplot as plt
from constants import *





class QuerySimilarityComputation():
    def __init__(self,):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.clip_model, _ = clip.load('ViT-L/14@336px', self.device)

    def get_query_embedding(self, text_query):
        text_input_processed = clip.tokenize(text_query).to(self.device)
        with torch.no_grad():
            sentence_embedding = self.clip_model.encode_text(text_input_processed)

        sentence_embedding_normalized =  (sentence_embedding/sentence_embedding.norm(dim=-1, keepdim=True)).float().cpu()
        return sentence_embedding_normalized.squeeze().numpy()
 
    def compute_similarity_scores(self, mask_features, text_query):
        text_emb = self.get_query_embedding(text_query)

        scores = np.zeros(len(mask_features))
        for mask_idx, mask_emb in enumerate(mask_features):
            mask_norm = np.linalg.norm(mask_emb)
            if mask_norm < 0.001:
                continue
            normalized_emb = (mask_emb/mask_norm)
            scores[mask_idx] = normalized_emb@text_emb

        return scores
    
    def get_per_point_colors_for_similarity(self, 
                                            per_mask_scores, 
                                            masks, 
                                            normalize_based_on_current_min_max=False, 
                                            normalize_min_bound=0.16, #only used for visualization if normalize_based_on_current_min_max is False
                                            normalize_max_bound=0.26, #only used for visualization if normalize_based_on_current_min_max is False
                                            background_color=(0.77, 0.77, 0.77)
                                        ):
        # get colors based on the openmask3d per mask scores
        non_zero_points = per_mask_scores!=0
        openmask3d_per_mask_scores_rescaled = np.zeros_like(per_mask_scores)
        pms = per_mask_scores[non_zero_points]

        # in order to be able to visualize the score differences better, we can use a normalization scheme
        if normalize_based_on_current_min_max: # if true, normalize the scores based on the min. and max. scores for this scene
            openmask3d_per_mask_scores_rescaled[non_zero_points] = (pms-pms.min()) / (pms.max() - pms.min())
        else: # if false, normalize the scores based on a pre-defined color scheme with min and max clipping bounds, normalize_min_bound and normalize_max_bound.
            new_scores = np.zeros_like(openmask3d_per_mask_scores_rescaled)
            new_indices = np.zeros_like(non_zero_points)
            new_indices[non_zero_points] += pms>normalize_min_bound
            new_scores[new_indices] = ((pms[pms>normalize_min_bound]-normalize_min_bound)/(normalize_max_bound-normalize_min_bound))
            openmask3d_per_mask_scores_rescaled = new_scores

        new_colors = np.ones((masks.shape[1], 3))*0 + background_color
        jet = matplotlib.cm.get_cmap("jet")
        for mask_idx, mask in enumerate(masks[::-1, :]):
            # get color from matplotlib colormap
            new_colors[mask>0.5, :] = jet(openmask3d_per_mask_scores_rescaled[len(masks)-mask_idx-1])[:3]

        return new_colors



def cli_query_loop(scene_name: str):
        # ─── Paths based on the chosen scene ─────────────────────────────────
    base_scan_dir = "/home/ninol/openbench3d/data/scans"
    base_masks_dir = "/home/ninol/openbench3d/output/clip/masks"
    base_feats_dir = "/home/ninol/openbench3d/output/clip/mask_features_final"

    #  Make sure scene exists
    pcd_path = os.path.join(base_scan_dir, f"{scene_name}/{scene_name}_vh_clean_2.ply")
    masks_path = os.path.join(base_masks_dir, f"{scene_name}_masks.pt")
    feats_path = os.path.join(base_feats_dir, f"{scene_name}_openmask3d_features.npy")

    if not (os.path.isfile(pcd_path) and
            os.path.isfile(masks_path) and
            os.path.isfile(feats_path)):
        # List available scenes by looking in features folder
        available = []
        for fname in os.listdir(base_feats_dir):
            if fname.endswith("_openmask3d_features.npy"):
                available.append(fname.replace("_openmask3d_features.npy", ""))
        print(f"Scene '{scene_name}' not found.\nAvailable scenes:")
        for s in sorted(available):
            print("  •", s)
        sys.exit(1)
    query_similarity_computer = QuerySimilarityComputation()
    scene_pcd = o3d.io.read_point_cloud(pcd_path)
    pred_masks = np.asarray(torch.load(masks_path)).T
    openmask3d_features = np.load(feats_path)
    print("Type a query and press Enter to update the visualization. Type 'exit' to quit.")
    last_query = "a table in a scene"
    while True:
        query = input(f"Query [{last_query}]: ").strip()
        if query == "":
            query = last_query
        if query.lower() == "exit":
            break
        last_query = query
        per_mask_query_sim_scores = query_similarity_computer.compute_similarity_scores(
            openmask3d_features, query)
        per_point_similarity_colors = query_similarity_computer.get_per_point_colors_for_similarity(
            per_mask_query_sim_scores, pred_masks)
        scene_pcd.colors = o3d.utility.Vector3dVector(per_point_similarity_colors)
        o3d.visualization.draw_geometries([scene_pcd], window_name=f"Query: {query}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "scene",
        nargs="?",
        default="scene0030_02",
        help="Name of the ScanNet scene (e.g. 'scene0030_02').",
    )
    args = parser.parse_args()
    cli_query_loop(args.scene)