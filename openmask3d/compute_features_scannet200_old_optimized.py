#!/usr/bin/env python3

from openmask3d.compute_features_scannet200_old import main as old_main
import hydra
from omegaconf import DictConfig
import sys
import os

# Add the optimized version to the path
sys.path.insert(0, '/home/ninol/openbench3d/openmask3d/mask_features_computation')

def swap_to_optimized_extractor():
    """Temporarily replace the old extractor with the optimized version"""
    import openmask3d.mask_features_computation.features_extractor_siglip_old as old_module
    import features_extractor_siglip_old_optimized as optimized_module
    
    # Replace the class in the old module
    old_module.FeaturesExtractorSiglip = optimized_module.FeaturesExtractorSiglip
    print("[INFO] Using memory-optimized version of the old feature extractor")

@hydra.main(config_path="configs", config_name="openmask3d_scannet200_eval")
def main(cfg: DictConfig) -> None:
    # Use the optimized extractor
    swap_to_optimized_extractor()
    
    # Run the old main function with optimized extractor
    old_main(cfg)

if __name__ == "__main__":
    main()
