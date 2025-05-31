from segment_anything import sam_model_registry, SamPredictor
import numpy as np
import torch

def initialize_sam_model(device, sam_model_type, sam_checkpoint):
    sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
    sam.to(device)
    predictor_sam = SamPredictor(sam) 
    return predictor_sam

def mask2box(mask: torch.Tensor):
    row = torch.nonzero(mask.sum(axis=0))[:, 0]
    if len(row) == 0:
        return None
    x1 = row.min().item()
    x2 = row.max().item()
    col = np.nonzero(mask.sum(axis=1))[:, 0]
    y1 = col.min().item()
    y2 = col.max().item()
    return x1, y1, x2 + 1, y2 + 1

def mask2box_multi_level(mask: torch.Tensor, level, expansion_ratio):
    x1, y1, x2 , y2  = mask2box(mask)
    if level == 0:
        return x1, y1, x2 , y2
    shape = mask.shape
    x_exp = int(abs(x2- x1)*expansion_ratio) * level
    y_exp = int(abs(y2-y1)*expansion_ratio) * level
    return max(0, x1 - x_exp), max(0, y1 - y_exp), min(shape[1], x2 + x_exp), min(shape[0], y2 + y_exp)

# def run_sam(image_size, num_random_rounds, num_selected_points, point_coords, predictor_sam: SamPredictor):
#     best_score = 0
#     best_mask = np.zeros_like(image_size, dtype=bool)
    
#     point_coords_new = np.zeros_like(point_coords)
#     point_coords_new[:,0] = point_coords[:,1]
#     point_coords_new[:,1] = point_coords[:,0]
    
#     # Get only a random subsample of them for num_random_rounds times and choose the mask with highest confidence score
#     for i in range(num_random_rounds):
#         np.random.shuffle(point_coords_new)
#         masks, scores, logits = predictor_sam.predict(
#             point_coords=point_coords_new[:num_selected_points],
#             point_labels=np.ones(point_coords_new[:num_selected_points].shape[0]),
#             multimask_output=False,
#         )  
        
#         if scores[0] > best_score:
#             best_score = scores[0]
#             best_mask = masks[0]
            
#     return best_mask

def run_sam(image_size, num_random_rounds, num_selected_points, point_coords, predictor_sam: SamPredictor):
    best_score = 0
    best_mask = np.zeros_like(image_size, dtype=bool)
    
    point_coords_new = np.zeros_like(point_coords)
    point_coords_new[:,0] = point_coords[:,1]
    point_coords_new[:,1] = point_coords[:,0]
    
    # Prepare batch data for all random rounds
    batch_point_coords = []
    batch_point_labels = []
    
    for i in range(num_random_rounds):
        np.random.shuffle(point_coords_new)
        selected_coords = point_coords_new[:num_selected_points].copy()
        selected_labels = np.ones(selected_coords.shape[0])
        
        batch_point_coords.append(selected_coords)
        batch_point_labels.append(selected_labels)
    
    # Convert to torch tensors with batch dimension
    # Shape: [num_random_rounds, num_selected_points, 2]
    batch_coords_tensor = torch.tensor(np.array(batch_point_coords), dtype=torch.float32)
    batch_labels_tensor = torch.tensor(np.array(batch_point_labels), dtype=torch.int32)
    
    # Move to same device as SAM model
    if hasattr(predictor_sam.model, 'device'):
        device = predictor_sam.model.device
    else:
        device = next(predictor_sam.model.parameters()).device
    
    batch_coords_tensor = batch_coords_tensor.to(device)
    batch_labels_tensor = batch_labels_tensor.to(device)
    
    # Run batch prediction
    try:
        masks, scores, logits = predictor_sam.predict_torch(
            point_coords=batch_coords_tensor,
            point_labels=batch_labels_tensor,
            multimask_output=False,
        )
        
        # Convert back to numpy and find best mask
        scores_np = scores.cpu().numpy()
        masks_np = masks.cpu().numpy()
        
        # Find the round with the best score
        best_round_idx = np.argmax(scores_np[:, 0])  # scores shape: [batch_size, 1]
        best_score = scores_np[best_round_idx, 0]
        best_mask = masks_np[best_round_idx, 0]  # masks shape: [batch_size, 1, H, W]
        
    except Exception as e:
        # Fallback to sequential processing if batch processing fails
        print(f"Batch SAM processing failed, falling back to sequential: {e}")
        for i in range(num_random_rounds):
            np.random.shuffle(point_coords_new)
            masks, scores, logits = predictor_sam.predict(
                point_coords=point_coords_new[:num_selected_points],
                point_labels=np.ones(point_coords_new[:num_selected_points].shape[0]),
                multimask_output=False,
            )  
            
            if scores[0] > best_score:
                best_score = scores[0]
                best_mask = masks[0]
    
    return best_mask