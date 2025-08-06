import torch
import torch.nn.functional as F
import numpy as np

# def continuous_to_discrete_sentiment(labels):
#     """
#     Convert continuous sentiment labels to discrete classes.
#     Maps values from [-3, +3] range to classes [0, 6].
    
#     Args:
#         labels: Tensor of shape [batch_size, 1] with continuous values
    
#     Returns:
#         discrete_labels: Tensor of shape [batch_size] with class indices [0-6]
#     """
#     # Round to nearest integer and clamp to valid range
#     discrete = torch.round(labels.squeeze()).long()  # Remove the dimension of size 1
#     discrete = torch.clamp(discrete + 3, min=0, max=6)  # Shift from [-3,3] to [0,6]
#     return discrete


def continuous_to_discrete_sentiment(labels):
    """
    Convert continuous sentiment labels to discrete classes using explicit ranges.
    This approach ensures predictable and semantically meaningful class boundaries.
    
    Args:
        labels: Tensor of shape [batch_size, 1] with continuous values
    
    Returns:
        discrete_labels: Tensor of shape [batch_size] with class indices [0-6]
    """
    labels_flat = labels.squeeze()
    discrete = torch.zeros_like(labels_flat, dtype=torch.long)
    discrete[labels_flat < -2.5] = 0    # Highly negative: [-3.0, -2.5)
    discrete[(labels_flat >= -2.5) & (labels_flat < -1.5)] = 1  # Negative: [-2.5, -1.5)
    discrete[(labels_flat >= -1.5) & (labels_flat < -0.5)] = 2  # Weakly negative: [-1.5, -0.5)
    discrete[(labels_flat >= -0.5) & (labels_flat < 0.5)] = 3   # Neutral: [-0.5, 0.5)
    discrete[(labels_flat >= 0.5) & (labels_flat < 1.5)] = 4    # Weakly positive: [0.5, 1.5)
    discrete[(labels_flat >= 1.5) & (labels_flat < 2.5)] = 5    # Positive: [1.5, 2.5)
    discrete[labels_flat >= 2.5] = 6    # Highly positive: [2.5, 3.0]
    return discrete

    

def discrete_to_onehot_sentiment(discrete_labels, num_classes=7):
    """
    Convert discrete class labels to one-hot vectors.
    
    Args:
        discrete_labels: Tensor of shape [batch_size] with class indices
        num_classes: Number of sentiment classes (default 7)
    
    Returns:
        onehot_labels: Tensor of shape [batch_size, num_classes]
    """
    return F.one_hot(discrete_labels, num_classes=num_classes).float()

def continuous_to_onehot_sentiment(labels, num_classes=7):
    """
    Direct conversion from continuous to one-hot (combines above functions).
    
    Args:
        labels: Tensor of shape [batch_size, 1] with continuous values
        num_classes: Number of sentiment classes (default 7)
    
    Returns:
        onehot_labels: Tensor of shape [batch_size, num_classes]
    """
    discrete = continuous_to_discrete_sentiment(labels)
    return discrete_to_onehot_sentiment(discrete, num_classes)





# def discrete_to_continuous_sentiment(discrete_labels):
#     """
#     Convert discrete class indices back to continuous sentiment values.
#     Useful for evaluation with your existing regression metrics.
    
#     Args:
#         discrete_labels: Tensor of class indices [0-6]
    
#     Returns:
#         continuous_labels: Tensor of continuous values [-3 to +3]
#     """
#     return (discrete_labels - 3).float()

# def predictions_to_continuous(class_predictions):
#     """
#     Convert classification predictions (logits or probabilities) to continuous values.
#     Uses expected value calculation for smooth conversion.
    
#     Args:
#         class_predictions: Tensor of shape [batch_size, 7] (logits or probs)
    
#     Returns:
#         continuous_preds: Tensor of shape [batch_size, 1] with continuous values
#     """
#     # If logits, convert to probabilities
#     if class_predictions.max() > 1.0 or class_predictions.min() < 0.0:
#         probs = F.softmax(class_predictions, dim=1)
#     else:
#         probs = class_predictions
    
#     # Create class values tensor [-3, -2, -1, 0, 1, 2, 3]
#     class_values = torch.arange(-3, 4, dtype=torch.float, device=probs.device)
    
#     # Calculate expected value (weighted average)
#     continuous_preds = torch.sum(probs * class_values.unsqueeze(0), dim=1, keepdim=True)
    
#     return continuous_preds



def discrete_to_continuous_sentiment(discrete_labels):
    """
    Convert discrete class indices back to continuous sentiment values.
    Uses the midpoint of each custom boundary range for consistency.
    
    Custom ranges:
    - Class 0: [-3.0, -2.5) → midpoint: -2.75
    - Class 1: [-2.5, -1.5) → midpoint: -2.0
    - Class 2: [-1.5, -0.5) → midpoint: -1.0
    - Class 3: [-0.5, 0.5)  → midpoint: 0.0
    - Class 4: [0.5, 1.5)   → midpoint: 1.0
    - Class 5: [1.5, 2.5)   → midpoint: 2.0
    - Class 6: [2.5, 3.0]   → midpoint: 2.75
    
    Args:
        discrete_labels: Tensor of class indices [0-6]
    
    Returns:
        continuous_labels: Tensor of continuous values representing range midpoints
    """
    # Define the midpoint values for each class range
    class_midpoints = torch.tensor([-2.75, -2.0, -1.0, 0.0, 1.0, 2.0, 2.75], 
                                   dtype=torch.float, device=discrete_labels.device)
    
    # Map each discrete label to its corresponding midpoint
    continuous_labels = class_midpoints[discrete_labels]
    
    return continuous_labels


def predictions_to_continuous(class_predictions):
    """
    Convert classification predictions (logits or probabilities) to continuous values.
    Uses expected value calculation with custom boundary range midpoints.
    
    Args:
        class_predictions: Tensor of shape [batch_size, 7] (logits or probs)
    
    Returns:
        continuous_preds: Tensor of shape [batch_size, 1] with continuous values
    """
    # If logits, convert to probabilities
    if class_predictions.max() > 1.0 or class_predictions.min() < 0.0:
        probs = F.softmax(class_predictions, dim=1)
    else:
        probs = class_predictions
    
    # Use the same midpoint values as in discrete_to_continuous_sentiment
    # This ensures consistency between the two conversion approaches
    class_values = torch.tensor([-2.75, -2.0, -1.0, 0.0, 1.0, 2.0, 2.75], 
                                dtype=torch.float, device=probs.device)
    
    # Calculate expected value (weighted average) using custom range midpoints
    continuous_preds = torch.sum(probs * class_values.unsqueeze(0), dim=1, keepdim=True)
    
    return continuous_preds


# Optional: Helper function to get boundary information
def get_boundary_info():
    """
    Returns information about the custom boundary ranges for reference.
    
    Returns:
        Dictionary with boundary ranges and their midpoints
    """
    return {
        'ranges': {
            0: {'range': '[-3.0, -2.5)', 'label': 'Highly Negative', 'midpoint': -2.75},
            1: {'range': '[-2.5, -1.5)', 'label': 'Negative', 'midpoint': -2.0},
            2: {'range': '[-1.5, -0.5)', 'label': 'Weakly Negative', 'midpoint': -1.0},
            3: {'range': '[-0.5, 0.5)', 'label': 'Neutral', 'midpoint': 0.0},
            4: {'range': '[0.5, 1.5)', 'label': 'Weakly Positive', 'midpoint': 1.0},
            5: {'range': '[1.5, 2.5)', 'label': 'Positive', 'midpoint': 2.0},
            6: {'range': '[2.5, 3.0]', 'label': 'Highly Positive', 'midpoint': 2.75}
        },
        'midpoints': [-2.75, -2.0, -1.0, 0.0, 1.0, 2.0, 2.75]
    }
