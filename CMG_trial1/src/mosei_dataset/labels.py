import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

def multiclass_acc(preds, truths):
    """
    Compute the multiclass accuracy w.r.t. groundtruth.
    
    :param preds: Float array representing the predictions, dimension (N,)
    :param truths: Float/int array representing the groundtruth classes, dimension (N,)
    :return: Classification accuracy
    """
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))

def eval_mosei_senti_return(results, truths, exclude_zero=False):
    """Evaluate MOSEI and return metric list.

    Args:
        results (torch.Tensor): Predicted values with shape [batch_size, 1]
        truths (torch.Tensor): True values with shape [batch_size, 1]
        exclude_zero (bool, optional): Whether to exclude zero. Defaults to False.

    Returns:
        tuple(mae, corr, mult_a7, f_score, accuracy): Return statistics for MOSEI.
    """
    # Step 1: Flatten tensors to 1D numpy arrays
    test_preds = results.view(-1).cpu().detach().numpy()
    test_truth = truths.view(-1).cpu().detach().numpy()
    
    # Step 2: Create index for non-zero samples (if needed)
    non_zeros = np.array([i for i, e in enumerate(test_truth) 
                         if e != 0 or (not exclude_zero)])
    
    # Step 3: Create different accuracy versions by clipping
    test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
    test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
    test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
    test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)
    
    # Step 4: Compute all metrics
    
    # A. Mean Absolute Error
    mae = np.mean(np.absolute(test_preds - test_truth))
    
    # B. Correlation Coefficient
    corr = np.corrcoef(test_preds, test_truth)[0][1]
    
    # C. 7-Class Accuracy (round to integers in [-3, +3])
    mult_a7 = multiclass_acc(test_preds_a7, test_truth_a7)
    
    # D. 5-Class Accuracy (round to integers in [-2, +2])
    mult_a5 = multiclass_acc(test_preds_a5, test_truth_a5)
    
    # E. Binary F1 Score (positive vs negative/neutral)
    f_score = f1_score((test_preds[non_zeros] > 0),
                       (test_truth[non_zeros] > 0), average='weighted')
    
    # F. Binary Accuracy (positive vs negative/neutral)
    binary_truth = (test_truth[non_zeros] > 0)
    binary_preds = (test_preds[non_zeros] > 0)
    binary_accuracy = accuracy_score(binary_truth, binary_preds)
    
    return mae, corr, mult_a7, mult_a5, f_score, binary_accuracy

# Example usage with your actual MOSEI data format
if __name__ == "__main__":
    # Your actual ground truth labels (from your output)
    ground_truth = torch.tensor([
        [2.0000], [-0.3333], [1.0000], [1.3333], [0.6667], [-0.6667], 
        [0.3333], [0.0000], [0.3333], [-1.6667], [1.6667], [0.6667]
    ])
    
    # Example model predictions (same shape as ground truth)
    model_predictions = torch.tensor([
        [1.8], [-0.1], [0.9], [1.4], [0.8], [-0.8], 
        [0.4], [0.1], [0.4], [-1.5], [1.5], [0.5]
    ])
    
    print("Input shapes:")
    print(f"Ground truth: {ground_truth.shape}")
    print(f"Predictions: {model_predictions.shape}")
    print()
    
    # Compute all metrics
    mae, corr, mult_a7, mult_a5, f_score, binary_acc = eval_mosei_senti_return(
        model_predictions, ground_truth, exclude_zero=False
    )
    
    print("=== EVALUATION RESULTS ===")
    print(f"MAE (Mean Absolute Error): {mae:.4f}")
    print(f"Correlation Coefficient: {corr:.4f}")
    print(f"7-Class Accuracy (mult_a7): {mult_a7:.4f}")
    print(f"5-Class Accuracy (mult_a5): {mult_a5:.4f}")
    print(f"Binary F1 Score: {f_score:.4f}")
    print(f"Binary Accuracy: {binary_acc:.4f}")
    print()
    
    # Let's trace through the step-by-step calculation for better understanding
    print("=== STEP-BY-STEP BREAKDOWN ===")
    
    # Step 1: Flatten to 1D
    test_preds = model_predictions.view(-1).numpy()
    test_truth = ground_truth.view(-1).numpy()
    print(f"Flattened predictions: {test_preds}")
    print(f"Flattened ground truth: {test_truth}")
    print()
    
    # Step 2: 7-class accuracy breakdown
    test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
    test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
    
    rounded_preds = np.round(test_preds_a7)
    rounded_truth = np.round(test_truth_a7)
    
    print("7-Class Accuracy Breakdown:")
    print(f"Clipped predictions: {test_preds_a7}")
    print(f"Clipped ground truth: {test_truth_a7}")
    print(f"Rounded predictions: {rounded_preds}")
    print(f"Rounded ground truth: {rounded_truth}")
    print(f"Matches: {rounded_preds == rounded_truth}")
    print(f"Accuracy: {np.sum(rounded_preds == rounded_truth)} / {len(rounded_truth)} = {mult_a7:.4f}")
    print()
    
    # Step 3: Binary accuracy breakdown
    binary_preds = test_preds > 0
    binary_truth = test_truth > 0
    
    print("Binary Accuracy Breakdown:")
    print(f"Predictions > 0: {binary_preds}")
    print(f"Ground truth > 0: {binary_truth}")
    print(f"Matches: {binary_preds == binary_truth}")
    print(f"Binary accuracy: {np.sum(binary_preds == binary_truth)} / {len(binary_truth)} = {np.sum(binary_preds == binary_truth) / len(binary_truth):.4f}")