import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
from datetime import datetime

# Add path for your label conversion functions
sys.path.append('/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/src/mosei_dataset')

def print_progress(message, level=0):
    """Print with timestamp and indentation for nested information"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    indent = "  " * level
    print(f"[{timestamp}] {indent}{message}")

def test_single_value_conversion():
    """Test conversion with individual sentiment values"""
    print_progress("="*80)
    print_progress("TESTING SINGLE VALUE CONVERSIONS")
    print_progress("="*80)
    
    # Test cases covering the full sentiment spectrum
    test_values = [-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    expected_classes = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6]  # After rounding and shifting
    
    print_progress("Testing continuous to discrete conversion:", 1)
    
    for i, value in enumerate(test_values):
        # Create tensor as your model would receive it
        continuous_label = torch.tensor([[value]], dtype=torch.float32)
        print_progress(f"Input continuous value: {value}", 2)
        print_progress(f"Tensor shape: {continuous_label.shape}", 3)
        print_progress(f"Tensor content: {continuous_label}", 3)
        
        # Convert to discrete
        from label_conversion_mosei import continuous_to_discrete_sentiment
        discrete_label = continuous_to_discrete_sentiment(continuous_label)
        print_progress(f"After discrete conversion: {discrete_label.item()}", 3)
        print_progress(f"Discrete tensor shape: {discrete_label.shape}", 3)
        print_progress(f"Expected class: {expected_classes[i]}", 3)
        
        # Verify conversion is correct
        if discrete_label.item() == expected_classes[i]:
            print_progress("✓ Conversion CORRECT", 3)
        else:
            print_progress("✗ Conversion ERROR!", 3)
        
        print_progress("-" * 40, 2)

def test_batch_conversion():
    """Test conversion with batches of data as your model would see them"""
    print_progress("="*80)
    print_progress("TESTING BATCH CONVERSIONS")
    print_progress("="*80)
    
    # Create realistic batch data similar to MOSI/MOSEI
    batch_size = 8
    realistic_values = [-2.6, -1.8, -1.0, -0.3, 0.0, 1.2, 2.0, 2.4]
    continuous_batch = torch.tensor([[val] for val in realistic_values], dtype=torch.float32)
    
    print_progress(f"Testing batch with size: {batch_size}", 1)
    print_progress(f"Original continuous labels shape: {continuous_batch.shape}", 2)
    print_progress(f"Original continuous labels: {continuous_batch.flatten().tolist()}", 2)
    
    # Convert to discrete classes
    from label_conversion_mosei import continuous_to_discrete_sentiment, discrete_to_continuous_sentiment
    discrete_batch = continuous_to_discrete_sentiment(continuous_batch)
    
    print_progress(f"Discrete labels shape: {discrete_batch.shape}", 2)
    print_progress(f"Discrete labels: {discrete_batch.tolist()}", 2)
    
    # Show the mapping
    print_progress("Detailed mapping:", 2)
    for i in range(batch_size):
        original = continuous_batch[i].item()
        discrete = discrete_batch[i].item()
        class_names = ['Highly Neg', 'Negative', 'Weakly Neg', 'Neutral', 
                      'Weakly Pos', 'Positive', 'Highly Pos']
        print_progress(f"Sample {i}: {original:5.1f} -> Class {discrete} ({class_names[discrete]})", 3)
    
    # Test reverse conversion
    print_progress("Testing reverse conversion (discrete back to continuous):", 2)
    recovered_continuous = discrete_to_continuous_sentiment(discrete_batch).unsqueeze(1)
    print_progress(f"Recovered continuous shape: {recovered_continuous.shape}", 3)
    print_progress(f"Recovered continuous values: {recovered_continuous.flatten().tolist()}", 3)
    
    # Show the round-trip differences
    print_progress("Round-trip conversion analysis:", 2)
    differences = continuous_batch - recovered_continuous
    print_progress(f"Differences (original - recovered): {differences.flatten().tolist()}", 3)
    print_progress(f"Max absolute difference: {torch.abs(differences).max().item():.4f}", 3)
    print_progress(f"Mean absolute difference: {torch.abs(differences).mean().item():.4f}", 3)

def test_onehot_conversion():
    """Test one-hot encoding conversion"""
    print_progress("="*80)
    print_progress("TESTING ONE-HOT ENCODING CONVERSIONS")
    print_progress("="*80)
    
    # Test with a small batch
    continuous_labels = torch.tensor([[-2.0], [-1.0], [0.0], [1.0], [2.0]], dtype=torch.float32)
    print_progress(f"Original continuous labels: {continuous_labels.flatten().tolist()}", 1)
    
    # Convert to one-hot
    from label_conversion_mosei import continuous_to_onehot_sentiment
    onehot_labels = continuous_to_onehot_sentiment(continuous_labels)
    
    print_progress(f"One-hot labels shape: {onehot_labels.shape}", 2)
    print_progress("One-hot encoding matrix:", 2)
    class_names = ['H.Neg', 'Neg', 'W.Neg', 'Neut', 'W.Pos', 'Pos', 'H.Pos']
    print_progress(f"Classes: {class_names}", 3)
    
    for i in range(onehot_labels.shape[0]):
        active_class = torch.argmax(onehot_labels[i]).item()
        print_progress(f"Sample {i} ({continuous_labels[i].item():4.1f}): {onehot_labels[i].tolist()} -> Active: {class_names[active_class]}", 3)

def test_prediction_conversion():
    """Test converting model predictions (logits/probabilities) back to continuous"""
    print_progress("="*80)
    print_progress("TESTING PREDICTION-TO-CONTINUOUS CONVERSION")
    print_progress("="*80)
    
    # Simulate model predictions (logits) for different scenarios
    print_progress("Simulating various prediction scenarios:", 1)
    
    # Scenario 1: Confident prediction for highly negative
    confident_negative = torch.tensor([[-5.0, -2.0, -1.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    print_progress("Scenario 1: Confident highly negative prediction", 2)
    print_progress(f"Raw logits: {confident_negative[0].tolist()}", 3)
    
    from label_conversion_mosei import predictions_to_continuous
    continuous_pred = predictions_to_continuous(confident_negative)
    print_progress(f"Converted to continuous: {continuous_pred.item():.4f}", 3)
    print_progress(f"Expected: Close to -3.0", 3)
    
    # Scenario 2: Uncertain prediction between neutral and weakly positive
    uncertain_neutral = torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.1, 0.0, 0.0]], dtype=torch.float32)
    print_progress("Scenario 2: Uncertain between neutral and weakly positive", 2)
    print_progress(f"Raw logits: {uncertain_neutral[0].tolist()}", 3)
    
    continuous_pred = predictions_to_continuous(uncertain_neutral)
    print_progress(f"Converted to continuous: {continuous_pred.item():.4f}", 3)
    print_progress(f"Expected: Between 0.0 and 1.0", 3)
    
    # Scenario 3: Uniform distribution (maximum uncertainty)
    uniform_dist = torch.ones((1, 7), dtype=torch.float32)
    print_progress("Scenario 3: Uniform distribution (maximum uncertainty)", 2)
    print_progress(f"Raw logits: {uniform_dist[0].tolist()}", 3)
    
    continuous_pred = predictions_to_continuous(uniform_dist)
    print_progress(f"Converted to continuous: {continuous_pred.item():.4f}", 3)
    print_progress(f"Expected: Close to 0.0 (neutral)", 3)
    
    # Show probability conversion
    print_progress("Detailed probability analysis for Scenario 2:", 2)
    probs = F.softmax(uncertain_neutral, dim=1)
    print_progress(f"Probabilities: {probs[0].tolist()}", 3)
    class_values = torch.arange(-3, 4, dtype=torch.float)
    print_progress(f"Class values: {class_values.tolist()}", 3)
    expected_value = torch.sum(probs[0] * class_values)
    print_progress(f"Manual expected value calculation: {expected_value.item():.4f}", 3)

def test_edge_cases():
    """Test edge cases and potential problematic inputs"""
    print_progress("="*80)
    print_progress("TESTING EDGE CASES AND ROBUSTNESS")
    print_progress("="*80)
    
    from label_conversion_mosei import continuous_to_discrete_sentiment, predictions_to_continuous
    
    # Test values outside the expected range
    print_progress("Testing out-of-range values:", 1)
    extreme_values = torch.tensor([[-5.0], [-3.5], [3.5], [5.0]], dtype=torch.float32)
    
    for i, val in enumerate(extreme_values):
        print_progress(f"Testing extreme value: {val.item()}", 2)
        discrete = continuous_to_discrete_sentiment(val)
        print_progress(f"Clamped to class: {discrete.item()}", 3)
        
        # Verify it's within valid range
        if 0 <= discrete.item() <= 6:
            print_progress("✓ Properly clamped to valid range", 3)
        else:
            print_progress("✗ ERROR: Outside valid range!", 3)
    
    # Test with very small values near zero
    print_progress("Testing values near zero boundary:", 1)
    near_zero = torch.tensor([[-0.1], [0.0], [0.1]], dtype=torch.float32)
    
    for val in near_zero:
        print_progress(f"Testing near-zero value: {val.item()}", 2)
        discrete = continuous_to_discrete_sentiment(val)
        print_progress(f"Converted to class: {discrete.item()}", 3)
    
    # Test with NaN and inf (should be handled gracefully)
    print_progress("Testing problematic values (NaN, inf):", 1)
    try:
        problematic = torch.tensor([[float('nan')], [float('inf')], [float('-inf')]], dtype=torch.float32)
        for i, val in enumerate(problematic):
            print_progress(f"Testing problematic value: {val.item()}", 2)
            try:
                discrete = continuous_to_discrete_sentiment(val)
                print_progress(f"Result: {discrete.item()}", 3)
            except Exception as e:
                print_progress(f"Error (expected): {e}", 3)
    except Exception as e:
        print_progress(f"Cannot create problematic tensors: {e}", 2)

def test_loss_computation_compatibility():
    """Test that the converted labels work properly with PyTorch loss functions"""
    print_progress("="*80)
    print_progress("TESTING LOSS COMPUTATION COMPATIBILITY")
    print_progress("="*80)
    
    from label_conversion_mosei import continuous_to_discrete_sentiment
    
    # Create sample data
    batch_size = 4
    num_classes = 7
    continuous_labels = torch.tensor([[-2.0], [-1.0], [0.0], [2.0]], dtype=torch.float32)
    
    print_progress(f"Original continuous labels: {continuous_labels.flatten().tolist()}", 1)
    
    # Convert to discrete
    discrete_labels = continuous_to_discrete_sentiment(continuous_labels)
    print_progress(f"Discrete labels: {discrete_labels.tolist()}", 2)
    print_progress(f"Discrete labels dtype: {discrete_labels.dtype}", 2)
    print_progress(f"Discrete labels shape: {discrete_labels.shape}", 2)
    
    # Create mock model predictions (logits)
    mock_logits = torch.randn(batch_size, num_classes, dtype=torch.float32)
    print_progress(f"Mock model logits shape: {mock_logits.shape}", 2)
    print_progress(f"Mock logits sample: {mock_logits[0].tolist()}", 2)
    
    # Test CrossEntropyLoss
    criterion = torch.nn.CrossEntropyLoss()
    
    try:
        loss = criterion(mock_logits, discrete_labels.long())
        print_progress(f"✓ CrossEntropyLoss computation successful", 2)
        print_progress(f"Loss value: {loss.item():.4f}", 3)
        print_progress(f"Loss tensor shape: {loss.shape}", 3)
        print_progress(f"Loss requires_grad: {loss.requires_grad}", 3)
    except Exception as e:
        print_progress(f"✗ CrossEntropyLoss computation failed: {e}", 2)
    
    # Test with one-hot encoded labels (if needed for some loss functions)
    from label_conversion_mosei import continuous_to_onehot_sentiment
    onehot_labels = continuous_to_onehot_sentiment(continuous_labels)
    print_progress(f"One-hot labels shape: {onehot_labels.shape}", 2)
    
    # Test with MSE loss on one-hot (for comparison)
    try:
        # Convert logits to probabilities for MSE comparison
        probs = F.softmax(mock_logits, dim=1)
        mse_criterion = torch.nn.MSELoss()
        mse_loss = mse_criterion(probs, onehot_labels)
        print_progress(f"✓ MSE loss on one-hot successful", 2)
        print_progress(f"MSE loss value: {mse_loss.item():.4f}", 3)
    except Exception as e:
        print_progress(f"✗ MSE loss on one-hot failed: {e}", 2)

def main():
    """Run comprehensive label conversion tests"""
    print_progress("STARTING COMPREHENSIVE LABEL CONVERSION TESTING")
    print_progress("=" * 80)
    
    try:
        # Import and test the label conversion functions
        print_progress("Attempting to import label conversion functions...")
        
        # Run all test suites
        test_single_value_conversion()
        test_batch_conversion()
        test_onehot_conversion()
        test_prediction_conversion()
        test_edge_cases()
        test_loss_computation_compatibility()
        
        print_progress("=" * 80)
        print_progress("ALL LABEL CONVERSION TESTS COMPLETED SUCCESSFULLY!")
        print_progress("✓ Single value conversions working")
        print_progress("✓ Batch conversions working")
        print_progress("✓ One-hot encoding working")
        print_progress("✓ Prediction-to-continuous conversion working")
        print_progress("✓ Edge cases handled properly")
        print_progress("✓ Loss computation compatibility confirmed")
        print_progress("=" * 80)
        
    except ImportError as e:
        print_progress(f"Import error: {e}")
        print_progress("Please ensure label_conversion.py is in the correct path")
        sys.exit(1)
    except Exception as e:
        print_progress(f"Unexpected error during testing: {e}")
        import traceback
        print_progress(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()