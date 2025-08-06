#!/bin/bash

#SBATCH --job-name=Label_Conversion_Test
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G 
#SBATCH --time=1:00:00
#SBATCH --output=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm/0_MultiBench_2/label_conversion_test_%j.out
#SBATCH --error=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm/0_MultiBench_2/label_conversion_test_%j.err

# Project and source directories
SRCDIR=/project/ag-jafra/Souptik/CMG_New/Experiments
TESTDIR=$SRCDIR/CMG_trial1/src

# Create output directory for logs
mkdir -p $SRCDIR/CMG_trial1/slurm/0_MultiBench_2

# Activate conda environment
source ~/.bashrc
conda activate /project/ag-jafra/Souptik/CMG_New/Experiments/envs/CMG_new

# Print job information
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "Python Version: $(python --version)"
echo "Conda Environment: $CONDA_DEFAULT_ENV"
echo "Test Directory: $TESTDIR"
echo "========================================="

# Navigate to the test directory
cd $TESTDIR

# Check if required files exist
echo "Checking for required files..."
if [ ! -f "label_conversion_mosei.py" ]; then
    echo "WARNING: label_conversion_mosei.py not found in $TESTDIR"
    echo "Please ensure the label conversion functions are implemented"
fi

if [ ! -f "label_test.py" ]; then
    echo "ERROR: label_test.py not found in $TESTDIR"
    echo "Please ensure the test script is in the correct location"
    exit 1
fi

echo "All required files found. Starting label conversion tests..."
echo "========================================="

# Run the comprehensive label conversion test
echo "Running comprehensive label conversion tests..."
python label_test.py

# Capture exit status
TEST_STATUS=$?

echo "========================================="
if [ $TEST_STATUS -eq 0 ]; then
    echo "SUCCESS: All label conversion tests passed!"
    echo "Your label conversion implementation is working correctly."
    echo ""
    echo "Next steps:"
    echo "1. Review the test output above to understand the conversions"
    echo "2. Integrate the label_conversion.py into your training scripts"
    echo "3. Update your model architecture for classification output"
    echo "4. Test with a small training run to verify end-to-end functionality"
else
    echo "FAILURE: Label conversion tests failed!"
    echo "Please check the error messages above and fix the issues."
    echo ""
    echo "Common issues to check:"
    echo "1. Ensure label_conversion.py is properly implemented"
    echo "2. Check that all required PyTorch functions are available"
    echo "3. Verify tensor shapes and data types are correct"
    echo "4. Make sure the conversion logic handles edge cases properly"
fi

echo "End Time: $(date)"
echo "========================================="

# Additional system information for debugging
echo "Additional debugging information:"
echo "Python path: $PYTHONPATH"
echo "Current directory contents:"
ls -la
echo "Python import path test:"
python -c "import sys; print('\\n'.join(sys.path))"

exit $TEST_STATUS