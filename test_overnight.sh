#!/bin/bash
################################################################################
# Overnight Test Script for Bayesian Optimization
################################################################################
# 
# This script runs a test optimization with logging enabled.
# Useful for overnight runs where you want to review the output later.
#
################################################################################

# Create logs directory if it doesn't exist
mkdir -p logs

# Generate timestamp for log file
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
LOG_FILE="logs/bayesian_batch_test_${TIMESTAMP}.log"

echo "Starting Bayesian optimization batch test..."
echo "Log file: ${LOG_FILE}"
echo ""
echo "Command: ./optimize_bayesian_batch.sh --n_trials 20 --n_jobs 4"
echo ""
echo "Press Ctrl+C to cancel, or wait 5 seconds to continue..."
sleep 5

# Run the optimization and capture output
./optimize_bayesian_batch.sh --n_trials 20 --n_jobs 4 2>&1 | tee "${LOG_FILE}"

# Check exit code
EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "======================================================================"
echo "Optimization completed with exit code: ${EXIT_CODE}"
echo "Log saved to: ${LOG_FILE}"
echo "======================================================================"

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✓ All optimizations completed successfully!"
else
    echo "⚠ Some optimizations failed. Check the log for details."
fi

exit ${EXIT_CODE}
