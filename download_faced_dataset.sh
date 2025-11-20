#!/bin/bash
# Quick Start Script for FACED Dataset Preprocessing
# we have downloaded data from Download URL:
# https://www.synapse.org/#!Synapse:syn50614194/files/ 

# present in ../data/Processed_data.zip

# we need to do preprocessing using 

# https://torcheeg.readthedocs.io/en/stable/generated/torcheeg.datasets.FACEDDataset.html
# # ===================================================

set -e  # Exit on error

echo "=========================================="
echo "FACED Dataset Preprocessing Quick Start"
echo "=========================================="
echo ""

# Configuration
ZIP_FILE="../data/Processed_data.zip"
OUTPUT_DIR="../data/Processed_data"
CHUNK_SIZE=250      # 1 second at 250 Hz
OVERLAP=0           # No overlap
NORMALIZE="zscore"  # Z-score normalization

# Check if zip file exists
if [ ! -f "$ZIP_FILE" ]; then
    echo "❌ Error: $ZIP_FILE not found!"
    echo ""
    echo "Please download the dataset from:"
    echo "https://www.synapse.org/#!Synapse:syn50614194/files/"
    echo ""
    echo "Place it at: $ZIP_FILE"
    exit 1
fi

echo "✓ Found dataset: $ZIP_FILE"
FILE_SIZE=$(du -h "$ZIP_FILE" | cut -f1)
echo "  Size: $FILE_SIZE"
echo ""

# Check if already processed
if [ -d "$OUTPUT_DIR" ] && [ -f "$OUTPUT_DIR/train_data.npz" ]; then
    echo "⚠ Preprocessed data already exists at: $OUTPUT_DIR"
    echo ""
    read -p "Do you want to re-process? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping preprocessing. Using existing data."
        exit 0
    fi
    echo "Re-processing..."
    echo ""
fi

# Install dependencies if needed
echo "Checking dependencies..."
python3 -c "import numpy, scipy, torch, sklearn, tqdm" 2>/dev/null || {
    echo "Installing required packages..."
    pip install numpy scipy torch scikit-learn tqdm
}
echo "✓ Dependencies OK"
echo ""

# Run preprocessing
echo "Starting preprocessing..."
echo "Parameters:"
echo "  - Chunk size: $CHUNK_SIZE samples ($(echo "scale=2; $CHUNK_SIZE/250" | bc)s)"
echo "  - Overlap: $OVERLAP samples"
echo "  - Normalization: $NORMALIZE"
echo "  - Output: $OUTPUT_DIR"
echo ""

python3 preprocess_faced_data.py \
    --input "$ZIP_FILE" \
    --output "$OUTPUT_DIR" \
    --chunk-size $CHUNK_SIZE \
    --overlap $OVERLAP \
    --normalize $NORMALIZE

# Check if successful
if [ $? -eq 0 ] && [ -f "$OUTPUT_DIR/train_data.npz" ]; then
    echo ""
    echo "=========================================="
    echo "✅ Preprocessing Complete!"
    echo "=========================================="
    echo ""
    echo "Preprocessed data saved to: $OUTPUT_DIR"
    echo ""
    echo "Files created:"
    ls -lh "$OUTPUT_DIR"/*.npz 2>/dev/null | awk '{print "  - " $9 " (" $5 ")"}'
    ls -lh "$OUTPUT_DIR"/*.pkl 2>/dev/null | awk '{print "  - " $9 " (" $5 ")"}'
    echo ""
    echo "Next steps:"
    echo "  1. Load data in Python:"
    echo "     import numpy as np"
    echo "     data = np.load('$OUTPUT_DIR/train_data.npz')"
    echo "     X, y = data['X'], data['y']"
    echo ""
    echo "  2. Or use the data loader:"
    echo "     from faced_dataloader import FACEDDataLoader"
    echo "     loader = FACEDDataLoader(root_path='../data/Processed_data')"
    echo ""
    echo "  3. See FACED_PREPROCESSING_GUIDE.md for more details"
    echo ""
else
    echo ""
    echo "❌ Preprocessing failed!"
    echo "Check the error messages above."
    echo ""
    echo "Troubleshooting:"
    echo "  1. Verify zip file is not corrupted"
    echo "  2. Ensure sufficient disk space"
    echo "  3. Try with fewer subjects: --subjects 0 1 2"
    echo "  4. Inspect data: python preprocess_faced_data.py --inspect-only"
    exit 1
fi
