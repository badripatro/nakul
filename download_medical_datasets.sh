#!/bin/bash
# Master Download Script for Medical Imaging Datasets
# ====================================================
# This script provides automated downloads for all 5 medical imaging datasets.

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Base data directory
DATA_DIR="../data"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Medical Imaging Dataset Downloader${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Function to print status
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[i]${NC} $1"
}

# Create data directory
mkdir -p "$DATA_DIR"

# ============================================================================
# 1. OpenNeuro Dataset (ds000030) - fMRI for Depression Tracking
# ============================================================================
download_openneuro() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}1. OpenNeuro ds000030 - fMRI (Depression Tracking)${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    OPENNEURO_DIR="$DATA_DIR/openneuro/ds000030"
    
    if [ -d "$OPENNEURO_DIR" ] && [ "$(ls -A $OPENNEURO_DIR)" ]; then
        print_status "OpenNeuro dataset already exists"
        return 0
    fi
    
    print_info "Downloading OpenNeuro ds000030..."
    print_info "Method 1: AWS CLI (No authentication required)"
    echo "  aws s3 sync --no-sign-request s3://openneuro.org/ds000030 $OPENNEURO_DIR"
    
    print_info "Method 2: DataLad"
    echo "  datalad install https://github.com/OpenNeuroDatasets/ds000030"
    
    print_warning "Dataset size: ~50-100 GB"
    print_warning "Please run one of the commands above to download"
    
    # Create placeholder
    mkdir -p "$OPENNEURO_DIR"
    echo "Download manually using AWS CLI or DataLad" > "$OPENNEURO_DIR/README.txt"
}

# ============================================================================
# 2. SeizeIT1 (SzCORE) - EEG-fMRI for Seizure Prediction
# ============================================================================
download_seizeit() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}2. SeizeIT1 (SzCORE) - EEG-fMRI (Seizure Prediction)${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    SEIZEIT_DIR="$DATA_DIR/seizeit"
    
    if [ -d "$SEIZEIT_DIR" ] && [ "$(ls -A $SEIZEIT_DIR)" ]; then
        print_status "SeizeIT1 dataset already exists"
        return 0
    fi
    
    print_info "Downloading SeizeIT1 dataset..."
    print_info "Access via OpenNeuro:"
    echo "  URL: https://openneuro.org/datasets/ds004100"
    echo "  aws s3 sync --no-sign-request s3://openneuro.org/ds004100 $SEIZEIT_DIR"
    
    print_warning "Dataset size: ~30-50 GB"
    print_warning "Requires OpenNeuro/DataLad access"
    
    # Create placeholder
    mkdir -p "$SEIZEIT_DIR"
    echo "Download from OpenNeuro: https://openneuro.org/datasets/ds004100" > "$SEIZEIT_DIR/README.txt"
}

# ============================================================================
# 3. LIDC-IDRI - Lung CT for Cancer Staging
# ============================================================================
download_lidc() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}3. LIDC-IDRI - Lung CT (Cancer Staging)${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    LIDC_DIR="$DATA_DIR/LIDC-IDRI"
    
    if [ -d "$LIDC_DIR" ] && [ "$(ls -A $LIDC_DIR)" ]; then
        print_status "LIDC-IDRI dataset already exists"
        return 0
    fi
    
    print_info "Downloading LIDC-IDRI dataset..."
    print_info "Method 1: NBIA Data Retriever (Recommended)"
    echo "  1. Download NBIA Data Retriever from:"
    echo "     https://wiki.cancerimagingarchive.net/x/2QKPAQ"
    echo "  2. Download manifest from:"
    echo "     https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI"
    echo "  3. Use Data Retriever to download"
    
    print_info "Method 2: Python tcia-utils"
    echo "  pip install tcia-utils"
    echo "  # Use Python script to download"
    
    print_warning "Dataset size: ~124 GB (1,018 cases)"
    print_warning "Requires TCIA account (free)"
    
    # Create placeholder
    mkdir -p "$LIDC_DIR"
    echo "Download using NBIA Data Retriever or tcia-utils" > "$LIDC_DIR/README.txt"
}

# ============================================================================
# 4. BUSI - Breast Ultrasound for Tumor Classification
# ============================================================================
download_busi() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}4. BUSI - Breast Ultrasound (Tumor Classification)${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    BUSI_DIR="$DATA_DIR/BUSI"
    
    if [ -d "$BUSI_DIR" ] && [ "$(ls -A $BUSI_DIR)" ]; then
        print_status "BUSI dataset already exists"
        return 0
    fi
    
    print_info "Downloading BUSI dataset..."
    
    # Check if kaggle CLI is available
    if command -v kaggle &> /dev/null; then
        print_status "Kaggle CLI found"
        print_info "Attempting download..."
        
        # Download using Kaggle API
        kaggle datasets download -d aryashah2k/breast-ultrasound-images-dataset -p "$DATA_DIR"
        
        # Unzip
        if [ -f "$DATA_DIR/breast-ultrasound-images-dataset.zip" ]; then
            print_info "Extracting dataset..."
            unzip -q "$DATA_DIR/breast-ultrasound-images-dataset.zip" -d "$BUSI_DIR"
            rm "$DATA_DIR/breast-ultrasound-images-dataset.zip"
            print_status "BUSI dataset downloaded and extracted"
        fi
    else
        print_warning "Kaggle CLI not found"
        print_info "Manual download instructions:"
        echo "  1. Install Kaggle CLI: pip install kaggle"
        echo "  2. Setup credentials:"
        echo "     - Go to https://www.kaggle.com/settings"
        echo "     - Create new API token (downloads kaggle.json)"
        echo "     - Move to ~/.kaggle/kaggle.json"
        echo "     - chmod 600 ~/.kaggle/kaggle.json"
        echo "  3. Download:"
        echo "     kaggle datasets download -d aryashah2k/breast-ultrasound-images-dataset"
        
        # Create placeholder
        mkdir -p "$BUSI_DIR"
        echo "Download from Kaggle using API or web interface" > "$BUSI_DIR/README.txt"
    fi
    
    print_warning "Dataset size: ~180 MB (780 images)"
}

# ============================================================================
# 5. POCUS - Point-of-Care Ultrasound for Cardiac Staging
# ============================================================================
download_pocus() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}5. POCUS - Ultrasound (Cardiac Function Staging)${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    POCUS_DIR="$DATA_DIR/POCUS"
    
    if [ -d "$POCUS_DIR" ] && [ "$(ls -A $POCUS_DIR)" ]; then
        print_status "POCUS dataset already exists"
        return 0
    fi
    
    print_info "Downloading POCUS dataset..."
    print_info "Access instructions:"
    echo "  1. Stanford AIMI Shared Datasets:"
    echo "     https://stanfordaimi.azurewebsites.net/datasets/"
    echo "  2. Some POCUS datasets on PhysioNet:"
    echo "     https://physionet.org/"
    echo "  3. OpenICPSR may have relevant datasets"
    
    print_warning "Dataset size: Variable (depends on specific POCUS dataset)"
    print_warning "May require institutional access or data use agreement"
    
    # Create placeholder
    mkdir -p "$POCUS_DIR"
    echo "Access via Stanford AIMI or PhysioNet" > "$POCUS_DIR/README.txt"
}

# ============================================================================
# Main execution
# ============================================================================

# Parse command line arguments
if [ $# -eq 0 ]; then
    # Download all datasets
    download_openneuro
    download_seizeit
    download_lidc
    download_busi
    download_pocus
else
    # Download specific datasets
    for arg in "$@"; do
        case $arg in
            openneuro|1)
                download_openneuro
                ;;
            seizeit|2)
                download_seizeit
                ;;
            lidc|3)
                download_lidc
                ;;
            busi|4)
                download_busi
                ;;
            pocus|5)
                download_pocus
                ;;
            *)
                print_error "Unknown dataset: $arg"
                echo "Available datasets: openneuro, seizeit, lidc, busi, pocus"
                echo "Or use numbers: 1, 2, 3, 4, 5"
                exit 1
                ;;
        esac
    done
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Download script completed!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${YELLOW}Note:${NC} Some datasets require manual download due to:"
echo "  - Large file sizes"
echo "  - Authentication requirements"
echo "  - Data use agreements"
echo ""
echo "Please check the README.txt files in each dataset directory"
echo "for specific download instructions."
echo ""
print_info "Data directory: $DATA_DIR"
