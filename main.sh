#!/bin/bash
# Multi-Modal Medical Dataset Training Pipeline
# Supports: EEG (BCI, SEED, FACED), Ultrasound (BUSI), EEG-fMRI (SeizeIT), fMRI (OpenNeuro)

# ============================================================================
# DATASET DOWNLOADS
# ============================================================================

# Option 1: Download REAL datasets (requires internet and registration for some)
# ----------------------------------------------------------------------------
# for faced_dataset 
# bash download_faced_dataset.sh

# BCI dataset download
# bash download_bci_dataset.sh


# BCI Competition IV 2a - Motor Imagery (REQUIRES MANUAL DOWNLOAD)
# ⚠️ Auto-download doesn't work - direct links are disabled by the host
# Solution: Visit http://bnci-horizon-2020.eu/database/data-sets/001-2014/
#           and download 18 files manually (see BCI_DOWNLOAD_SOLUTION.md)
# python download_real_datasets.py --dataset bci --output-dir ../data

# SEED - Emotion EEG (requires manual registration at SJTU)
# python download_real_datasets.py --dataset seed --output-dir ../data

# FACED - Facial Expression and EEG (requires GitHub download)
# python download_real_datasets.py --dataset faced --output-dir ../data

# Show download instructions for all
# python download_real_datasets.py --dataset all --output-dir ../data


# Option 2: Generate SYNTHETIC datasets for testing (no download needed)
# ----------------------------------------------------------------------------

# Generate FACED synthetic data (5 subjects, 30 channels)
# python generate_datasets.py --faced-only --faced-subjects 5 --num-channel 30 --data-dir ../data

# Generate SEED synthetic data (15 subjects)
# python generate_datasets.py --seed-only --seed-subjects 15 --data-dir ../data

# Generate BCI synthetic data (9 subjects)
# python generate_datasets.py --generate-bci-synthetic --data-dir ../data

# Generate all synthetic datasets
# python generate_datasets.py --seed-subjects 15 --faced-subjects 5 --data-dir ../data

# ============================================================================
# MEDICAL IMAGING DATASETS DOWNLOAD
# ============================================================================

# BUSI - Breast Ultrasound (Easiest, ~180 MB)
# kaggle datasets download -d aryashah2k/breast-ultrasound-images-dataset
# unzip breast-ultrasound-images-dataset.zip -d ../data/BUSI

# OpenNeuro ds000030 - Task-based fMRI (~50-100 GB, subset recommended)
# aws s3 sync --no-sign-request s3://openneuro.org/ds000030 ../data/openneuro/ds000030

# SeizeIT1 - EEG-fMRI Seizure Data (~30-50 GB)
# aws s3 sync --no-sign-request s3://openneuro.org/ds004100 ../data/seizeit

# See DOWNLOAD_COMMANDS.md for detailed instructions

# ============================================================================
# TEST DATA LOADERS
# ============================================================================
#     --models eegnet,deepconvnet,shallowconvnet,mamba,spectral_mamba,nakul \
#     --data_dir data --epochs_override 40
#
# # Test loaders (✅ already passed!)
# python test_data_loaders.py

# # Train on BCI dataset
# python train_all_models.py --dataset bci --models nakul

# Train on SEED
# python train_all_models.py --dataset seed --models nakul


# # Train on all datasets
# python train_all_models.py --dataset all --models eegnet,nakul --window_size 2.0


# Full benchmark
python train_all_models.py \
    --dataset all --data_dir ../data \
    --models eegnet,deepconvnet,shallowconvnet,mamba,spectral_mamba,nakul \
    --batch_size 32 --epochs_override 50 --window_size 2.0



# Full benchmark
python train_all_models.py \
    --dataset bci --data_dir ../data \
    --models eegnet,deepconvnet,shallowconvnet,mamba,spectral_mamba,nakul \
    --batch_size 32 --epochs_override 50 --window_size 2.0


