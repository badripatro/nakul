#!/bin/bash
# Quick BCI Dataset Download - One-Liner Commands
# Copy and paste these commands into your terminal

# Create output directory
mkdir -p ../data/BCI
cd ../data/BCI

echo "Downloading BCI Competition IV Dataset 2a..."
echo "Trying GDF format..."

# Download all files (GDF format)
for i in 01 02 03 04 05 06 07 08 09; do
    echo "Subject A$i..."
    wget -c https://bnci-horizon-2020.eu/database/data-sets/001-2014/A${i}T.gdf 2>&1 | grep -E "(saved|failed|404)"
    wget -c https://bnci-horizon-2020.eu/database/data-sets/001-2014/A${i}E.gdf 2>&1 | grep -E "(saved|failed|404)"
done

echo ""
echo "Download complete. Checking files..."
count=$(ls -1 *.gdf 2>/dev/null | wc -l)
echo "Files downloaded: $count / 18"

if [ $count -eq 0 ]; then
    echo ""
    echo "GDF download failed. Trying MAT format..."
    for i in 01 02 03 04 05 06 07 08 09; do
        echo "Subject A$i..."
        wget -c https://bnci-horizon-2020.eu/database/data-sets/001-2014/A${i}T.mat 2>&1 | grep -E "(saved|failed|404)"
        wget -c https://bnci-horizon-2020.eu/database/data-sets/001-2014/A${i}E.mat 2>&1 | grep -E "(saved|failed|404)"
    done
    
    count=$(ls -1 *.mat 2>/dev/null | wc -l)
    echo "MAT files downloaded: $count / 18"
fi

if [ $count -eq 18 ]; then
    echo ""
    echo "✓ Success! All files downloaded."
    echo ""
    echo "Verify with:"
    echo "  ls -lh *.gdf | wc -l  # or *.mat"
elif [ $count -gt 0 ]; then
    echo ""
    echo "⚠ Partial success: $count / 18 files downloaded"
    echo ""
    echo "Missing files may need manual download from:"
    echo "  http://bnci-horizon-2020.eu/database/data-sets/001-2014/"
else
    echo ""
    echo "✗ Automated download failed."
    echo ""
    echo "Please download manually:"
    echo "  1. Visit: http://bnci-horizon-2020.eu/database/data-sets/001-2014/"
    echo "  2. Click each file to download"
    echo "  3. Save to: $(pwd)"
fi
