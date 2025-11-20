# Quick Reference Guide: Medical Imaging Data Loaders


## 🔗 Quick Links

- **Full Documentation:** `MEDICAL_LOADERS_README.md`
- **Test Script:** `python test_medical_loaders.py`
- **Download Script:** `./download_medical_datasets.sh`
- **Issue Tracker:** [GitHub Issues](#)

---

## 📞 Get Help

```bash
# Test specific loader
python test_medical_loaders.py --loader busi

# Run benchmark
python test_medical_loaders.py --benchmark

# Check installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

---

## ✅ Checklist

Before training:
- [ ] Downloaded dataset
- [ ] Tested loader with `test_medical_loaders.py`
- [ ] Verified batch shapes
- [ ] Checked class distribution
- [ ] Set appropriate batch size for GPU memory
- [ ] Enabled multi-worker loading

---

## 🎓 Learn More

- OpenNeuro: https://openneuro.org/
- TCIA: https://www.cancerimagingarchive.net/
- Kaggle Datasets: https://www.kaggle.com/datasets
- PyTorch DataLoader: https://pytorch.org/docs/stable/data.html

---

**Last Updated:** November 2025  
**Version:** 1.0.0
