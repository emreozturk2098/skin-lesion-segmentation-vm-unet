# Running and verifying the archive
## Verify the supplied materials without training
From the repository root, run:
```bash
python tools/check_archive.py
```
This uses only Python's standard library, parses Python/notebook files and checks the supplemental-source checksums. It does not import model libraries, scrape sites, fit models or establish reproduction of published metrics.

## Project workflow and remaining dependencies
The archived upstream README in third_party/vm-unet documents its original Python 3.8, PyTorch/CUDA and Mamba/selective-scan environment. Follow that source setup in a separate compatible Linux/CUDA environment, then review course-specific config_setting, train_new.py, dataset_new.py and hyperparameter_tuning.py. They do not become Windows/CPU-compatible merely by installing a generic requirements list. ISIC images/masks and pretrained weights must be obtained separately according to the upstream setup. Do not substitute the upstream 7:3 split for the coursework split and assume it reproduces the archived course experiment. The recorded Optuna objective is validation loss, not a test accuracy. No CUDA training was executed for this packaging update.

## Evidence and scope
Original source archives and results are preserved. Packaging validation is separate from running the research. Missing original code, external data or hardware dependencies are documented rather than replaced with invented implementations.
