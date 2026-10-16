# Skin Lesion Segmentation with VM-UNet — CS554
Graduate coursework at Özyeğin University, CS554 Introduction to Machine Learning and Artificial Neural Networks. The supplied proposal/report names Ahmet Mete Dokgoz, Emre Öztürk and Ali Baki Türköz. Detailed individual contribution proportions have not been confirmed.

## Purpose and work
The initial proposal describes skin-lesion classification; the implementation, report and presentations focus on **segmentation**: predicting a lesion mask for each image. The supplied archive contains a VM-UNet implementation, dataset/training variants, an Optuna search script, a report and a 10-trial result CSV. The slides describe ISIC 2016/2017/2018 images and masks, training/validation splitting and separate test images.

## Evidence and interpretation
The Optuna script minimizes checkpoint validation loss. The smallest stored objective is **0.3390127431490634**, trial **0**. This is a validation-loss search record, not test accuracy, Dice or IoU, and no new experiment was run. The draft report contains comparative and novelty claims that have not been independently verified; they are not adopted as portfolio conclusions. No trained checkpoint, actual ISIC dataset or independent test-run output was found in the supplied ZIPs.

## Contents
- [Archived search results](results/optuna_study_report.csv)
- [Archived course report](reports/archived_report.tex)
- [Original proposal/slides](reports/original/)
- [Upstream-derived implementation and course variants](third_party/vm-unet/)
- [Scope and provenance](docs/PROVENANCE.md)

## Attribution and execution
The implementation derives from VM-UNet and is not claimed as an architecture invented by this student team. Its original README, authorship and Apache-2.0 licence are preserved in `third_party/vm-unet/`. `train_new.py`, `dataset_new.py` and `hyperparameter_tuning.py` are present in the supplied course archive; without a pinned upstream comparison, exact authorship of each edit is not asserted.

Follow the bundled upstream README for environment/data setup and review course-specific configurations before execution. Dataset and pretrained weights must be obtained separately; no clinical diagnostic performance is claimed. This is a public academic portfolio repository.

## Setup and archive validation
[Environment, paths and execution scope](RUNNING.md). Run `python tools/check_archive.py` to verify the archive without training.
