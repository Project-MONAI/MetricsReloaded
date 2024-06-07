# MetricsReloaded quick-start guide

## Installation

ℹ️ Note that we use our [ivadomed fork](https://github.com/ivadomed/MetricsReloaded) of the MetricsReloaded repository.

⚠️ Always install MetricsReloaded inside a virtual environment:

```
# Create and activate a new conda environment
conda create -n metrics_reloaded python=3.10 pip
conda activate metrics_reloaded

# Clone the MetricsReloaded repository (ivadomed fork)
cd ~/code
git clone https://github.com/ivadomed/MetricsReloaded
cd MetricsReloaded

# Install the package
python -m pip install .
# You can alternatively install the package in editable mode:
python -m pip install -e .
```

The installation instructions are available also [here](https://github.com/ivadomed/MetricsReloaded?tab=readme-ov-file#installation).

## Usage

You can use the [compute_metrics_reloaded.py](compute_metrics_reloaded.py) wrapper script to compute metrics using the MetricsReloaded package.

Example usage:

```commandline
python compute_metrics_reloaded.py 
-reference sub-001_T2w_seg.nii.gz 
-prediction sub-001_T2w_prediction.nii.gz 
```

The metrics to be computed can be specified using the `-metrics` argument. For example, to compute only the Dice 
similarity coefficient (DSC) and Normalized surface distance (NSD), use:

```commandline
python compute_metrics_reloaded.py 
-reference sub-001_T2w_seg.nii.gz 
-prediction sub-001_T2w_prediction.nii.gz 
-metrics dsc nsd
```

ℹ️ See https://arxiv.org/abs/2206.01653v5 for nice figures explaining the metrics!

The output is saved to a CSV file, for example:

```csv
reference   prediction	label	dsc nsd	EmptyRef	EmptyPred
seg.nii.gz	pred.nii.gz	1.0	0.819	0.945   False	False
seg.nii.gz	pred.nii.gz	2.0	0.743	0.923   False	False
```

ℹ️ The script is compatible with both binary (voxels with label values `1`) and multi-class segmentations (voxels with 
label values `1`, `2`, etc.; e.g., nnunet region-based).

ℹ️ The metrics are computed for each unique label (class) in the reference (ground truth) image.

## Useful links:
- [MetricsReloaded documentation](https://metricsreloaded.readthedocs.io/en/latest/)
- [MetricsReloaded publication](https://www.nature.com/articles/s41592-023-02151-z)
- [MetricsReloaded preprint](https://arxiv.org/pdf/2206.01653v5.pdf) - preprint contains more figures than the publication
