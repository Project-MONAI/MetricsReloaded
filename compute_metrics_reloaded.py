"""
Compute MetricsReloaded metrics for segmentation tasks.

For MetricsReloaded installation and usage of this script, see:
https://github.com/ivadomed/utilities/blob/main/quick_start_guides/MetricsReloaded_quick_start_guide.md

Example usage (single reference-prediction pair):
    python compute_metrics_reloaded.py
        -reference sub-001_T2w_seg.nii.gz
        -prediction sub-001_T2w_prediction.nii.gz

Example usage (multiple reference-prediction pairs, e.g., multiple subjects in the dataset):
    python compute_metrics_reloaded.py
        -reference /path/to/reference
        -prediction /path/to/prediction
NOTE: The prediction and reference files are matched based on the participant_id, acq_id, and run_id.

The metrics to be computed can be specified using the `-metrics` argument. For example, to compute only the Dice
similarity coefficient (DSC) and Normalized surface distance (NSD), use:
    python compute_metrics_reloaded.py
        -reference sub-001_T2w_seg.nii.gz
        -prediction sub-001_T2w_prediction.nii.gz
        -metrics dsc nsd

See https://arxiv.org/abs/2206.01653v5 for nice figures explaining the metrics!

The output is saved to a CSV file, for example:

reference   prediction	label	dsc nsd	EmptyRef	EmptyPred
seg.nii.gz	pred.nii.gz	1.0	0.819	0.945   False	False
seg.nii.gz	pred.nii.gz	2.0	0.743	0.923   False	False

The script is compatible with both binary and multi-class segmentation tasks (e.g., nnunet region-based).
The metrics are computed for each unique label (class) in the reference (ground truth) image.

Authors: Jan Valosek, Naga Karthik
"""


import os
import re
import argparse
import numpy as np
import nibabel as nib
import pandas as pd
from multiprocessing import Pool, cpu_count
from functools import partial

from MetricsReloaded.metrics.pairwise_measures import BinaryPairwiseMeasures as BPM


# This dictionary is used to rename the metric columns in the output CSV file
METRICS_TO_NAME = {
    'dsc': 'DiceSimilarityCoefficient',
    'hd': 'HausdorffDistance95',
    'fbeta': 'F1score',
    'nsd': 'NormalizedSurfaceDistance',
    'vol_diff': 'VolumeDifference',
    'rel_vol_error': 'RelativeVolumeError',
    'lesion_ppv': 'LesionWisePositivePredictiveValue',
    'lesion_sensitivity': 'LesionWiseSensitivity',
    'lesion_f1_score': 'LesionWiseF1Score',
    'ref_count': 'RefLesionsCount',
    'pred_count': 'PredLesionsCount',
    'lcwa': 'LesionCountWeightedByAssignment'
}


def get_parser():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Compute MetricsReloaded metrics for segmentation tasks.')

    # Arguments for model, data, and training
    parser.add_argument('-prediction', required=True, type=str,
                        help='Path to the folder with nifti images of test predictions or path to a single nifti image '
                             'of test prediction.')
    parser.add_argument('-reference', required=True, type=str,
                        help='Path to the folder with nifti images of reference (ground truth) or path to a single '
                             'nifti image of reference (ground truth).')
    parser.add_argument('-metrics', nargs='+', required=False,
                        default=['dsc', 'fbeta', 'nsd', 'vol_diff', 'rel_vol_error',
                                 'lesion_ppv', 'lesion_sensitivity', 'lesion_f1_score',
                                 'ref_count', 'pred_count', 'lcwa'],
                        help='List of metrics to compute. For details, '
                             'see: https://metricsreloaded.readthedocs.io/en/latest/reference/metrics/metrics.html.')
    parser.add_argument('-output', type=str, default='metrics.csv', required=False,
                        help='Path to the output CSV file to save the metrics. Default: metrics.csv')
    parser.add_argument('-jobs', type=int, default=cpu_count()//8, required=False,
                        help='Number of CPU cores to use in parallel. Default: cpu_count()//8.')

    return parser


def load_nifti_image(file_path):
    """
    Construct absolute path to the nifti image, check if it exists, and load the image data.
    :param file_path: path to the nifti image
    :return: nifti image data
    """
    file_path = os.path.expanduser(file_path)   # resolve '~' in the path
    file_path = os.path.abspath(file_path)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'File {file_path} does not exist.')
    nifti_image = nib.load(file_path)
    return nifti_image.get_fdata()


def fetch_participant_id_acq_id_run_id(filename_path, prefix='sub-'):
    """
    Get participant_id, acq_id, and run_id from the input BIDS-compatible filename or file path
    The function works both on absolute file paths as well as filenames
    :param filename_path: input nifti filename (e.g., sub-001_ses-01_T1w.nii.gz) or file path
    :param prefix: prefix of the participant ID in the filename (default: 'sub-')
    (e.g., /home/user/bids/sub-001/ses-01/anat/sub-001_ses-01_T1w.nii.gz
    :return: participant_id: participant ID (e.g., sub-001)
    :return: acq_id: acquisition ID (e.g., acq-01)
    :return: run_id: run ID (e.g., run-01)
    """

    participant = re.search(f'{prefix}(.*?)[_/]', filename_path)     # [_/] means either underscore or slash
    participant_id = participant.group(0)[:-1] if participant else ""   # [:-1] removes the last underscore or slash

    acquisition = re.search('acq-(.*?)[_/]', filename_path)     # [_/] means either underscore or slash
    acq_id = acquisition.group(0)[:-1] if acquisition else ""   # [:-1] removes the last underscore or slash

    run = re.search('run-(.*?)[_/]', filename_path)     # [_/] means either underscore or slash
    run_id = run.group(0)[:-1] if run else ""   # [:-1] removes the last underscore or slash

    # REGEX explanation
    # . - match any character (except newline)
    # *? - match the previous element as few times as possible (zero or more times)

    return participant_id, acq_id, run_id


def get_images_in_folder(prediction, reference):
    """
    Get all files (predictions and references/ground truths) in the input directories.
    The prediction and reference files are matched based on the participant_id, acq_id, and run_id.
    :param prediction: path to the directory with prediction files
    :param reference: path to the directory with reference (ground truth) files
    :return: list of prediction files, list of reference/ground truth files
    """
    # Get all files in the directories
    prediction_files = [os.path.join(prediction, f) for f in os.listdir(prediction) if f.endswith('.nii.gz')]
    reference_files = [os.path.join(reference, f) for f in os.listdir(reference) if f.endswith('.nii.gz')]

    # Create dataframe for prediction_files with participant_id, acq_id, run_id
    df_pred = pd.DataFrame(prediction_files, columns=['filename'])
    df_pred['participant_id'], df_pred['acq_id'], df_pred['run_id'] = zip(*df_pred['filename'].apply(fetch_participant_id_acq_id_run_id))

    # Create dataframe for reference_files with participant_id, acq_id, run_id
    df_ref = pd.DataFrame(reference_files, columns=['filename'])
    df_ref['participant_id'], df_ref['acq_id'], df_ref['run_id'] = zip(*df_ref['filename'].apply(fetch_participant_id_acq_id_run_id))

    # Merge the two dataframes on participant_id, acq_id, run_id
    df = pd.merge(df_pred, df_ref, on=['participant_id', 'acq_id', 'run_id'], how='outer', suffixes=('_pred', '_ref'))
    # Drop 'participant_id', 'acq_id', 'run_id'
    df.drop(['participant_id', 'acq_id', 'run_id'], axis=1, inplace=True)
    # Drop rows with NaN values
    df.dropna(inplace=True)

    prediction_files = df['filename_pred'].tolist()
    reference_files = df['filename_ref'].tolist()

    return prediction_files, reference_files


def compute_metrics_single_subject(prediction, reference, metrics):
    """
    Compute MetricsReloaded metrics for a single subject
    :param prediction: path to the nifti image with the prediction
    :param reference: path to the nifti image with the reference (ground truth)
    :param metrics: list of metrics to compute
    """
    # load nifti images
    print(f'\nProcessing:\n\tPrediction: {os.path.basename(prediction)}\n\tReference: {os.path.basename(reference)}')
    prediction_data = load_nifti_image(prediction)
    reference_data = load_nifti_image(reference)

    # check whether the images have the same shape and orientation
    if prediction_data.shape != reference_data.shape:
        raise ValueError(f'The prediction and reference (ground truth) images must have the same shape. '
                         f'The prediction image has shape {prediction_data.shape} and the ground truth image has '
                         f'shape {reference_data.shape}.')

    # get all unique labels (classes)
    # for example, for nnunet region-based segmentation, spinal cord has label 1, and lesions have label 2
    unique_labels_reference = np.unique(reference_data)
    unique_labels_reference = unique_labels_reference[unique_labels_reference != 0]  # remove background
    unique_labels_prediction = np.unique(prediction_data)
    unique_labels_prediction = unique_labels_prediction[unique_labels_prediction != 0]  # remove background

    # Get the unique labels that are present in the reference OR prediction images
    unique_labels = np.unique(np.concatenate((unique_labels_reference, unique_labels_prediction)))

    # append entry into the output_list to store the metrics for the current subject
    metrics_dict = {'reference': reference, 'prediction': prediction}

    # NOTE: this is hacky fix to try to speed up metrics computation, tread very carefully
    if len(unique_labels) == 2:
        # compute metrics only for lesions
        unique_labels = unique_labels[1:]

    # loop over all unique labels, e.g., voxels with values 1, 2, ...
    # by doing this, we can compute metrics for each label separately, e.g., separately for spinal cord and lesions
    for label in unique_labels:
        # create binary masks for the current label
        prediction_data_label = np.array(prediction_data == label, dtype=float)
        reference_data_label = np.array(reference_data == label, dtype=float)

        bpm = BPM(prediction_data_label, reference_data_label, measures=metrics)
        dict_seg = bpm.to_dict_meas()
        # Store info whether the reference or prediction is empty
        dict_seg['EmptyRef'] = bpm.flag_empty_ref
        dict_seg['EmptyPred'] = bpm.flag_empty_pred
        # add the metrics to the output dictionary
        metrics_dict[label] = dict_seg

    # # Special case when both the reference and prediction images are empty
    # else:
    #     label = 1
    #     bpm = BPM(prediction_data, reference_data, measures=metrics)
    #     dict_seg = bpm.to_dict_meas()

    #     # Store info whether the reference or prediction is empty
    #     dict_seg['EmptyRef'] = bpm.flag_empty_ref
    #     dict_seg['EmptyPred'] = bpm.flag_empty_pred
    #     # add the metrics to the output dictionary
    #     metrics_dict[label] = dict_seg

    return metrics_dict


def build_output_dataframe(output_list):
    """
    Convert JSON data to pandas DataFrame
    :param output_list: list of dictionaries with metrics
    :return: pandas DataFrame
    """
    rows = []
    for item in output_list:
        # Extract all keys except 'reference' and 'prediction' to get labels (e.g. 1.0, 2.0, etc.) dynamically
        labels = [key for key in item.keys() if key not in ['reference', 'prediction']]
        for label in labels:
            metrics = item[label]  # Get the dictionary of metrics
            # Dynamically add all metrics for the label
            row = {
                "reference": item["reference"],
                "prediction": item["prediction"],
                "label": label,
            }
            # Update row with all metrics dynamically
            row.update(metrics)
            rows.append(row)

    df = pd.DataFrame(rows)

    return df


def process_subject(prediction_file, reference_file, metrics):
    """
    Wrapper function to process a single subject.
    """
    return compute_metrics_single_subject(prediction_file, reference_file, metrics)


def main():
    # parse command line arguments
    parser = get_parser()
    args = parser.parse_args()

    # Initialize a list to store the output dictionaries (representing a single reference-prediction pair per subject)
    output_list = list()

    # Print the metrics to be computed
    print(f'Computing metrics: {args.metrics}')
    print(f'Using {args.jobs} CPU cores in parallel ...')

    # Args.prediction and args.reference are paths to folders with multiple nii.gz files (i.e., MULTIPLE subjects)
    if os.path.isdir(args.prediction) and os.path.isdir(args.reference):
        # Get all files in the directories
        prediction_files, reference_files = get_images_in_folder(args.prediction, args.reference)

        # Use multiprocessing to parallelize the computation
        with Pool(args.jobs) as pool:
            # Create a partial function to pass the metrics argument to the process_subject function
            func = partial(process_subject, metrics=args.metrics)
            # Compute metrics for each subject in parallel
            results = pool.starmap(func, zip(prediction_files, reference_files))

            # Collect the results
            output_list.extend(results)
    else:
        metrics_dict = compute_metrics_single_subject(args.prediction, args.reference, args.metrics)
        # Append the output dictionary (representing a single reference-prediction pair per subject) to the output_list
        output_list.append(metrics_dict)

    # Convert JSON data to pandas DataFrame
    df = build_output_dataframe(output_list)

    # create a separate dataframe for columns where EmptyRef and EmptyPred is True
    df_empty_masks = df[(df['EmptyRef'] == True) & (df['EmptyPred'] == True)]

    # keep only the rows where either pred or ref is non-empty or both are non-empty
    df = df[(df['EmptyRef'] == False) | (df['EmptyPred'] == False)]

    # Compute mean and standard deviation of metrics across all subjects
    df_mean = (df.drop(columns=['reference', 'prediction', 'EmptyRef', 'EmptyPred']).groupby('label').
               agg(['mean', 'std']).reset_index())

    # Convert multi-index to flat index
    df_mean.columns = ['_'.join(col).strip() for col in df_mean.columns.values]
    # Rename column `label_` back to `label`
    df_mean.rename(columns={'label_': 'label'}, inplace=True)

    # Rename columns
    df.rename(columns={metric: METRICS_TO_NAME[metric] for metric in METRICS_TO_NAME}, inplace=True)
    df_mean.rename(columns={metric: METRICS_TO_NAME[metric] for metric in METRICS_TO_NAME}, inplace=True)

    # format output up to 3 decimal places
    df = df.round(3)
    df_mean = df_mean.round(3)

    # save as CSV
    fname_output_csv = os.path.abspath(args.output)
    df.to_csv(fname_output_csv, index=False)
    print(f'Saved metrics to {fname_output_csv}.')

    # save as CSV
    fname_output_csv_mean = os.path.abspath(args.output.replace('.csv', '_mean.csv'))
    df_mean.to_csv(fname_output_csv_mean, index=False)
    print(f'Saved mean and standard deviation of metrics across all subjects to {fname_output_csv_mean}.')


if __name__ == '__main__':
    main()
