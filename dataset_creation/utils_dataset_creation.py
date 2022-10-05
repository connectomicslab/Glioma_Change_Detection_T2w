import argparse
import json
import os
from typing import Tuple
import subprocess
import glob
import re
import pydicom
import pandas as pd
from pathlib import Path
import numpy as np
import nibabel as nib
import ants
from ants.utils import bias_correction
from shutil import rmtree
from datetime import datetime
from HD_BET.run import run_hd_bet
from ants import image_read, image_similarity
from training_and_inference.utils_volume_classification import create_dir_if_not_exist
from utils_tdinoto.utils_strings import keep_only_digits


__author__ = "Tommaso Di Noto"
__version__ = "0.0.1"
__email__ = "tommydino@hotmail.it"
__status__ = "Prototype"


def get_parser() -> argparse.ArgumentParser:
    """This function creates a parser for handling input arguments
    Returns:
        p: parser
    """
    p = argparse.ArgumentParser()  # type: argparse.ArgumentParser
    # add argument to config file
    p.add_argument('--config', type=str, required=True, help='Path to json configuration file.')
    return p


def load_config_file() -> dict:
    """This function loads the input config file
    Returns:
        config_dictionary: it contains the input arguments
    """
    parser = get_parser()  # create parser
    args = parser.parse_args()  # convert argument strings to objects
    with open(args.config, 'r') as f:
        config_dictionary = json.load(f)

    return config_dictionary


def select_t2_volume(folder_path: str,
                     partial_session: str) -> str:
    """This function loops over all the sequences in the input folder and extracts, if present, the T2 volume
    Args:
        folder_path: folder where we search for the T2 Dicom sequence
        partial_session: partial session for this subject (i.e. only yyyymmdd, without hh:mmm:ss)
    Returns:
        t2_volume_path: path of T2 volume
    Raises:
        AssertionError: if input path does not exist
        ValueError: if T2 volume is not found
        ValueError: if more than one T2 volume is found
    """
    t2_volume_path = ""  # type: str # initialize as empty string
    if os.path.exists(folder_path):
        full_path = glob.glob(os.path.join(folder_path, '{}*'.format(partial_session)))  # type: list
        # if list is not empty
        if full_path:
            found_t2 = 0  # type: int # flag to check that only one T2 sequence was found
            if len(full_path) == 1:
                all_sequences = os.listdir(full_path[0])  # group all sequences in a list
                r = re.compile("(.+)?[Tt]2")  # define regex to match
                t2_sequences = list(filter(r.match, all_sequences))  # only extract sequences that match the regex
                sequence_description_keywords = ["cerveau", "CERVEAU", "Cerveau", "BRAIN", "Brain", "brain", "HEAD",
                                                 "Head", "head", "Cerebrale", "CEREBRALE", "cerebrale", "neuro", "Neuro", "NEURO"]  # sequence names to keep

                # if list is not empty (i.e. if at least one sequence matched the T2 regex)
                if t2_sequences:
                    if len(t2_sequences) == 1:  # if only one sequence matched
                        first_image = os.listdir(os.path.join(full_path[0], t2_sequences[0]))[0]  # extract first image of volume (only needed for pydicom)
                        dcm_tags = pydicom.dcmread(os.path.join(full_path[0], t2_sequences[0], first_image))  # read dicom tags

                        # first check if the tag BodyPartExamined exists
                        if hasattr(dcm_tags, "BodyPartExamined"):
                            if any(keyword in dcm_tags.BodyPartExamined for keyword in sequence_description_keywords):  # if in the tag there is one of the keyword in the list above
                                t2_volume_path = os.path.join(full_path[0], t2_sequences[0])  # type: str
                                found_t2 += 1  # increment flag

                        # then check if the tag PerformedProcedureStepDescription exists
                        if hasattr(dcm_tags, "PerformedProcedureStepDescription") and found_t2 == 0:
                            if any(keyword in dcm_tags.PerformedProcedureStepDescription for keyword in sequence_description_keywords):  # if in the tag there is one of the keyword in the list above
                                t2_volume_path = os.path.join(full_path[0], t2_sequences[0])  # type: str
                                found_t2 += 1  # increment flag

                        # then check if the tag RequestedProcedureDescription exists
                        if hasattr(dcm_tags, "RequestedProcedureDescription") and found_t2 == 0:
                            if any(keyword in dcm_tags.RequestedProcedureDescription for keyword in sequence_description_keywords):  # if in the tag there is one of the keyword in the list above
                                t2_volume_path = os.path.join(full_path[0], t2_sequences[0])  # type: str
                                found_t2 += 1  # increment flag

                    elif len(t2_sequences) > 1:  # if there is more than one T2 sequence that matched the regex
                        dcm_tags = [pydicom.dcmread(os.path.join(full_path[0], t2_sequences[idx], os.listdir(os.path.join(full_path[0], t2_sequences[idx]))[0])) for idx in range(len(t2_sequences))]
                        valid_brain_sequences_idxs = []
                        for idx, tags in enumerate(dcm_tags):
                            flag = 0
                            if hasattr(tags, "BodyPartExamined"):
                                if any(keyword in tags.BodyPartExamined for keyword in sequence_description_keywords):
                                    valid_brain_sequences_idxs.append(idx)  # append valid index
                                    flag += 1
                            if hasattr(tags, "PerformedProcedureStepDescription") and flag == 0:
                                if any(keyword in tags.PerformedProcedureStepDescription for keyword in sequence_description_keywords):
                                    valid_brain_sequences_idxs.append(idx)  # append valid index
                                    flag += 1
                            if hasattr(tags, "RequestedProcedureDescription") and flag == 0:
                                if any(keyword in tags.RequestedProcedureDescription for keyword in sequence_description_keywords):
                                    valid_brain_sequences_idxs.append(idx)  # append valid index

                        # if there is only one valid index (i.e. only one valid brain volume)
                        if len(valid_brain_sequences_idxs) == 1:
                            valid_idx = valid_brain_sequences_idxs[0]
                            t2_volume_path = os.path.join(full_path[0], t2_sequences[valid_idx])  # type: str
                        # if there is more than one valid index (i.e. more than one valid brain volume)
                        else:
                            if valid_brain_sequences_idxs:  # if list is not empty
                                # count number of slices per volume and take the one with the highest number of slices
                                nb_slices_per_volume = [len(os.listdir(os.path.join(full_path[0], t2_sequences[idx]))) for idx in valid_brain_sequences_idxs]
                                argmax_idx = nb_slices_per_volume.index(max(nb_slices_per_volume))  # find argmax
                                t2_volume_path = os.path.join(full_path[0], t2_sequences[argmax_idx])  # type: str
                else:
                    print("No T2 sequence was found")
            elif len(full_path) > 1:
                print("More than one partial session matched. Check {}".format(full_path))
        else:
            print("No matching session found")
    else:
        print("Path {} does not exist".format(folder_path))

    return t2_volume_path


def save_reg_quality_metrics(metrics_list: list,
                             out_file_path: str) -> None:
    """This function saves the registration quality metrics to disk
    Args:
       metrics_list: list where metrics were stored
       out_file_path: output file path where the list will be saved as csv
    """
    parent_dir = str(Path(out_file_path).parent)
    df = pd.DataFrame([metrics_list], columns=["neigh_corr", "mut_inf"])  # convert list to dataframe
    create_dir_if_not_exist(parent_dir)  # if parent folder does not exist, create it
    df.to_csv(out_file_path, index=False)


def register_comparative_2_current(current_volume_path: str,
                                   previous_volume_path: str,
                                   tmp_folder: str,
                                   sub: str,
                                   ses_diff: str) -> Tuple[np.ndarray, np.ndarray, str, str]:
    """This function registers the previous exam (i.e. comparative exam) to the current one (i.e. the one for which we have the report annotation)
    Args:
        current_volume_path: path of the current nifti volume (fixed image)
        previous_volume_path: path of the previous (i.e. comparative) nifti exam that we want to register (moving image)
        tmp_folder: temporary folder where to save the registered volume
        sub: sub id
        ses_diff: two sessions being compared
    Returns:
        registered_prev_2_current: registered output volume (from previous to current)
        registered_volume_aff: affine matrix of registered volume
        reg_quality_metrics_filename: filename of csv file where we save the registration quality metrics
        mat_file_name: filename of .mat file used to warp the moving volume
    """
    create_dir_if_not_exist(tmp_folder)  # if folder does not exist, create it

    out_path = os.path.join(tmp_folder, "{}_{}_out_prev_2_curr_".format(sub, ses_diff))
    if not os.path.exists(out_path + "Warped.nii.gz"):
        out_dict = ants.registration(fixed=ants.image_read(current_volume_path),
                                     moving=ants.image_read(previous_volume_path),
                                     type_of_transform="Affine",
                                     outprefix=out_path)

        out_dict["warpedmovout"].to_file(os.path.join(out_path + "Warped.nii.gz"))  # save warped volume to disk

        # cmd = ["antsRegistrationSyNQuick.sh", "-m", previous_volume_path, "-f", current_volume_path, "-t", "a", "-o", out_path]
        # process = subprocess.Popen(cmd, stdout=subprocess.PIPE)  # pass the list as input to Popen
        # _ = process.communicate()[0]  # the [0] is to return just the output, because otherwise it would be outs, errs = proc.communicate()

    mat_file_name = "{}_{}_out_prev_2_curr_0GenericAffine.mat".format(sub, ses_diff)

    # SAVE registration quality metrics
    reg_quality_metrics_filename = "{}_{}_reg_quality_metrics_comp2curr.csv".format(sub, ses_diff)  # set filename
    out_file_metrics_comp2curr = os.path.join(tmp_folder, reg_quality_metrics_filename)
    if not os.path.exists(out_file_metrics_comp2curr):
        comp_2_curr_cc_metric = image_similarity(image_read(out_path + "Warped.nii.gz"),
                                                 image_read(current_volume_path),
                                                 metric_type='ANTsNeighborhoodCorrelation')
        comp_2_curr_mi_metric = image_similarity(image_read(out_path + "Warped.nii.gz"),
                                                 image_read(current_volume_path),
                                                 metric_type="MattesMutualInformation")

        # SAVE REGISTRATION QUALITY METRICS
        save_reg_quality_metrics([comp_2_curr_cc_metric, comp_2_curr_mi_metric], out_file_metrics_comp2curr)

    # initialize output variables with zeros
    registered_prev_2_current = np.zeros((4, 4))
    registered_volume_aff = np.zeros((4, 4))

    warped_volume_path = out_path + "Warped.nii.gz"
    if os.path.exists(warped_volume_path):
        registered_prev_2_current_obj = nib.load(out_path + "Warped.nii.gz")
        registered_prev_2_current = np.asanyarray(registered_prev_2_current_obj.dataobj)
        registered_prev_2_current = np.float32(registered_prev_2_current)  # ensure volume has type float32
        registered_volume_aff = registered_prev_2_current_obj.affine

    return registered_prev_2_current, registered_volume_aff, reg_quality_metrics_filename, mat_file_name


def dcm2niix_wrapper(out_dir: str,
                     volume_path: str,
                     out_name: str) -> None:
    """This function is a wrapper for the commandline command dcm2niix.
    It is used to convert a dicom file into the corresponding nifti+json
    Args:
        out_dir: folder where the nifti (and json) will be saved
        volume_path: path to dicom volume
        out_name: name of output files
    """
    create_dir_if_not_exist(out_dir)  # if output folder does not exist, create it

    if not os.path.exists(os.path.join(out_dir, "{}.nii.gz".format(out_name))):
        cmd = ["dcm2niix", "-f", out_name, "-z", "y", "-o", out_dir, volume_path]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE)  # pass the list as input to Popen
        _ = process.communicate()[0]  # the [0] is to return just the output, because otherwise it would be outs, errs = proc.communicate()
    else:
        print("Nifti volume already exists")


def fsl_brain_extraction(out_dir: str,
                         path_volume_to_skullstrip: str,
                         registered_comparative_t2_volume: np.ndarray,
                         output_file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """This function performs the FSL brain extraction tool (BET).
    Args:
        out_dir: path to output folder
        path_volume_to_skullstrip: path of volume that we want to skull-strip
        registered_comparative_t2_volume: registered comparative volume. Will get the exact same brain extraction as the first volume
        output_file_path: path of brain-extracted volume
    Returns:
        current_bet_t2_volume: main volume skull-stripped
        registered_bet_comparative_t2_volume: second volume skull-stripped exactly like the main volume
    """
    create_dir_if_not_exist(out_dir)  # if output folder does not exist, create it

    if not os.path.exists(output_file_path):
        cmd = ['bet', path_volume_to_skullstrip, output_file_path, '-f', '0.5']  # type: list # create list as if it was a command line expression
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE)  # pass the list as input to Popen
        _ = process.communicate()[0]  # the [0] is to return just the output, because otherwise it would be outs, errs = process.communicate()

    current_bet_t2_volume_obj = nib.load(output_file_path)  # load skull-stripped volume
    current_bet_t2_volume = np.asanyarray(current_bet_t2_volume_obj.dataobj)  # convert to numpy array
    registered_bet_comparative_t2_volume = np.copy(registered_comparative_t2_volume)  # first create hard copy
    registered_bet_comparative_t2_volume[np.where(current_bet_t2_volume == 0)] = 0  # apply same brain extraction mask of current volume

    return current_bet_t2_volume, registered_bet_comparative_t2_volume


def hd_bet_brain_extraction(out_dir: str,
                            path_volume_to_skullstrip: str,
                            registered_comparative_t2_volume: np.ndarray,
                            output_file_path: str,
                            on_hpc_cluster: bool) -> Tuple[np.ndarray, np.ndarray]:
    """This function performs the brain extraction with the HD-BET tool (Isensee et al.).
    Args:
        out_dir: path to output folder
        path_volume_to_skullstrip: path of volume that we want to skull-strip
        registered_comparative_t2_volume: registered comparative volume. Will get the exact same brain extraction as the first volume
        output_file_path: path of brain-extracted volume
        on_hpc_cluster: indicates if the script is run on the cluster or locally
    Returns:
        current_bet_t2_volume: main volume skull-stripped
        registered_bet_comparative_t2_volume: second volume skull-stripped exactly like the main volume
    """
    create_dir_if_not_exist(out_dir)  # if output folder does not exist, create it

    if not os.path.exists(output_file_path):  # if the output file does not exist
        if on_hpc_cluster:
            run_hd_bet(path_volume_to_skullstrip, output_file_path)  # run on GPU
        else:
            run_hd_bet(path_volume_to_skullstrip, output_file_path, device="cpu", do_tta=False, mode="fast")  # run on CPU

    current_bet_t2_volume_obj = nib.load(output_file_path)  # load skull-stripped volume
    current_bet_t2_volume = np.asanyarray(current_bet_t2_volume_obj.dataobj)  # convert to numpy array
    registered_bet_comparative_t2_volume = np.copy(registered_comparative_t2_volume)  # first create hard copy
    registered_bet_comparative_t2_volume[current_bet_t2_volume == 0] = 0  # apply same brain extraction mask of current volume

    # ensure also the opposite cause sometimes the registration crops some slices
    current_bet_t2_volume[registered_bet_comparative_t2_volume == 0] = 0  # set to 0 voxels where the registered volume is zero

    return current_bet_t2_volume, registered_bet_comparative_t2_volume


def apply_n4_bias_field_correction_ants(input_volume_path: str,
                                        output_volume_path: str) -> None:
    """This function applies the N4 bias field correction from ants.
    Args:
        input_volume_path: path to input volume that we want to correct
        output_volume_path: path to bias-field-corrected output volume
    """
    volume_ants = ants.image_read(input_volume_path)  # type: ants.ANTsImage
    if not os.path.exists(output_volume_path):  # if output path does not exist
        volume_ants_n4 = bias_correction.n4_bias_field_correction(volume_ants)  # type: ants.ANTsImage
        volume_ants_n4.to_file(output_volume_path)  # save output volume to disk


def set_to_zero_axial_slices_where_only_one_volume_is_nonzero(current_volume: np.ndarray,
                                                              comparative_volume: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """This function takes as input two volumes and sets to zero the slices for which one volume has values
    and the other only has two unique values out of which one is zero.
    Args:
        current_volume: first volume
        comparative_volume: second volume
    Returns:
        current_volume: first volume with some slices that may have been set to 0
        comparative_volume: second volume with some slices that may have been set to 0
    """
    def set_slices_to_zero(first_volume, second_volume):
        for k in range(first_volume.shape[2]):  # loop over axial slices
            one_axial_slice_first_volume = first_volume[:, :, k]  # extract one slice at a time
            unique_values_first = np.unique(one_axial_slice_first_volume)  # find unique values
            if len(unique_values_first) <= 2 and (unique_values_first == 0).any():  # if there are only two unique values and one of them is zero
                same_axial_slice_second_volume = second_volume[:, :, k]  # extract same axial slice from second volume
                unique_values_second = np.unique(same_axial_slice_second_volume)  # extract unique values
                # if instead the second volume has more than 2 unique values
                if len(unique_values_second) > 2:
                    # set slice to zero for both volumes
                    first_volume[:, :, k] = 0
                    second_volume[:, :, k] = 0

        return first_volume, second_volume

    current_volume, comparative_volume = set_slices_to_zero(current_volume, comparative_volume)
    comparative_volume, current_volume = set_slices_to_zero(comparative_volume, current_volume)

    # ensure dtype
    current_volume = np.float32(current_volume)
    comparative_volume = np.float32(comparative_volume)

    return current_volume, comparative_volume


def remove_zeros_ijk_from_registered_volumes(current_volume: np.ndarray,
                                             comparative_volume: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """This function removes rows, columns and slices that only contain zeros.
    Args:
        current_volume (np.ndarray): current volume
        comparative_volume (np.ndarray): comparative volume
    Returns:
        current_volume (np.ndarray): cropped current volume
        comparative_volume (np.ndarray): cropped comparative volume
    """
    def remove_zeros_one_coordinate(first_volume: np.ndarray, second_volume: np.ndarray, range_spatial_dim: int, spatial_dim: int):
        idxs_nonzero_slices = []  # will the contain the indexes of all the slices that have nonzero values
        for idx in range(range_spatial_dim):  # loop over coordinate
            if spatial_dim == 0:
                one_slice = first_volume[idx, :, :]
            elif spatial_dim == 1:
                one_slice = first_volume[:, idx, :]
            elif spatial_dim == 2:
                one_slice = first_volume[:, :, idx]
            else:
                raise ValueError("spatial_dim can only be 0, 1, or 2. Got {} instead".format(spatial_dim))

            if np.count_nonzero(one_slice) > 0:  # if the slice has some nonzero values
                idxs_nonzero_slices.append(idx)  # save slice index

        # retain only indexes with nonzero values from the two input volumes
        if spatial_dim == 0:
            first_cropped_volume = first_volume[idxs_nonzero_slices, :, :]
            second_cropped_volume = second_volume[idxs_nonzero_slices, :, :]
        elif spatial_dim == 1:
            first_cropped_volume = first_volume[:, idxs_nonzero_slices, :]
            second_cropped_volume = second_volume[:, idxs_nonzero_slices, :]
        elif spatial_dim == 2:
            first_cropped_volume = first_volume[:, :, idxs_nonzero_slices]
            second_cropped_volume = second_volume[:, :, idxs_nonzero_slices]
        else:
            raise ValueError("spatial_dim can only be 0, 1, or 2. Got {} instead".format(spatial_dim))

        return first_cropped_volume, second_cropped_volume

    assert current_volume.shape == comparative_volume.shape, "The two volumes must have the same shape"

    # i coordinate
    current_volume, comparative_volume = remove_zeros_one_coordinate(current_volume, comparative_volume, current_volume.shape[0], spatial_dim=0)
    # j coordinate
    current_volume, comparative_volume = remove_zeros_one_coordinate(current_volume, comparative_volume, current_volume.shape[1], spatial_dim=1)
    # k coordinate
    current_volume, comparative_volume = remove_zeros_one_coordinate(current_volume, comparative_volume, current_volume.shape[2], spatial_dim=2)

    # zero pad just a little bit on the borders of the volumes otherwise it might be hard to perform convolutions later
    current_volume = np.pad(current_volume, 3, 'constant', constant_values=0)
    comparative_volume = np.pad(comparative_volume, 3, 'constant', constant_values=0)

    return current_volume, comparative_volume


def delete_unused_files(out_folder: str,
                        files_to_keep: list) -> None:
    """This function removes unused files from the output folder
    Args:
        out_folder: path to output folder
        files_to_keep: files to keep (i.e. that must not be deleted)
    """
    # loop over files in output folder
    for file in os.listdir(out_folder):
        if file not in files_to_keep:  # if file is not in list of files to keep
            os.remove(os.path.join(out_folder, file))  # remove file

    # if after deleting the files the folder is empty (it happens when one of the preprocessing steps fails) or has more/less files than the expected ones (something went wrong)
    if len(os.listdir(out_folder)) == 0 or len(os.listdir(out_folder)) != len(files_to_keep):
        rmtree(out_folder)  # remove folder


def ensure_sessions_are_far_from_surgery(sub_only_digits: str,
                                         df_dates_of_surgery: pd.DataFrame,
                                         current_ses: str,
                                         comparative_ses: str) -> bool:
    """This function checks whether the sessions of the session pair that we are creating are far from the day of surgery for this patient.
    We want to exclude any session pair that contains any session too close to surgery because otherwise the images would still show post-surgery
    abnormalities that are confusing for the downstream task.
    Args:
        sub_only_digits: subject being analyzed
        df_dates_of_surgery: it contains the dates of surgery for all subjects
        current_ses: current (i.e. today's) session
        comparative_ses: comparative (i.e. previous) session
    Returns:
        sessions_are_far_from_surgery: indicates whether both sessions are far from the date of surgery
    """
    sessions_are_far_from_surgery = True  # initialize to False

    ses_list = [comparative_ses, current_ses]
    ses_list_only_digits = [keep_only_digits(ses) for ses in ses_list]  # keep only digits
    ses_list_only_dates = [datetime.strptime(ses, '%Y%m%d') for ses in ses_list_only_digits]  # convert to datetime format
    row_of_interest = df_dates_of_surgery.loc[df_dates_of_surgery["ipp_anonymized"] == int(sub_only_digits)]  # extract row(s) corresponding to this sub
    assert row_of_interest.shape[0] == 1, "Only one row should match"
    row_of_interest = row_of_interest.astype({"date_of_surgery": str})  # change dtype
    date_of_surgery = datetime.strptime(row_of_interest["date_of_surgery"].iloc[0], '%Y%m%d')  # extract date of surgery for this sub
    print(sub_only_digits, ses_list, row_of_interest["date_of_surgery"].iloc[0])
    days_from_surgery = [abs((ses - date_of_surgery).days) for ses in ses_list_only_dates]  # count days from surgery for each session

    # if any of the two days_of_difference from surgery is below 4 weeks, we exclude this ses-diff
    if any(days_diff <= 28 for days_diff in days_from_surgery):
        sessions_are_far_from_surgery = False

    return sessions_are_far_from_surgery


def save_volume_to_disk(volume_np: np.ndarray,
                        affine_matrix: np.ndarray,
                        out_dir: str,
                        out_filename: str) -> None:
    """This function saves the input numpy array to disk
    Args:
        volume_np: volume to save
        affine_matrix: affine matrix to use when saving
        out_dir: output directory
        out_filename: filename of output file
    """
    volume_np = np.float32(volume_np)  # ensure dtype
    volume_obj = nib.Nifti1Image(volume_np, affine=affine_matrix)  # convert from numpy array to nib object
    nib.save(volume_obj, os.path.join(out_dir, out_filename))  # save nib object as nifti
