import os
import sys
sys.path.append('/home/to5743/glioma_project/Change_Detection_Glioma/')  # make root project dir a python package
import pandas as pd
from datetime import datetime
import nibabel as nib
import numpy as np
from joblib import Parallel, delayed
import getpass
from scipy.stats import zscore
import torch
from dataset_creation.utils_dataset_creation import load_config_file, select_t2_volume, register_comparative_2_current, dcm2niix_wrapper,\
    hd_bet_brain_extraction, apply_n4_bias_field_correction_ants, set_to_zero_axial_slices_where_only_one_volume_is_nonzero,\
    remove_zeros_ijk_from_registered_volumes, delete_unused_files, ensure_sessions_are_far_from_surgery, save_volume_to_disk
from utils_tdinoto.utils_strings import add_leading_zeros, str2bool
from utils_tdinoto.utils_lists import load_list_from_disk_with_pickle


def create_difference_volume_one_sub(idx: int,
                                     row: pd.Series,
                                     data_path: str,
                                     df_link: pd.DataFrame,
                                     out_folder: str,
                                     on_hpc_cluster: bool,
                                     manual_subs_list: list,
                                     df_dates_of_surgery: pd.DataFrame) -> None:
    """This function creates the difference volume for one subject
    Args:
        idx (int): index corresponding to the row of the comparative dates dataframe
        row (pd.Series): row (of the comparative dates dataframe) corresponding to the difference volume we want to create
        data_path (str): directory where the volumes are stored
        df_link (pd.DataFrame): it contains the comparative and current sessions, together with the labels
        out_folder (str): output directory where we save the difference volumes
        on_hpc_cluster (bool): indicates if the script is run on the cluster or locally
        manual_subs_list (list): if we are creating the dataset for the automatically-annotated data, this list contains the subs already tagged manually; these cannot be used so they will be skipped
        df_dates_of_surgery (pd.DataFrame): it contains the dates of surgery of all subjects
    """
    sub_number = row['ipp']
    sub_number = add_leading_zeros(sub_number, out_len=6)  # add leading zeros to match the correct anonymized ipp
    sub = "sub-{}".format(sub_number)
    sub_folder = os.path.join(data_path, sub)
    current_exam_partial_ses = "ses-{}".format(row['exam_date'])
    comparative_exam_partial_ses = "ses-{}".format(row['comparative_date'])
    if comparative_exam_partial_ses < current_exam_partial_ses:
        ses_diff = "{}_vs_{}".format(comparative_exam_partial_ses, current_exam_partial_ses)
        print("{}/{}: {} {}".format(idx + 1, df_link.shape[0], sub, ses_diff))

        # ensure that both sessions are far from surgery; if they are not, we don't compute the difference maps for this sub_ses-diff
        sessions_are_far_from_surgery = ensure_sessions_are_far_from_surgery(sub_number, df_dates_of_surgery, current_exam_partial_ses, comparative_exam_partial_ses)

        # when creating the dataset for the automatically-annotated data, we skip the subjects that were tagged manually; also, skip sub_ses-pairs for which any session is too close to date of surgery
        if sub not in manual_subs_list and sessions_are_far_from_surgery:

            # initialize empty filenames; they should be overwritten later if all the preprocessing steps work well
            current_bet_t2_volume_n4_filename = ""
            comparative_bet_t2_volume_n4_filename = ""
            out_filename_current_cropped = ""
            out_filename_comparative_cropped = ""
            out_filename_current_cropped_zscored = ""
            out_filename_comparative_cropped_zscored = ""
            difference_t2_volumes_filename = ""
            reg_quality_metrics_filename = ""
            registration_mat_file_path = ""

            # define unique output folder for this sub and sessions compared
            out_folder_unique = os.path.join(out_folder, sub, ses_diff)

            # extract paths of T2 volumes (current and comparative)
            current_t2_volume_path = select_t2_volume(sub_folder, current_exam_partial_ses)
            comparative_t2_volume_path = select_t2_volume(sub_folder, comparative_exam_partial_ses)

            # if both paths are not empty
            if current_t2_volume_path and comparative_t2_volume_path:
                # if the output folder does not exist
                if not os.path.exists(out_folder_unique):
                    # ------------------------- convert T2 volumes from dicom to nifti
                    # ----- current
                    out_name_current_t2 = "{}_{}_current_t2".format(sub, ses_diff)
                    dcm2niix_wrapper(out_folder_unique, current_t2_volume_path, out_name_current_t2)
                    # make sure there are exactly 2 files (the .nii and the .json); for some weird problematic cases a file with "_Eq_1" is created
                    if len(os.listdir(out_folder_unique)) == 2:
                        # ----- comparative
                        out_name_comparative_t2 = "{}_{}_comparative_t2".format(sub, ses_diff)
                        dcm2niix_wrapper(out_folder_unique, comparative_t2_volume_path, out_name_comparative_t2)
                        # make sure there are exactly 4 files (the 2 .nii and the 2 .json); for some weird problematic cases an extra file with "_Eq_1" is created
                        if len(os.listdir(out_folder_unique)) == 4:
                            # ------------------------- apply ants bias-field-correction (if the volumes exist)
                            if os.path.exists(os.path.join(out_folder_unique, "{}.nii.gz".format(out_name_current_t2))) and os.path.exists(os.path.join(out_folder_unique, "{}.nii.gz".format(out_name_comparative_t2))):
                                # ----- current
                                out_n4_current_volume_filename = out_name_current_t2 + "_n4.nii.gz"
                                out_n4_current_volume_path = os.path.join(out_folder_unique, out_n4_current_volume_filename)
                                apply_n4_bias_field_correction_ants(os.path.join(out_folder_unique, out_name_current_t2 + ".nii.gz"),
                                                                    out_n4_current_volume_path)
                                # ----- comparative
                                out_n4_comparative_volume_filename = out_name_comparative_t2 + "_n4.nii.gz"
                                out_n4_comparative_volume_path = os.path.join(out_folder_unique, out_n4_comparative_volume_filename)
                                apply_n4_bias_field_correction_ants(os.path.join(out_folder_unique, out_name_comparative_t2 + ".nii.gz"),
                                                                    out_n4_comparative_volume_path)

                                # ------------------------- apply registration -> the moving volume is the comparative and the fixed volume is the current
                                registered_comparative_t2_volume_n4, registered_aff,\
                                    reg_quality_metrics_filename, registration_mat_file_path = register_comparative_2_current(out_n4_current_volume_path,
                                                                                                                              out_n4_comparative_volume_path,
                                                                                                                              out_folder_unique,
                                                                                                                              sub,
                                                                                                                              ses_diff)

                                # if the registered volume and the affine matrix contain some non-zero values, and if the .mat file exists
                                if np.any(registered_comparative_t2_volume_n4) and np.any(registered_aff) and os.path.exists(os.path.join(out_folder_unique, registration_mat_file_path)):
                                    # -------------- BRAIN EXTRACTION: perform the same brain extraction on both volumes
                                    current_bet_t2_volume_n4_filename = out_n4_current_volume_filename.replace("current", "current_bet")  # set filename
                                    output_file_path = os.path.join(out_folder_unique, current_bet_t2_volume_n4_filename)
                                    current_bet_t2_volume_n4, registered_bet_comparative_t2_volume_n4 = hd_bet_brain_extraction(out_folder_unique,
                                                                                                                                out_n4_current_volume_path,
                                                                                                                                registered_comparative_t2_volume_n4,
                                                                                                                                output_file_path,
                                                                                                                                on_hpc_cluster)
                                    # save comparative BET to disk
                                    comparative_bet_t2_volume_n4_filename = out_n4_comparative_volume_filename.replace("comparative", "comparative_bet")
                                    save_volume_to_disk(registered_bet_comparative_t2_volume_n4, registered_aff, out_dir=out_folder_unique,
                                                        out_filename=comparative_bet_t2_volume_n4_filename)

                                    if registered_bet_comparative_t2_volume_n4.shape == current_bet_t2_volume_n4.shape:  # if the two volumes have same shape
                                        # set to zero axial slices where only one volume is nonzero
                                        current_bet_t2_volume_n4, registered_bet_comparative_t2_volume_n4 = set_to_zero_axial_slices_where_only_one_volume_is_nonzero(current_bet_t2_volume_n4,
                                                                                                                                                                      registered_bet_comparative_t2_volume_n4)
                                        # remove rows, columns and slices that only contain zeros
                                        current_bet_t2_volume_n4, registered_bet_comparative_t2_volume_n4 = remove_zeros_ijk_from_registered_volumes(current_bet_t2_volume_n4,
                                                                                                                                                     registered_bet_comparative_t2_volume_n4)

                                        # save volumes to disk before zscore normalization
                                        out_filename_current_cropped = current_bet_t2_volume_n4_filename.replace("bet", "bet_cropped")
                                        save_volume_to_disk(current_bet_t2_volume_n4, registered_aff, out_dir=out_folder_unique, out_filename=out_filename_current_cropped)

                                        out_filename_comparative_cropped = comparative_bet_t2_volume_n4_filename.replace("bet", "bet_cropped")
                                        save_volume_to_disk(registered_bet_comparative_t2_volume_n4, registered_aff, out_dir=out_folder_unique, out_filename=out_filename_comparative_cropped)

                                        # ---------------------------------- apply z-score normalization and save cropped volumes to disk
                                        # ---- current
                                        current_bet_t2_volume_n4_normalized = zscore(current_bet_t2_volume_n4, axis=None)  # z-score normalization
                                        current_bet_t2_volume_n4_normalized[current_bet_t2_volume_n4 == 0] = 0  # set background back to zero as it was after brain extraction (otherwise the normalization changes its value)
                                        out_filename_current_cropped_zscored = out_filename_current_cropped.replace("cropped", "cropped_zscored")
                                        save_volume_to_disk(current_bet_t2_volume_n4_normalized, registered_aff, out_folder_unique, out_filename_current_cropped_zscored)

                                        # ---- registered comparative
                                        registered_bet_comparative_t2_volume_n4_normalized = zscore(registered_bet_comparative_t2_volume_n4, axis=None)  # z-score normalization
                                        registered_bet_comparative_t2_volume_n4_normalized[registered_bet_comparative_t2_volume_n4 == 0] = 0  # set background back to zero as it was after brain extraction (otherwise the normalization changes its value)
                                        out_filename_comparative_cropped_zscored = out_filename_comparative_cropped.replace("cropped", "cropped_zscored")
                                        save_volume_to_disk(registered_bet_comparative_t2_volume_n4_normalized, registered_aff, out_folder_unique, out_filename_comparative_cropped_zscored)

                                        # check that both volumes have non-zero voxels. Sometimes they can be empty
                                        if np.count_nonzero(current_bet_t2_volume_n4_normalized) and np.count_nonzero(registered_bet_comparative_t2_volume_n4_normalized):
                                            # compute absolute difference map of normalized volumes
                                            difference_t2_volumes = np.absolute(np.subtract(current_bet_t2_volume_n4_normalized, registered_bet_comparative_t2_volume_n4_normalized),
                                                                                dtype=np.float32)
                                            # if difference map is not empty
                                            if np.count_nonzero(difference_t2_volumes):
                                                # save difference volume to disk
                                                difference_t2_volumes_filename = "{}_{}_difference_t2_volumes.nii.gz".format(sub, ses_diff)  # set filename
                                                save_volume_to_disk(difference_t2_volumes, registered_aff, out_folder_unique, difference_t2_volumes_filename)
                            else:
                                print("Warning: something went wrong with dcm2niix for {}-{}".format(sub, ses_diff))

                    # delete unused files
                    files_to_keep = [current_bet_t2_volume_n4_filename,
                                     current_bet_t2_volume_n4_filename.replace("n4", "n4_mask"),
                                     comparative_bet_t2_volume_n4_filename,
                                     out_filename_current_cropped,
                                     out_filename_comparative_cropped,
                                     out_filename_current_cropped_zscored,
                                     out_filename_comparative_cropped_zscored,
                                     difference_t2_volumes_filename,
                                     reg_quality_metrics_filename,
                                     registration_mat_file_path]
                    delete_unused_files(out_folder_unique, files_to_keep)
    else:
        print("Warning {}: {} not < than {}".format(sub, comparative_exam_partial_ses, current_exam_partial_ses))


def create_dataset_in_parallel(data_path: str,
                               df_link: pd.DataFrame,
                               out_folder: str,
                               nb_parallel_jobs: int,
                               on_hpc_cluster: bool,
                               manual_subs_list: list,
                               df_dates_of_surgery: pd.DataFrame) -> None:
    """This function creates the dataset of t2 difference volumes
    Args:
        data_path (str): path to folder containing all the sub-ses-volumes for the glioma dataset
        df_link (pd.Dataframe): it contains the current and the previous exam date, together with the linking annotation between them
        out_folder (str): folder where we save output files
        nb_parallel_jobs (int): number of jobs to run in parallel with joblib
        on_hpc_cluster (bool): indicates if the script is run on the cluster or locally
        manual_subs_list (list): if we are creating the dataset for the automatically-annotated data, this list contains the subs already tagged manually; these cannot be used so they will be skipped
        df_dates_of_surgery (pd.Dataframe): it contains the dates of surgery for each subject
    """
    Parallel(n_jobs=nb_parallel_jobs,
             backend='threading')(delayed(create_difference_volume_one_sub)(idx,
                                                                            row,
                                                                            data_path,
                                                                            df_link,
                                                                            out_folder,
                                                                            on_hpc_cluster,
                                                                            manual_subs_list,
                                                                            df_dates_of_surgery) for idx, row in df_link.iterrows())


def create_difference_t2_images_dataset(df_comparative_dates_path: str,
                                        data_path: str,
                                        out_folder: str,
                                        t2_label_column_name: str,
                                        nb_parallel_jobs: int,
                                        dtype_dict: dict,
                                        on_hpc_cluster: bool,
                                        manual_subs_list: list,
                                        path_df_dates_of_surgery: str) -> None:
    """This function cleans the dataframe of comparative/current dates and invokes the actual function that creates the dataset
    Args:
        df_comparative_dates_path (str): path to dataframe that for each report, it contains ipp, current_date, comparative_date, global_label, t1_label, t2_label, "significant_change_label"
        data_path (str): folder that contains all the sub-ses DICOM series for this study
        out_folder (str): path where to save output files
        t2_label_column_name (str): name of the column that contains the t2 labels
        nb_parallel_jobs (int): number of jobs to run in parallel with joblib
        dtype_dict (dict): specifies the dtypes for the columns of the pandas dataframe
        on_hpc_cluster (bool): indicates if the script is run on the cluster or locally
        manual_subs_list (list): if we are creating the dataset for the automatically-annotated data, this list contains the subs already tagged manually; these cannot be used so they will be skipped
        path_df_dates_of_surgery (str): path to dataframe containing the dates of surgery for each subject
    """
    # load dataframe
    df_comparative_dates = pd.read_csv(df_comparative_dates_path, dtype=dtype_dict)  # type: pd.DataFrame

    # remove rows with NaN, NaT, etc.
    df_comparative_dates = df_comparative_dates.dropna()  # type: pd.DataFrame

    # set names of columns of interest
    important_columns = ["ipp", "exam_date", "comparative_date", t2_label_column_name]  # type: list

    # extract sub-dataframe only with the columns of interest
    sub_df_comparative_dates = df_comparative_dates.loc[:, important_columns]  # type: pd.DataFrame

    # remove rows with unknown/not mentioned T2 label
    df_rows_with_known_label = sub_df_comparative_dates.query("{} != 3".format(t2_label_column_name))

    # restart dataframe's indexes from 0 and make them increase without holes (i.e. missing rows)
    df_rows_with_known_label = df_rows_with_known_label.reset_index(drop=True)

    # load dataframe with dates of surgery
    df_dates_of_surgery = pd.read_csv(path_df_dates_of_surgery)

    # invoke function that creates the dataset
    create_dataset_in_parallel(data_path, df_rows_with_known_label, out_folder, nb_parallel_jobs, on_hpc_cluster, manual_subs_list, df_dates_of_surgery)


def main():
    out_date = (datetime.today().strftime('%b_%d_%Y'))  # type: str # save today's date

    # the code inside here is run only when THIS script is run, and not just imported
    config_dict = load_config_file()  # load input config file with argparser

    # extract training args
    df_comparative_dates_path = config_dict['df_comparative_dates_path']
    data_dir_hard_disk = config_dict['data_dir_hard_disk']
    unique_outdir_name = config_dict['unique_outdir_name']
    out_folder = config_dict['out_folder']
    annotation_type = config_dict['annotation_type']
    path_manual_subs_list = config_dict['path_manual_subs_list']
    path_df_dates_of_surgery = config_dict['path_df_dates_of_surgery']
    include_manual_subs = str2bool(config_dict['include_manual_subs'])

    # define input args
    on_hpc_cluster = getpass.getuser() in ['to5743']  # type: bool # check if user is in list of authorized users
    if on_hpc_cluster:  # if we are running in the HPC
        assert torch.cuda.is_available(), "We expect to have a GPU when running on the HPC"
        nb_parallel_jobs = -1  # type: int # number of jobs to run in parallel with joblib
    else:  # if instead we are running the script locally
        nb_parallel_jobs = 1  # type: int # number of jobs to run in parallel with joblib

    out_folder = os.path.join(out_folder, "out_{}_{}".format(out_date, unique_outdir_name))

    if annotation_type == "manual":
        manual_subs_list = []  # set as empty list
        dtype_dict = {'ipp': str,
                      'exam_date': int,
                      'comparative_date': int,
                      'acc_num': str,
                      'report': str,
                      't2_label_A1': int}
        t2_label_column_name = "t2_label_A1"
    elif annotation_type == "automatic":
        manual_subs_list = load_list_from_disk_with_pickle(path_manual_subs_list)
        if include_manual_subs:
            manual_subs_list = []

        dtype_dict = {'ipp': str,
                      'exam_date': int,
                      'comparative_date': int,
                      'acc_num': str,
                      'report': str,
                      't2_label_random_forest': int,
                      't2_label_random_forest_probab': float}
        t2_label_column_name = "t2_label_random_forest"
    else:
        raise ValueError("annotation_type can only be manual or automatic; got {} instead".format(annotation_type))

    create_difference_t2_images_dataset(df_comparative_dates_path,
                                        data_dir_hard_disk,
                                        out_folder,
                                        t2_label_column_name,
                                        nb_parallel_jobs,
                                        dtype_dict,
                                        on_hpc_cluster,
                                        manual_subs_list,
                                        path_df_dates_of_surgery)


if __name__ == '__main__':
    main()
