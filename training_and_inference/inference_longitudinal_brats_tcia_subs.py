import os
import sys
sys.path.append('/home/to5743/glioma_project/Glioma_Change_Detection_T2w/')  # this line is needed on the HPC cluster to recognize the dir as a python package
from dataset_creation.utils_dataset_creation import load_config_file
from datetime import datetime
from training_and_inference.utils_volume_classification import create_dir_if_not_exist, extract_volumes_and_labels_brats_tcia, \
    extract_unique_elements, select_model, run_inference, all_elements_in_list_are_identical
from shutil import copyfile
import torch
import pandas as pd
import numpy as np
from monai.transforms import AddChanneld, Compose, Resized, EnsureTyped, LoadImaged
from monai.data import Dataset
from torch.utils.data import DataLoader
import getpass
from scipy.stats import mode
from show_results.utils_show_results import classification_metrics, plot_roc_curve, plot_pr_curve
from utils_tdinoto.utils_lists import save_list_to_disk_with_pickle


__author__ = "Tommaso Di Noto"
__version__ = "0.0.1"
__email__ = "tommydino@hotmail.it"
__status__ = "Prototype"


def inference_on_brats_tcia_subs(batch_size: int,
                                 network: str,
                                 annotation_type: str,
                                 legend_label: str,
                                 path_df_sub_ses_label: str,
                                 input_dir: str,
                                 out_dir: str,
                                 median_shape_training_set: tuple,
                                 path_dir_previous_training: str,
                                 nb_workers,
                                 experiment):

    print("\nnetwork: {}, experiment: {}".format(network, experiment))
    print("Load weights from {}".format(path_dir_previous_training))

    date_hours_minutes = datetime.today().strftime('%b_%d_%Y_%Hh')  # type: str # save today's date
    out_dir = os.path.join(out_dir, "{}_{}".format(legend_label,
                                                   date_hours_minutes))  # type: str # update name of output directory

    create_dir_if_not_exist(out_dir)  # if output dir does not exist, create it

    # copy config file to output dir so that we know every time which were the input parameters
    config_file_path = sys.argv[2]  # type: str
    config_out_path = os.path.join(out_dir, "config_file.json")  # type: str
    if not os.path.exists(config_out_path):
        copyfile(config_file_path, config_out_path)
    
    df_sub_ses_label = pd.read_csv(path_df_sub_ses_label)  # load dataframe
    all_subs, volume_differences_t2, classification_labels = extract_volumes_and_labels_brats_tcia(input_dir,
                                                                                                   df_sub_ses_label,
                                                                                                   annotation_type)

    nb_output_classes = len(extract_unique_elements(classification_labels))  # type: int # find number of unique classes

    # define preprocessing and transforms for validation/test volumes
    val_transforms = Compose([LoadImaged(keys="volume"),
                              AddChanneld(keys="volume"),  # add channel; monai expects channel-first tensors
                              Resized(keys="volume", spatial_size=median_shape_training_set),  # resize all volumes to median volume shape
                              EnsureTyped(keys="volume")])  # ensure that input data is either a PyTorch Tensor or np array

    # choose network, loss function and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # if available, set GPU to use during training loop; otherwise, use cpu
    conv_filters_vgg, fc_nodes_vgg = (4, 8, 16, 32), (256, 128, 32)
    _, model_for_inference = select_model(network, device, nb_output_classes, conv_filters_vgg, fc_nodes_vgg)

    # create brats_tcia test data loader
    test_files = [{"volume": volume_name, 'label': label} for volume_name, label in zip(volume_differences_t2, classification_labels)]
    test_ds = Dataset(data=test_files, transform=val_transforms)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=nb_workers, pin_memory=torch.cuda.is_available())

    # -------- run inference on test subjects loading every time from a different training fold directory
    y_filenames_all_folds = []
    y_pred_binary_test_all_folds = []
    y_pred_probab_test_all_folds = []
    y_true_test_all_folds = []

    for training_fold in sorted(os.listdir(path_dir_previous_training)):
        if os.path.isdir(os.path.join(path_dir_previous_training, training_fold)):
            print("Computing predictions loading weights of training {}".format(training_fold))
            best_model_path = os.path.join(path_dir_previous_training, training_fold, "best_model.pth")
            model_for_inference.load_state_dict(torch.load(best_model_path, map_location=device))  # load best params according to validation results
            filenames, y_pred_binary_test, y_pred_probab_test, y_true_test = run_inference(model_for_inference, test_loader, device)

            # append to external lists
            y_filenames_all_folds.append(filenames)
            y_pred_binary_test_all_folds.append(y_pred_binary_test)
            y_pred_probab_test_all_folds.append(y_pred_probab_test)
            y_true_test_all_folds.append(y_true_test)

    assert all_elements_in_list_are_identical(y_filenames_all_folds), "We can only apply the majority vote if the order of the patients is always the same"
    assert all_elements_in_list_are_identical(y_true_test_all_folds)

    # compute average y_pred_probab
    y_pred_probab_test_all_folds_np = np.asarray(y_pred_probab_test_all_folds)  # convert list to np array
    average_y_pred_probab = np.mean(y_pred_probab_test_all_folds_np, axis=0)  # compute mean
    save_list_to_disk_with_pickle(list(average_y_pred_probab), out_dir, "y_pred_probab_avg.pkl")  # save it to disk

    # compute majority voting with mode
    majority_vote_y_pred_binary, _ = mode(np.asarray(y_pred_binary_test_all_folds))

    # --------------------- SAVE PREDICTIONS TO DISK
    save_list_to_disk_with_pickle(y_filenames_all_folds[0], out_dir, "filenames.pkl")  # save first element of list since they are all identical
    save_list_to_disk_with_pickle(y_true_test_all_folds[0], out_dir, "y_true.pkl")  # save first element of list since they are all identical
    save_list_to_disk_with_pickle(list(majority_vote_y_pred_binary.flatten()), out_dir, "y_pred_majority_voting.pkl")

    # --------------------- PRINT CLASSIFICATION RESULTS
    conf_mat, acc, rec_macro, spec, prec_macro, npv, f1_macro, _, _, _ = classification_metrics(np.asarray(y_true_test_all_folds[0]),
                                                                                                majority_vote_y_pred_binary.flatten())

    _, _, auc_roc = plot_roc_curve(y_true_test_all_folds[0],
                                   list(majority_vote_y_pred_binary.flatten()),
                                   cv_folds=5,
                                   embedding_label="",
                                   plot=False)
    _, _, aupr = plot_pr_curve(y_true_test_all_folds[0],
                               list(majority_vote_y_pred_binary.flatten()),
                               cv_folds=5,
                               embedding_label="",
                               plot=False)

    print("-------------------------------------------------------------------")
    print("\n{}".format(legend_label))
    print("Load weights from {}".format(path_dir_previous_training))
    print("Accuracy = {:.2f}".format(acc))
    print("Sensitivity (recall) = {:.2f}".format(rec_macro))
    print("Specificity = {:.2f}".format(spec))
    print("Precision (PPV) = {:.2f}".format(prec_macro))
    print("NPV = {:.2f}".format(npv))
    print("f1-score = {:.2f}".format(f1_macro))
    print("AUC = {:.2f}".format(auc_roc))
    print("AUPR = {:.2f}".format(aupr))


def main():
    # the code inside here is run only when THIS script is run, and not just imported
    config_dict = load_config_file()  # load input config file with argparser

    # extract training args
    network = config_dict['network']
    assert network in ("customVGG", "seresnext50"), "Only 'customVGG' and 'seresnext50' are allowed: found '{}' instead".format(network)
    experiment = config_dict['experiment']
    assert experiment in ("baseline", "transfer_learning"), "Only 'baseline' and 'transfer_learning' are allowed: found {} instead".format(experiment)

    batch_size = config_dict['batch_size']  # batch size used during training
    annotation_type = config_dict['annotation_type']  # type: str # used to distinguish between the two annotation schemes (manual and automatic)
    legend_label = config_dict['legend_label']  # type: str # label to use in the legend of figures
    legend_label = legend_label.replace("EXPERIMENT", "{}".format(experiment))
    legend_label = legend_label.replace("NETWORK", "{}".format(network))
    median_shape_training_set = tuple(config_dict['median_shape_training_set'])  # median volume shape used during training
    path_df_sub_ses_label = config_dict['path_df_sub_ses_label']  # path to dataframe containing the labels
    input_dir = config_dict['input_dir']  # path to folder containing volume differences
    out_dir = config_dict['out_dir']  # path where we save model parameters
    path_dir_previous_training = config_dict['path_dir_previous_training']
    path_dir_previous_training = path_dir_previous_training.replace("EXPERIMENT", "{}".format(experiment))
    path_dir_previous_training = path_dir_previous_training.replace("NETWORK", "{}".format(network))
    assert os.path.exists(path_dir_previous_training), "Path {} does not exist".format(path_dir_previous_training)
    
    # set number of jobs to run in parallel
    on_hpc_cluster = getpass.getuser() in ['to5743']  # type: bool # check if user is in list of authorized users
    if on_hpc_cluster:  # if we are running in the HPC
        assert torch.cuda.is_available(), "We expect to have a GPU when running on the HPC"
        nb_workers = 4  # type: int # number of jobs to run in parallel
    else:  # if instead we run the script locally
        nb_workers = 1  # type: int # number of jobs to run in parallel

    inference_on_brats_tcia_subs(batch_size,
                                 network,
                                 annotation_type,
                                 legend_label,
                                 path_df_sub_ses_label,
                                 input_dir,
                                 out_dir,
                                 median_shape_training_set,
                                 path_dir_previous_training,
                                 nb_workers,
                                 experiment)


if __name__ == '__main__':
    main()
