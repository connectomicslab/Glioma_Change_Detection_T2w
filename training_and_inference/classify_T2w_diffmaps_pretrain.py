import os
import sys
sys.path.append('/home/to5743/glioma_project/Change_Detection_Glioma/')  # this line is needed on the HPC cluster to recognize the dir as a python package
import pandas as pd
import torch
from torch.utils.data import DataLoader
from monai.data import Dataset
from shutil import copyfile
import numpy as np
from monai.transforms import AddChanneld, Compose, Resized, EnsureTyped, LoadImaged, \
    Lambda, RandFlipd, RandGaussianNoised, RandZoomd, Rand3DElasticd
from datetime import datetime
import getpass
from dataset_creation.utils_dataset_creation import load_config_file
from utils_tdinoto.utils_strings import str2bool
from utils_tdinoto.utils_lists import extract_unique_elements
from training_and_inference.utils_volume_classification import training_loop_with_validation, \
     extract_volumes_and_labels, extract_reports_with_known_t2_label_agreed_between_annotators,\
     extract_validation_data, select_model, remove_val_and_test_subs_used_in_baseline, create_dir_if_not_exist


def classification_t2_volumes_difference(df_comparative_dates_path: str,
                                         input_dir: str,
                                         batch_size: int,
                                         nb_workers: int,
                                         nb_epochs: int,
                                         learning_rate: float,
                                         out_dir: str,
                                         val_interval: int,
                                         network: str,
                                         binary_classification: bool,
                                         annotation_type: str,
                                         legend_label: str,
                                         class_weights: list,
                                         nb_annotators: int,
                                         percentage_val_subs: float,
                                         fold_to_do: int,
                                         path_to_output_baseline_dir: str,
                                         ext_cv_folds: int,
                                         patience: int) -> None:
    """This function extracts the image pairs where both annotators agree for the T2 conclusion and invokes the actual classification function
    Args:
        df_comparative_dates_path: path to dataframe which contain the link between current and comparative exams
        input_dir: path where the volume differences are saved
        batch_size: batch size
        nb_workers: number of workers to use in parallel operations
        nb_epochs: number of training epochs
        learning_rate: learning rate to use during training
        out_dir: output folder
        val_interval: epoch frequency with which we print/save validation metrics
        network: model to use for training and inference
        binary_classification: whether to perform binary classification or not
        annotation_type: used to distinguish between the two annotation schemes (manual and automatic)
        legend_label: label to use in the legend of figures
        class_weights: weights to use during training; used to penalize more for minority class
        nb_annotators: number of human annotators
        percentage_val_subs: percentage of subjects used for validation
        fold_to_do: cross-validation fold to do (used so that we can run multiple folds in parallel in the HPC cluster)
        path_to_output_baseline_dir: path to directory containing the outputs of the baseline classification
        ext_cv_folds: number of cross validation folds performed
        patience: number of epochs to wait for early-stopping if there is no improvement across epochs
    Returns:
        None
    """
    date = datetime.today().strftime('%b_%d_%Y')  # type: str # save today's date
    out_dir = os.path.join(out_dir, "{}_{}_lr_{}_{}".format(legend_label,
                                                            network,
                                                            str(learning_rate).replace(".", "_"),
                                                            date))  # type: str # update name of output directory

    create_dir_if_not_exist(out_dir)  # if output dir does not exist, create it

    # copy config file to output dir so that we know every time which were the input parameters
    config_file_path = sys.argv[2]  # type: str
    config_out_path = os.path.join(out_dir, "config_file.json")  # type: str
    if not os.path.exists(config_out_path):
        copyfile(config_file_path, config_out_path)

    # set names of columns of interest that we want to extract
    df_rows_with_known_t2_label = extract_reports_with_known_t2_label_agreed_between_annotators(df_comparative_dates_path,
                                                                                                binary_classification,
                                                                                                nb_annotators,
                                                                                                annotation_type)  # type: pd.DataFrame

    all_subs, volume_differences_t2, classification_labels, median_shape = extract_volumes_and_labels(input_dir,
                                                                                                      df_rows_with_known_t2_label,
                                                                                                      annotation_type)
    print("\nMedian shape: {}".format(median_shape))

    nb_output_classes = len(extract_unique_elements(classification_labels))  # type: int # find number of unique classes

    # define monai data augmentation
    augmentation_transforms = Compose([RandFlipd(keys="volume", prob=0.2, spatial_axis=None),
                                       RandGaussianNoised(keys="volume", prob=0.2, mean=0.0, std=0.1),
                                       RandZoomd(keys="volume", prob=0.2, min_zoom=0.7, max_zoom=1.3),
                                       Rand3DElasticd(keys="volume", prob=0.2, sigma_range=(5, 7), magnitude_range=(50, 150), padding_mode="zeros")])

    # define preprocessing and transforms for training volumes
    train_transforms = Compose([LoadImaged(keys="volume"),
                                AddChanneld(keys="volume"),  # add channel; monai expects channel-first tensors
                                Resized(keys="volume", spatial_size=median_shape),  # resize all volumes to median volume shape
                                Lambda(lambda x: augmentation_transforms(x)),  # apply data augmentation(s)
                                EnsureTyped(keys="volume")])  # ensure that input data is either a PyTorch Tensor or np array

    # define preprocessing and transforms for validation/test volumes
    val_transforms = Compose([LoadImaged(keys="volume"),
                              AddChanneld(keys="volume"),  # add channel; monai expects channel-first tensors
                              Resized(keys="volume", spatial_size=median_shape),  # resize all volumes to median volume shape
                              EnsureTyped(keys="volume")])  # ensure that input data is either a PyTorch Tensor or np array

    # choose network, loss function and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # if available, set GPU to use during training loop; otherwise, use cpu
    conv_filters_vgg, fc_nodes_vgg = (4, 8, 16, 32), (256, 128, 32)
    model, _ = select_model(network, device, nb_output_classes, conv_filters_vgg, fc_nodes_vgg)

    # define loss function with weights to use during training; the higher the weight for one class the higher the loss;
    # we "force" the net to learn more on the minority class
    class_weights = torch.Tensor(class_weights).to(device)  # type: torch.Tensor # cast from list to tensor and set device
    loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)  # define loss function

    all_subs = remove_val_and_test_subs_used_in_baseline(all_subs, fold_to_do, path_to_output_baseline_dir)

    # select portion of training subs (and corresponding sessions) for validation
    int_train_subs, val_subs, x_int_train, x_val, y_int_train, y_val = extract_validation_data(all_subs, volume_differences_t2, classification_labels, percentage_val_subs)
    print("Nb. int_train_subs: {}; nb. diffmaps: {}".format(len(int_train_subs), len(x_int_train)))
    print("Nb. val_subs: {}; nb. diffmaps: {}".format(len(val_subs), len(x_val)))
    print("\nFirst ten val_subs: {} ...".format(sorted(val_subs)[:10]))

    # check labels' distribution
    unique_y_int_train, cnt_y_int_train = np.unique(y_int_train, return_counts=True)
    unique_y_val, cnt_y_val = np.unique(y_val, return_counts=True)
    print("External training set: unique labels -> {}, counts -> {}".format(unique_y_int_train, cnt_y_int_train))
    print("Validation set: unique labels -> {}, counts -> {}".format(unique_y_val, cnt_y_val))

    # create list of dictionaries
    int_train_files = [{"volume": volume_name, 'label': label_name} for volume_name, label_name in zip(x_int_train, y_int_train)]
    val_files = [{"volume": volume_name, 'label': label_name} for volume_name, label_name in zip(x_val, y_val)]

    # create a training data loader
    int_train_ds = Dataset(data=int_train_files, transform=train_transforms)
    int_train_loader = DataLoader(int_train_ds, batch_size=batch_size, shuffle=True, num_workers=nb_workers, pin_memory=torch.cuda.is_available())

    # create a validation data loader
    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=nb_workers, pin_memory=torch.cuda.is_available())

    # start typical PyTorch training
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # define optimizer and learning rate

    _ = training_loop_with_validation(nb_epochs, model, int_train_loader, device, optimizer, loss_function,
                                      val_loader, out_dir, date, fold_to_do, ext_cv_folds, val_interval, patience, legend_label)


def main():
    # the code inside here is run only when THIS script is run, and not just imported
    config_dict = load_config_file()  # load input config file with argparser

    # extract training args
    fold_to_do = config_dict['fold_to_do']
    network = config_dict['network']
    assert network in ("customVGG", "seresnext50"), "Only 'customVGG' and 'seresnext50' are allowed: found '{}' instead".format(network)
    df_comparative_dates_path = config_dict['df_comparative_dates_path']  # path to dataframe containing comparative dates and reports
    input_dir = config_dict['input_dir']  # path to folder containing volume differences
    out_dir = config_dict['out_dir']  # path where we save model parameters
    path_to_output_baseline_dir = config_dict['path_to_output_baseline_dir']

    batch_size = config_dict['batch_size']  # batch size used during training
    nb_epochs = config_dict['nb_epochs']  # number of training epochs
    learning_rate = config_dict['learning_rate']  # learning rate used during training
    val_interval = config_dict['val_interval']  # epoch frequency with which we print/save validation metrics
    binary_classification = str2bool(config_dict['binary_classification'])  # type: bool # whether to perform binary classification or not
    annotation_type = config_dict['annotation_type']  # type: str # used to distinguish between the two annotation schemes (manual and automatic)
    legend_label = config_dict['legend_label']  # type: str # label to use in the legend of figures
    class_weights = config_dict['class_weights']  # type: list # to use during training to weight more one class wrt another
    nb_annotators = config_dict['nb_annotators']  # type: int # number of human annotators
    percentage_val_subs = config_dict['percentage_val_subs']  # type: float
    ext_cv_folds = config_dict['ext_cv_folds']
    patience = config_dict['patience']

    # set number of jobs to run in parallel
    on_hpc_cluster = getpass.getuser() in ['to5743']  # type: bool # check if user is in list of authorized users
    if on_hpc_cluster:  # if we are running in the HPC
        assert torch.cuda.is_available(), "We expect to have a GPU when running on the HPC"
        nb_workers = 4  # type: int # number of jobs to run in parallel
    else:  # if instead we run the script locally
        nb_workers = 1  # type: int # number of jobs to run in parallel

    classification_t2_volumes_difference(df_comparative_dates_path,
                                         input_dir,
                                         batch_size,
                                         nb_workers,
                                         nb_epochs,
                                         learning_rate,
                                         out_dir,
                                         val_interval,
                                         network,
                                         binary_classification,
                                         annotation_type,
                                         legend_label,
                                         class_weights,
                                         nb_annotators,
                                         percentage_val_subs,
                                         fold_to_do,
                                         path_to_output_baseline_dir,
                                         ext_cv_folds,
                                         patience)


if __name__ == '__main__':
    main()
