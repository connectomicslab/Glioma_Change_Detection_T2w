import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
from typing import Tuple
from matplotlib.ticker import MaxNLocator
import monai
import torch
from utils_tdinoto.utils_lists import flatten_list, list_has_duplicates, extract_unique_elements,\
    shuffle_two_lists_with_same_order, split_list_equal_sized_groups, load_list_from_disk_with_pickle
from utils_tdinoto.numeric import round_half_up
import nibabel as nib
import torch.nn as nn
import random
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc
from torch.utils.tensorboard import SummaryWriter


__author__ = "Tommaso Di Noto"
__version__ = "0.0.1"
__email__ = "tommydino@hotmail.it"
__status__ = "Prototype"


def interpolate_val_values(val_metric: list,
                           val_interval: int,
                           x_axis_train: np.ndarray) -> list:
    """This function interpolates the input validation metric because this might not have the same length of the training metrics.
    This happens because we only compute validation metrics every val_interval epochs.
    Args:
        val_metric: validation metric that we want to interpolate
        val_interval: epoch frequency with which we compute validation metrics
        x_axis_train: x axis corresponding to training metric
    Returns:
        y_new_list: interpolated validation metrics with same length of training metrics
    """
    n = len(val_metric)  # compute length of validation metric
    x = np.arange(0, val_interval * n, val_interval)  # compute x_axis of validation metric
    y_new = np.interp(x_axis_train, x, val_metric)  # interpolate using the x axis of the training metric(s)
    y_new_list = list(y_new)  # convert to list
    return y_new_list


def save_train_loss_curves(train_loss: list,
                           out_dir: str,
                           out_filename: str) -> None:
    """ This function plots the training loss curve
    Args:
        train_loss: training loss
        out_dir: dir where we want to save the image
        out_filename: name of output file
    """
    create_dir_if_not_exist(out_dir)  # if output directory does not exist, create it

    out_image_path = os.path.join(out_dir, out_filename)  # create full path for saving the image

    x_axis = np.arange(1, len(train_loss) + 1, 1)  # since the input vectors have same length, just use one of them to extract epochs
    fig, ax1 = plt.subplots()  # create figure
    color_1 = 'tab:red'
    ax1.plot(x_axis, train_loss, color=color_1, label='Train loss')
    ax1.tick_params(axis='y', labelcolor=color_1)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel("Loss")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # only keep integers in x axis

    fig.suptitle('Train loss Curves', fontsize=16, fontweight='bold'), fig.legend(loc="upper right")
    fig.savefig(out_image_path)  # save the full figure


def save_loss_curves(train_loss: list,
                     val_loss: list,
                     image_dir: str,
                     image_filename: str) -> None:
    """ This function plots training and validation loss curves
    Args:
        train_loss: training loss
        val_loss: validation loss
        image_dir: path where we want to save the image
        image_filename: name of image
    """
    create_dir_if_not_exist(image_dir)  # if output dir does not exist, create it

    x_axis = np.arange(1, len(train_loss) + 1, 1)  # since the input vectors have same length, just use one of them to extract epochs
    assert len(train_loss) == len(val_loss), "We expect to have the same length for train_loss and val_loss"
    # val_loss_interp = interpolate_val_values(val_loss, val_interval, x_axis)
    # assert len(val_loss_interp) == len(x_axis), "Length mismatch: len(val_loss_interp) = {}, len(x_axis) = {}".format(len(val_loss_interp), len(x_axis))

    fig, ax1 = plt.subplots()  # create figure
    color_1 = 'tab:red'
    ax1.plot(x_axis, train_loss, color=color_1, label='Train loss')
    ax1.plot(x_axis, val_loss, "--", color=color_1, label='Val loss')

    idx_min = first_argmin(val_loss)  # find index of minimum value
    ax1.plot(x_axis[idx_min], min(val_loss), 'ro', markersize="10", label='min val_loss = {:.4f}'.format(min(val_loss)))  # highlight maximum value in the plot

    ax1.tick_params(axis='y', labelcolor=color_1)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel("Loss")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # only keep integers in x axis
    ax1.set_ylim([0, 1.3])

    fig.suptitle('Train/Val loss Curves', fontsize=16, fontweight='bold'), fig.legend(loc="upper right")
    image_path = os.path.join(image_dir, image_filename)
    fig.savefig(image_path)  # save the full figure


def save_train_val_metrics(val_accuracy: list,
                           val_weighted_f1: list,
                           image_dir: str,
                           image_filename: str) -> None:
    """This function plots the train/val metrics
    Args:
        val_accuracy: validation accuracy
        val_weighted_f1: validation weighted f1 score
        image_dir: path where we want to save the image
        image_filename: image filename
    """
    create_dir_if_not_exist(image_dir)  # if output dir does not exist, create it

    x_axis = np.arange(1, len(val_accuracy) + 1, 1)  # since the two input vectors have same length, just use one of the two to extract epochs
    fig2, ax1 = plt.subplots()  # create figure
    color_1 = 'tab:green'
    color_2 = 'tab:blue'
    ax1.plot(x_axis, val_accuracy, "--", color=color_1, label='Val accuracy')
    ax1.plot(x_axis, val_weighted_f1, "--", color=color_2, label='Val weighted f1-score')
    idx_max = first_argmax(val_weighted_f1)  # find index of maximum value
    ax1.plot(x_axis[idx_max], max(val_weighted_f1), 'ro', markersize="10", label='max val_f1 = {:.4f}'.format(max(val_weighted_f1)))  # highlight maximum value in the plot
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Validation metrics')
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # only keep integers in x axis

    fig2.suptitle('Val Curves', fontsize=16, fontweight='bold'), fig2.legend(loc="upper right")
    image_path = os.path.join(image_dir, image_filename)
    fig2.savefig(image_path)  # save the full figure


def extract_volumes_and_labels_add_automatic_sub_ses(input_dir: str,
                                                     df_rows_with_known_label: pd.DataFrame,
                                                     val_subs: list,
                                                     test_subs: list,
                                                     annotation_type: str) -> Tuple[list, list, list, np.ndarray]:
    """This function extracts the difference volumes, the corresponding labels and the median volume shape.
    Args:
        input_dir: path where the volume differences are saved
        df_rows_with_known_label: it contains the mapping between previous and current sessions
        val_subs: validation subjects; used to exclude these from the added subs
        test_subs: test subjects; we want to exclude these from the added subs
        annotation_type: used to distinguish between the two annotation schemes (manual and automatic)
    Returns:
        all_subs: it contains the subs ipps
        volume_differences_t2: it contains the paths to all difference volumes
        classification_label: it contains the corresponding labels
        median_shape: median volume shape across all volume differences
    """
    all_subs = []
    volume_differences_t2 = []
    classification_label = []
    volume_shapes = []
    for sub in sorted(os.listdir(input_dir)):
        # ensure that we do not use sessions from test subjects
        if sub not in test_subs and sub not in val_subs:
            for ses_diff in sorted(os.listdir(os.path.join(input_dir, sub))):
                t2_difference_volume_path = os.path.join(input_dir, sub, ses_diff, "{}_{}_difference_t2_volumes.nii.gz".format(sub, ses_diff))

                assert os.path.exists(t2_difference_volume_path), "Path {} does not exist".format(t2_difference_volume_path)
                sub_only_numbers = int(re.findall(r"\d+", sub)[0])  # extract sub number
                sessions = re.findall(r"\d+", ses_diff)  # type: list # extract session numbers and sort such that comparative is always the first and current is always the second

                # make sure that exactly two sessions were found and the comparative is lower than the current
                if len(sessions) == 2 and sessions[0] < sessions[1]:
                    # select only the row corresponding to this sub and ses diff
                    row_of_interest = df_rows_with_known_label.loc[(df_rows_with_known_label["ipp"] == sub_only_numbers)
                                                                   & (df_rows_with_known_label["comparative_date"] == int(sessions[0]))
                                                                   & (df_rows_with_known_label["exam_date"] == int(sessions[1]))]

                    if row_of_interest.shape[0] == 1:  # if the sub-dataframe has exactly one row (i.e. there is only one match of this sub-ses_diff combination)
                        if os.path.exists(t2_difference_volume_path):  # if the path to the T2 difference map exists

                            all_subs.append(sub)  # append sub

                            if annotation_type == 'manual':
                                t2_label = row_of_interest["t2_label_A1"].values[0]  # extract label for this exam pair
                            elif annotation_type == 'automatic':
                                t2_label = row_of_interest["t2_label_random_forest"].values[0]  # extract label for this exam pair
                            else:
                                raise ValueError("annotation_type can be either 'manual' or 'automatic'; instead {} was passed".format(annotation_type))

                            # check volume shape
                            diff_volume_obj = nib.load(t2_difference_volume_path)  # load volume as nibabel object
                            diff_volume = np.asanyarray(diff_volume_obj.dataobj)  # convert to numpy array
                            volume_shapes.append(diff_volume.shape)  # append shapes to external list

                            # append volumes and labels
                            volume_differences_t2.append(t2_difference_volume_path)
                            classification_label.append(t2_label)
                        else:
                            print("Discarded {}_{} cause t2_difference_volume_path does not exist".format(sub, ses_diff))

                    elif row_of_interest.shape[0] > 1:
                        raise ValueError("There should not be duplicate rows in the dataframe")
                    elif row_of_interest.shape[0] == 0:
                        # print("Discarded {}_{} cause not present in the dataframe".format(sub, ses_diff))
                        pass
                    else:
                        raise ValueError("Shape of row_of_interest should not be negative")
                else:
                    print("Discarded {}_{}: len(sessions) = {}, sessions[0] = {}; sessions[1] = {}".format(sub, ses_diff, len(sessions), sessions[0], sessions[1]))

    all_subs = extract_unique_elements(all_subs, ordered=True)  # remove duplicates
    volume_shapes_np = np.asarray(volume_shapes)  # type: np.ndarray # cast from list to numpy array
    median_shape = np.median(volume_shapes_np, axis=0).astype(np.int32)  # type: np.ndarray # extract median shape

    assert len(volume_differences_t2) == len(classification_label) != 0, "volume and labels lists must have same shape and they must be non-empty"

    return all_subs, volume_differences_t2, classification_label, median_shape


def extract_volumes_and_labels_brats_tcia(input_dir: str,
                                          df_sub_ses_label: pd.DataFrame,
                                          annotation_type: str) -> Tuple[list, list, list]:
    """
    Args:
        input_dir: path where the volume differences are stored
        df_sub_ses_label: dataframe containing the labels
        annotation_type: used to distinguish between the two annotation schemes (manual and automatic)
    Returns:
        all_subs: it contains the subs ipps
        volume_differences_t2: it contains the paths to all difference volumes
        classification_label: it contains the corresponding labels
    """
    all_subs = []
    volume_differences_t2 = []
    classification_label = []
    for sub in sorted(os.listdir(input_dir)):
        for ses_diff in sorted(os.listdir(os.path.join(input_dir, sub))):
            t2_difference_volume_path = os.path.join(input_dir, sub, ses_diff, "{}_{}_difference_t2_volumes.nii.gz".format(sub, ses_diff))

            sessions = re.findall(r"\d+", ses_diff)  # type: list # extract session numbers

            # make sure that exactly two sessions were found and the comparative is lower than the current
            if len(sessions) == 2 and sessions[0] < sessions[1]:
                # select only the row corresponding to this sub and ses diff
                row_of_interest = df_sub_ses_label.loc[(df_sub_ses_label["sub"] == sub)
                                                       & (df_sub_ses_label["ses_diff"] == ses_diff)]

                if row_of_interest.shape[0] == 1:  # if the sub-dataframe has exactly one row (i.e. there is only one match of this sub-ses_diff combination)
                    if os.path.exists(t2_difference_volume_path):  # if the path to the T2 difference map exists

                        all_subs.append(sub)  # append sub

                        if annotation_type == 'manual':
                            t2_label = row_of_interest["t2_label_numeric"].values[0]  # extract label for this exam pair
                        else:
                            raise ValueError("annotation_type must be 'manual' for brats-tcia patients; instead {} was passed".format(annotation_type))

                        # append volumes and labels
                        volume_differences_t2.append(t2_difference_volume_path)
                        classification_label.append(t2_label)
                    else:
                        print("Discarded {}_{} cause t2_difference_volume_path does not exist".format(sub, ses_diff))

                elif row_of_interest.shape[0] > 1:
                    raise ValueError("There should not be duplicate rows in the dataframe")
                elif row_of_interest.shape[0] == 0:
                    # print("Discarded {}_{} cause not present in the dataframe".format(sub, ses_diff))
                    pass
                else:
                    raise ValueError("Shape of row_of_interest should not be negative")

    all_subs = extract_unique_elements(all_subs, ordered=True)  # remove duplicates

    assert len(volume_differences_t2) == len(classification_label) != 0, "volume and labels lists must have same shape and they must be non-empty"

    return all_subs, volume_differences_t2, classification_label


def extract_volumes_and_labels(input_dir: str,
                               df_rows_with_known_label: pd.DataFrame,
                               annotation_type: str) -> Tuple[list, list, list, np.ndarray]:
    """This function extracts the difference volumes, the corresponding labels and the median volume shape.
    Args:
        input_dir: path where the volume differences are stored
        df_rows_with_known_label: it contains the mapping between previous and current sessions
        annotation_type: used to distinguish between the two annotation schemes (manual and automatic)
    Returns:
        all_subs: it contains the subs ipps
        volume_differences_t2: it contains the paths to all difference volumes
        classification_label: it contains the corresponding labels
        median_shape: median volume shape across all volume differences
    """
    all_subs = []
    volume_differences_t2 = []
    classification_label = []
    volume_shapes = []
    for sub in sorted(os.listdir(input_dir)):
        for ses_diff in sorted(os.listdir(os.path.join(input_dir, sub))):
            t2_difference_volume_path = os.path.join(input_dir, sub, ses_diff, "{}_{}_difference_t2_volumes.nii.gz".format(sub, ses_diff))

            sub_only_numbers = int(re.findall(r"\d+", sub)[0])  # extract sub number
            sessions = re.findall(r"\d+", ses_diff)  # type: list # extract session numbers

            # make sure that exactly two sessions were found and the comparative is lower than the current
            if len(sessions) == 2 and sessions[0] < sessions[1]:
                # select only the row corresponding to this sub and ses diff
                row_of_interest = df_rows_with_known_label.loc[(df_rows_with_known_label["ipp"] == sub_only_numbers)
                                                               & (df_rows_with_known_label["comparative_date"] == int(sessions[0]))
                                                               & (df_rows_with_known_label["exam_date"] == int(sessions[1]))]

                if row_of_interest.shape[0] == 1:  # if the sub-dataframe has exactly one row (i.e. there is only one match of this sub-ses_diff combination)
                    if os.path.exists(t2_difference_volume_path):  # if the path to the T2 difference map exists

                        all_subs.append(sub)  # append sub

                        if annotation_type == 'manual':
                            t2_label = row_of_interest["t2_label_A1"].values[0]  # extract label for this exam pair
                        elif annotation_type == 'automatic':
                            t2_label = row_of_interest["t2_label_random_forest"].values[0]  # extract label for this exam pair
                        else:
                            raise ValueError("annotation_type can be either 'manual' or 'automatic'; instead {} was passed".format(annotation_type))

                        # check volume shape
                        diff_volume_obj = nib.load(t2_difference_volume_path)  # load volume as nibabel object
                        diff_volume = np.asanyarray(diff_volume_obj.dataobj)  # convert to numpy array
                        volume_shapes.append(diff_volume.shape)  # append shapes to external list

                        # append volumes and labels
                        volume_differences_t2.append(t2_difference_volume_path)
                        classification_label.append(t2_label)
                    else:
                        print("Discarded {}_{} cause t2_difference_volume_path does not exist".format(sub, ses_diff))

                elif row_of_interest.shape[0] > 1:
                    raise ValueError("There should not be duplicate rows in the dataframe")
                elif row_of_interest.shape[0] == 0:
                    # print("Discarded {}_{} cause not present in the dataframe".format(sub, ses_diff))
                    pass
                else:
                    raise ValueError("Shape of row_of_interest should not be negative")

    all_subs = extract_unique_elements(all_subs, ordered=True)  # remove duplicates
    volume_shapes_np = np.asarray(volume_shapes)  # type: np.ndarray # cast from list to numpy array
    median_shape = np.median(volume_shapes_np, axis=0).astype(np.int32)  # type: np.ndarray # extract median shape

    assert len(volume_differences_t2) == len(classification_label) != 0, "volume and labels lists must have same shape and they must be non-empty"

    return all_subs, volume_differences_t2, classification_label, median_shape


def extract_train_and_test_volumes_and_labels(all_subs: list,
                                              train_idxs: list,
                                              test_idxs: list,
                                              volume_differences: list,
                                              labels: list) -> Tuple[list, list, list, list, list, list]:
    """This function extracts training and test volumes and labels
    Args:
        all_subs: it contains all subjects
        train_idxs: it contains the indexes of the train subjects
        test_idxs: it contains the indexes of the test subjects
        volume_differences: it contains the paths to all the difference volumes
        labels: it contains the labels of all difference volumes
    Returns:
        train_subs: train subjects
        test_subs: test subjects
        train_volume_differences: difference volumes of the training subjects
        train labels: labels corresponding to the train volumes
        test_volume_differences: difference volumes of the test subjects
        test_labels: labels corresponding to the test volumes
    """
    # extract train and test subs
    train_subs = [all_subs[i] for i in train_idxs]
    test_subs = [all_subs[j] for j in test_idxs]

    # extract difference volumes and labels corresponding to the train subjects
    train_volume_differences = [volume for volume in volume_differences if any(sub in volume for sub in train_subs)]
    idxs_train_volumes = [idx for idx, value in enumerate(volume_differences) if value in train_volume_differences]
    train_labels = [labels[idx] for idx in idxs_train_volumes]

    # extract difference volumes corresponding to the test subjects
    test_volume_differences = [volume for volume in volume_differences if any(sub in volume for sub in test_subs)]
    idxs_test_volumes = [idx for idx, value in enumerate(volume_differences) if value in test_volume_differences]
    test_labels = [labels[idx] for idx in idxs_test_volumes]

    return train_subs, test_subs, train_volume_differences, train_labels, test_volume_differences, test_labels


def extract_volumes_and_labels_from_data_loader(batch_data: list,
                                                device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """This function extracts the volumes and the labels from a batch
    Args:
        batch_data: one batch of dataset
        device: device where we are running the operations
    Returns:
        inputs: the volumes of the batch
        labels: the labels of the batch
    """
    input_tensors = [x["volume"] for x in batch_data]  # type: list # extract volume tensors
    input_labels = [x["label"] for x in batch_data]  # type: list # extract labels
    inputs = torch.stack(input_tensors, dim=0).to(device)  # type: torch.Tensor # stack volumes into one unique tensor
    labels = torch.tensor(input_labels).to(device)  # type: torch.Tensor # convert labels to tensor

    return inputs, labels


def print_running_time(start_time: float,
                       end_time: float,
                       process_name: str) -> None:
    """This function takes as input the start and the end time of a process and prints to console the time elapsed for this process
    Args:
        start_time: instant when the timer is started
        end_time: instant when the timer was stopped
        process_name: name of the process
    """
    sentence = str(process_name)  # convert to string whatever the user inputs as third argument
    temp = end_time - start_time  # compute time difference
    hours = temp // 3600  # compute hours
    temp = temp - 3600 * hours  # if hours is not zero, remove equivalent amount of seconds
    minutes = temp // 60  # compute minutes
    seconds = temp - 60 * minutes  # compute minutes
    print('\n%s time: %d hh %d mm %d ss' % (sentence, hours, minutes, seconds))


def training_loop_with_validation(nb_epochs: int,
                                  model: monai.networks.nets,
                                  train_data_loader: torch.utils.data.dataloader.DataLoader,
                                  device: torch.device,
                                  optimizer: torch.optim.Adam,
                                  loss_function: torch.nn.modules.loss,
                                  validation_loader: torch.utils.data.dataloader.DataLoader,
                                  out_dir: str,
                                  date: str,
                                  ext_cv_fold_counter: int,
                                  ext_cv_folds: int,
                                  val_interval: int,
                                  patience: int,
                                  legend_label: str) -> str:
    """This function performs the training loop
    Args:
        nb_epochs: number of training epochs
        model: model to use
        train_data_loader: data loader
        device: device on which training will be performed
        optimizer:  chosen optimizer
        loss_function: loss function
        validation_loader: data loader for inference
        out_dir: path where we save the training outputs
        date: today's date
        ext_cv_fold_counter: external cross validation fold that we are carrying out
        ext_cv_folds: number of cross validation folds performed
        val_interval: epoch frequency with which we print/save validation metrics
        patience: number of epochs to wait for early-stopping if there is no improvement across epochs
        legend_label: label to use for naming output files/dirs
    Returns:
        out_filename: path to best model state
    """
    out_dir_one_cv_fold = os.path.join(out_dir, "fold{}".format(ext_cv_fold_counter))  # create directory for this cross-validation fold
    create_dir_if_not_exist(out_dir_one_cv_fold)  # if output folder does not exist, create it

    out_dir_tb = os.path.join(out_dir_one_cv_fold, "tensorboard")  # create directory for tensorboard files
    create_dir_if_not_exist(out_dir_tb)  # if output folder does not exist, create it

    tb = SummaryWriter(log_dir=out_dir_tb)  # instantiate summary writer object for tensorboard
    # best_val_loss = 1_000_000.  # set initial loss extremely high such it will be modified at least once
    best_val_auc = -1  # set initial AUC value negative such that it will be modified at least once
    epoch_loss_values = []  # type: list # it will contain the training loss over the epochs
    val_accuracy_values = []  # type: list # it will contain the validation accuracies
    val_weighted_f1_values = []  # type: list # it will contain the validation weighted f1 scores
    val_loss_values = []  # type: list # it will containt the validation losses
    out_filename = os.path.join(out_dir_one_cv_fold, "best_model.pth")  # type: str # define output filename where we save the best model state
    trigger_times = 0  # counter used for early stopping; if this surpasses the patience, then training is early-stopped

    for epoch in range(nb_epochs):  # loop over epochs
        # start_epoch = time.time()  # start timer
        model.train()  # set the model in training mode
        epoch_loss = 0  # initialize loss
        step = 0  # initialize step

        for batch_data in train_data_loader:  # loop over batches
            step += 1  # increment step

            # extract volumes and labels from batch
            inputs, labels = (batch_data["volume"].to(device), batch_data["label"].to(device))

            input_filename = batch_data["volume_meta_dict"]["filename_or_obj"]
            assert not torch.any(inputs.isnan()), "Input tensor contains nans for {}".format(input_filename)

            optimizer.zero_grad()  # explicitly set the gradients to zero before starting back-propagation
            outputs = model(inputs)  # type: torch.Tensor # compute forward pass
            loss = loss_function(outputs, labels)  # type: torch.Tensor # compute loss
            loss.backward()  # compute dloss/dx for every parameter x which has requires_grad=True
            optimizer.step()  # update model parameters
            epoch_loss += loss.item()  # accumulate loss from this batch

        epoch_loss /= step  # compute average loss of this epoch across batches
        epoch_loss_values.append(epoch_loss)

        # add also histograms of model weights for tensorboard visualization
        for name, weight in model.named_parameters():
            if weight.requires_grad:  # if requires_grad == True
                tb.add_histogram(name, weight, epoch)
                tb.add_histogram(f'{name}.grad', weight.grad, epoch)

        # compute validation metrics
        if (epoch + 1) % val_interval == 0:  # if remainder is zero
            model.eval()  # set model in inference mode
            with torch.no_grad():  # set all requires_grad flags to False
                num_correct = 0.0
                metric_count = 0
                val_epoch_loss = 0
                val_step = 0
                all_val_predictions_binary = []  # will contain the binary predictions
                all_val_predictions_probab = []  # will contain the probabilistic predictions
                all_labels = []
                for val_data in validation_loader:  # loop over batches
                    val_step += 1  # increment step

                    # extract volumes and labels from batch
                    val_volumes, val_labels = (val_data["volume"].to(device), val_data["label"].to(device))
                    # val_volumes, val_labels = extract_volumes_and_labels_from_data_loader(val_data, device)

                    val_outputs = model(val_volumes)  # type: torch.Tensor # compute forward pass
                    val_loss = loss_function(val_outputs, val_labels)  # type: torch.Tensor # compute loss
                    val_epoch_loss += val_loss.item()  # accumulate loss from this batch

                    # extract output predictions (binary and probabilistic)
                    val_outputs_binary = val_outputs.argmax(dim=1)  # type: torch.Tensor  # find argmax along rows (i.e. for each sample)
                    val_outputs_np_binary = val_outputs_binary.cpu().numpy()  # type: np.ndarray
                    val_outputs_np_probab = val_outputs.cpu().detach().numpy()[:, 1]  # type: np.ndarray  # extract probabilistic output for the positive class

                    # compute validation accuracy as correct samples / all samples
                    value = torch.eq(val_outputs_binary, val_labels)  # type: torch.Tensor # compute element-wise equality between predictions and labels
                    num_correct += value.sum().item()  # add number of correctly classified examples
                    metric_count += len(value)  # increment counter with length of this batch

                    all_val_predictions_binary.append(val_outputs_np_binary)  # append binary predictions to external list
                    all_val_predictions_probab.append(val_outputs_np_probab)  # append probabilistic predictions to external list
                    all_labels.append(val_labels.cpu().numpy())  # extract the labels as numpy array and append to external list

                # ---------------------------- compute validation metrics
                # ------ validation accuracy
                val_accuracy = num_correct / metric_count  # compute accuracy
                val_accuracy_values.append(val_accuracy)

                # ------ validation loss
                val_epoch_loss /= val_step  # compute average loss of this epoch across batches
                val_loss_values.append(val_epoch_loss)

                # ------ validation F1-score
                val_weighted_f1 = f1_score(y_true=flatten_list(all_labels), y_pred=flatten_list(all_val_predictions_binary), average='weighted')  # type: float # compute weighted f1 score
                val_weighted_f1_values.append(val_weighted_f1)

                # ------ validation AUC score
                _, _, val_auc = plot_roc_curve(flatten_list(all_labels),
                                               flatten_list(all_val_predictions_probab),
                                               ext_cv_folds,
                                               out_path="",  # set empty path since we are not interested in saving the figure here
                                               legend_label=legend_label,
                                               plot=False,  # don't plot
                                               save=False)  # don't save figure to disk

                # choose best model according to one of the validation metrics
                if val_auc > best_val_auc:  # if the validation auc is higher than the previous best value
                    best_val_auc = val_auc  # update best value
                    best_metric_epoch = epoch + 1  # save epoch corresponding to best value
                    torch.save(model.state_dict(), out_filename)  # save model state (i.e. parameters), overwriting previous ones
                    trigger_times = 0  # reset counter to 0
                else:  # if instead the validation auc is equal or lower than the previous best value
                    trigger_times += 1  # increase counter

                    if trigger_times >= patience:  # if the counter is greater or equal wrt patience --> then we early stop training (it's pointless to continue)
                        print("epoch {}/{} train loss: {:.4f}, val loss: {:.4f}, val f1: {:.4f}, val AUC: {:.4f}, triggers: {}".format(epoch + 1,
                                                                                                                                       nb_epochs,
                                                                                                                                       epoch_loss,
                                                                                                                                       val_epoch_loss,
                                                                                                                                       val_weighted_f1,
                                                                                                                                       val_auc,
                                                                                                                                       trigger_times))
                        print('\nEarly stopping!')
                        print("\nTrain completed, best val_auc: {:.4f} at epoch: {}".format(best_val_auc, best_metric_epoch))

                        # save training/validation curves
                        img_filename = "train_val_loss_ext_fold{}_{}.png".format(str(ext_cv_fold_counter), date)
                        save_loss_curves(epoch_loss_values, val_loss_values, out_dir_one_cv_fold, img_filename)
                        img_filename_2 = "val_metrics_ext_fold{}_{}.png".format(str(ext_cv_fold_counter), date)
                        save_train_val_metrics(val_accuracy_values, val_weighted_f1_values, out_dir_one_cv_fold, img_filename_2)

                        return out_filename

        print("epoch {}/{} train loss: {:.4f}, val loss: {:.4f}, val f1: {:.4f}, val AUC: {:.4f}, triggers: {}".format(epoch + 1,
                                                                                                                       nb_epochs,
                                                                                                                       epoch_loss,
                                                                                                                       val_epoch_loss,
                                                                                                                       val_weighted_f1,
                                                                                                                       val_auc,
                                                                                                                       trigger_times))

        # end_epoch = time.time()  # stop timer
        # print_running_time(start_epoch, end_epoch, "One epoch")

    # if we arrive here, it means that no early stopping was reached (i.e. all epochs were performed)
    print("\nTrain completed, best val_auc: {:.4f} at epoch: {}".format(best_val_auc, best_metric_epoch))

    # save training/validation curves
    img_filename = "train_val_loss_ext_fold{}_{}.png".format(str(ext_cv_fold_counter), date)
    save_loss_curves(epoch_loss_values, val_loss_values, out_dir_one_cv_fold, img_filename)
    img_filename_2 = "val_metrics_ext_fold{}_{}.png".format(str(ext_cv_fold_counter), date)
    save_train_val_metrics(val_accuracy_values, val_weighted_f1_values, out_dir_one_cv_fold, img_filename_2)

    return out_filename


def training_loop_with_validation_optuna(nb_epochs: int,
                                         model: monai.networks.nets,
                                         train_data_loader: torch.utils.data.dataloader.DataLoader,
                                         device: torch.device,
                                         optimizer: torch.optim.Adam,
                                         loss_function: torch.nn.modules.loss,
                                         validation_loader: torch.utils.data.dataloader.DataLoader,
                                         out_dir: str,
                                         date: str,
                                         ext_cv_fold_counter: int,
                                         ext_cv_folds: int,
                                         val_interval: int,
                                         patience: int,
                                         legend_label: str) -> Tuple[str, float]:
    """This function performs the training loop
    Args:
        nb_epochs: number of training epochs
        model: model to use
        train_data_loader: data loader
        device: device on which training will be performed
        optimizer:  chosen optimizer
        loss_function: loss function
        validation_loader: data loader for inference
        out_dir: path where we save the training outputs
        date: today's date
        ext_cv_folds: number of cross validation folds
        ext_cv_fold_counter: cross validation fold that we are carrying out
        val_interval: epoch frequency with which we print/save validation metrics
        patience: number of epochs to wait for early-stopping if there is no improvement across epochs
        legend_label: label to use for naming output files/dirs
    Returns:
        outs: path to saved weights and best validation F1-score
    """
    out_dir_one_cv_fold = os.path.join(out_dir, "fold{}".format(ext_cv_fold_counter))  # create directory for this cross-validation fold
    create_dir_if_not_exist(out_dir_one_cv_fold)

    out_dir_tb = os.path.join(out_dir_one_cv_fold, "tensorboard")  # create directory for tensorboard files
    create_dir_if_not_exist(out_dir_tb)

    tb = SummaryWriter(log_dir=out_dir_tb)  # instantiate summary writer object for tensorboard
    best_val_auc = -1  # set initial AUC value negative such that it will be modified at least once
    epoch_loss_values = []  # type: list # it will contain the training loss over the epochs
    val_accuracy_values = []  # type: list # it will contain the validation accuracies
    val_weighted_f1_values = []  # type: list # it will contain the validation weighted f1 scores
    val_loss_values = []  # type: list # it will containt the validation losses
    out_filename = os.path.join(out_dir_one_cv_fold, "best_model.pth")  # type: str # define output filename where we save the best model state
    trigger_times = 0  # counter used for early stopping; if this surpasses the patience, then training is early-stopped

    for epoch in range(nb_epochs):  # loop over epochs
        model.train()  # set the model in training mode
        epoch_loss = 0  # initialize loss
        step = 0  # initialize step

        for batch_data in train_data_loader:  # loop over batches
            step += 1  # increment step

            # extract volumes and labels from batch
            inputs, labels = (batch_data["volume"].to(device), batch_data["label"].to(device))

            input_filename = batch_data["volume_meta_dict"]["filename_or_obj"]
            assert not torch.any(inputs.isnan()), "Input tensor contains nans for {}".format(input_filename)

            optimizer.zero_grad()  # explicitly set the gradients to zero before starting back-propagation
            outputs = model(inputs)  # type: torch.Tensor # compute forward pass
            loss = loss_function(outputs, labels)  # type: torch.Tensor # compute loss
            loss.backward()  # compute dloss/dx for every parameter x which has requires_grad=True
            optimizer.step()  # update model parameters
            epoch_loss += loss.item()  # accumulate loss from this batch

        epoch_loss /= step  # compute average loss of this epoch across batches
        epoch_loss_values.append(epoch_loss)

        # add also histograms of model weights for tensorboard visualization
        for name, weight in model.named_parameters():
            if weight.requires_grad:  # if requires_grad == True
                tb.add_histogram(name, weight, epoch)
                tb.add_histogram(f'{name}.grad', weight.grad, epoch)

        # compute validation metrics
        if (epoch + 1) % val_interval == 0:  # if remainder is zero
            model.eval()  # set model in inference mode
            with torch.no_grad():  # set all requires_grad flags to False
                num_correct = 0.0
                metric_count = 0
                val_epoch_loss = 0
                val_step = 0
                all_val_predictions_binary = []  # will contain the binary predictions
                all_val_predictions_probab = []  # will contain the probabilistic predictions
                all_labels = []
                for val_data in validation_loader:  # loop over batches
                    val_step += 1  # increment step

                    # extract volumes and labels from batch
                    val_volumes, val_labels = (val_data["volume"].to(device), val_data["label"].to(device))

                    val_outputs = model(val_volumes)  # type: torch.Tensor # compute forward pass
                    val_loss = loss_function(val_outputs, val_labels)  # type: torch.Tensor # compute loss
                    val_epoch_loss += val_loss.item()  # accumulate loss from this batch

                    # extract output predictions (binary and probabilistic)
                    val_outputs_binary = val_outputs.argmax(dim=1)  # type: torch.Tensor  # find argmax along rows (i.e. for each sample)
                    val_outputs_np_binary = val_outputs_binary.cpu().numpy()  # type: np.ndarray
                    val_outputs_np_probab = val_outputs.cpu().detach().numpy()[:, 1]  # type: np.ndarray  # extract probabilistic output for the positive class

                    # compute validation accuracy as correct samples / all samples
                    value = torch.eq(val_outputs_binary, val_labels)  # type: torch.Tensor # compute element-wise equality between predictions and labels
                    num_correct += value.sum().item()  # add number of correctly classified examples
                    metric_count += len(value)  # increment counter with length of this batch

                    all_val_predictions_binary.append(val_outputs_np_binary)  # append binary predictions to external list
                    all_val_predictions_probab.append(val_outputs_np_probab)  # append probabilistic predictions to external list
                    all_labels.append(val_labels.cpu().numpy())  # extract the labels as numpy array and append to external list

                # ---------------------------- compute validation metrics
                # ------ validation accuracy
                val_accuracy = num_correct / metric_count  # compute accuracy
                val_accuracy_values.append(val_accuracy)

                # ------ validation loss
                val_epoch_loss /= val_step  # compute average loss of this epoch across batches
                val_loss_values.append(val_epoch_loss)

                # ------ validation F1-score
                val_weighted_f1 = f1_score(y_true=flatten_list(all_labels), y_pred=flatten_list(all_val_predictions_binary), average='weighted')  # type: float # compute weighted f1 score
                val_weighted_f1_values.append(val_weighted_f1)

                # ------ validation AUC score
                _, _, val_auc = plot_roc_curve(flatten_list(all_labels),
                                               flatten_list(all_val_predictions_probab),
                                               ext_cv_folds,
                                               out_path="",  # set empty path since we are not interested in saving the figure here
                                               legend_label=legend_label,
                                               plot=False,  # don't plot
                                               save=False)  # don't save figure to disk

                # choose best model according to one of the validation metrics
                if val_auc > best_val_auc:  # if the validation auc is higher than the previous best value
                    best_val_auc = val_auc  # update best value
                    best_metric_epoch = epoch + 1  # save epoch corresponding to best value
                    torch.save(model.state_dict(), out_filename)  # save model state (i.e. parameters), overwriting previous ones
                    trigger_times = 0  # reset counter to 0
                else:  # if instead the validation loss is equal or higher than the previous best value
                    trigger_times += 1  # increase counter

                    if trigger_times >= patience:  # if the counter is greater or equal wrt patience --> then we early stop training (it's pointless to continue)
                        print("epoch {}/{} train loss: {:.4f}, val loss: {:.4f}, val f1: {:.4f}, val AUC: {:.4f}, triggers: {}".format(epoch + 1,
                                                                                                                                       nb_epochs,
                                                                                                                                       epoch_loss,
                                                                                                                                       val_epoch_loss,
                                                                                                                                       val_weighted_f1,
                                                                                                                                       val_auc,
                                                                                                                                       trigger_times))
                        print('\nEarly stopping!')
                        print("\nTrain completed, best val_auc: {:.4f} at epoch: {}".format(best_val_auc, best_metric_epoch))

                        # save training/validation curves
                        img_filename = "train_val_loss_ext_fold{}_{}.png".format(str(ext_cv_fold_counter), date)
                        save_loss_curves(epoch_loss_values, val_loss_values, out_dir_one_cv_fold, img_filename)
                        img_filename_2 = "val_metrics_ext_fold{}_{}.png".format(str(ext_cv_fold_counter), date)
                        save_train_val_metrics(val_accuracy_values, val_weighted_f1_values, out_dir_one_cv_fold, img_filename_2)

                        # group outputs into one tuple
                        outs = (out_filename, best_val_auc)  # type: Tuple[str,float]

                        return outs

        print("epoch {}/{} train loss: {:.4f}, val loss: {:.4f}, val f1: {:.4f}, val AUC: {:.4f}, triggers: {}".format(epoch + 1,
                                                                                                                       nb_epochs,
                                                                                                                       epoch_loss,
                                                                                                                       val_epoch_loss,
                                                                                                                       val_weighted_f1,
                                                                                                                       val_auc,
                                                                                                                       trigger_times))

    print("\nTrain completed, best val_auc: {:.4f} at epoch: {}".format(best_val_auc, best_metric_epoch))

    # save training/validation curves
    img_filename = "train_val_loss_ext_fold{}_{}.png".format(str(ext_cv_fold_counter), date)
    save_loss_curves(epoch_loss_values, val_loss_values, out_dir_one_cv_fold, img_filename)
    img_filename_2 = "val_metrics_ext_fold{}_{}.png".format(str(ext_cv_fold_counter), date)
    save_train_val_metrics(val_accuracy_values, val_weighted_f1_values, out_dir_one_cv_fold, img_filename_2)

    # group outputs into one tuple
    outs = (out_filename, best_val_auc)  # type: Tuple[str,float]

    return outs


def run_inference(trained_model: monai.networks.nets,
                  test_data_loader: torch.utils.data.dataloader.DataLoader,
                  device: torch.device) -> Tuple[list, list, list, list]:
    """This function performs the inference on the validation/test dataset
    Args:
        trained_model: the trained model with which we perform inference
        test_data_loader: data loader of validation/test set
        device: device on which inference will be performed
    Returns:
        all_filenames_flat: it contains the filenames of the test subjects; we use this to check which were the misclassified examples
        all_predictions_binary_flat: it contains all the binary predictions of the validation/test set
        all_predictions_probab_flat: it contains all the probabilistic predictions of the validation/test set
        all_labels_flat: it contains all the labels of the validation/test set
    """
    all_filenames = []  # type: list # will store the filenames cause later we want to check which sub_ses were misclassified
    all_predictions_binary = []  # type: list # will store binary predictions
    all_predictions_probab = []  # type: list # will store probabilistic predictions
    all_labels = []  # type: list # will store the ground truths

    trained_model.eval()  # set model in inference mode
    with torch.no_grad():  # set all requires_grad flags to False
        for batch_data in test_data_loader:  # loop over batches

            filenames_list = batch_data["volume_meta_dict"]["filename_or_obj"]  # extract filename
            all_filenames.append(filenames_list)  # append to external list

            # extract volumes and labels from batch
            inputs, labels = (batch_data["volume"].to(device), batch_data["label"].to(device))

            # compute forward pass
            outputs = trained_model(inputs)  # type: torch.Tensor

            # convert from torch tensor to numpy array
            outputs_np_binary = outputs.argmax(dim=1).cpu().numpy()  # type: np.ndarray # find argmax along rows (i.e. for each sample)
            outputs_np_probab = outputs.cpu().detach().numpy()[:, 1]  # type: np.ndarray  # extract probabilistic output for the positive class
            labels_np = labels.cpu().numpy()  # type: np.ndarray

            # append predictions and labels to external lists
            all_predictions_binary.append(outputs_np_binary)
            all_predictions_probab.append(outputs_np_probab)
            all_labels.append(labels_np)

    # flatten list of lists
    all_filenames_flat = flatten_list(all_filenames)
    all_predictions_binary_flat = flatten_list(all_predictions_binary)
    all_predictions_probab_flat = flatten_list(all_predictions_probab)
    all_labels_flat = flatten_list(all_labels)

    return all_filenames_flat, all_predictions_binary_flat, all_predictions_probab_flat, all_labels_flat


def extract_reports_with_known_t2_label_agreed_between_annotators(df_comparative_dates_path: str,
                                                                  binary_classification: bool,
                                                                  nb_annotators: int,
                                                                  annotation_type: str) -> pd.DataFrame:
    """This function returns only the report pairs for which the T2 labels is known and for which the two annotators agreed.
    Args:
        df_comparative_dates_path: path to dataframe containing current and comparative dates and labels
        binary_classification: whether we will perform binary cross validation or not
        nb_annotators: number of annotators
        annotation_type: used to distinguish between the two annotation schemes (manual and automatic)
    Returns:
        df_rows_with_known_label: output dataframe containing only the report pairs where the T2 labels is known and the annotators agreed
    """
    df_comparative_dates = pd.read_csv(df_comparative_dates_path)  # type: pd.DataFrame

    df_comparative_dates = df_comparative_dates.dropna()  # type: pd.DataFrame # remove rows with NaN, NaT, etc.

    if annotation_type == "manual":
        t2_label_name = "t2_label_A1"
    elif annotation_type == "automatic":
        t2_label_name = "t2_label_random_forest"
    else:
        raise ValueError("annotation_type can only be manual or automatic; got {} instead.".format(annotation_type))

    if nb_annotators == 2 and annotation_type == "manual":  # with 2 annotators, the dataframe contains the labels for the two of them
        cols_of_interest = ["ipp", "exam_date", "comparative_date", "t2_label_A1", "t2_label_A2"]  # type: list
    elif nb_annotators == 3 and annotation_type == "manual":  # with 3 annotators, I merged all of them beforehand
        cols_of_interest = ["ipp", "exam_date", "comparative_date", t2_label_name]  # type: list
    elif annotation_type == "automatic" and nb_annotators == 0:
        cols_of_interest = ["ipp", "exam_date", "comparative_date", t2_label_name]  # type: list
    else:
        raise ValueError("nb_annotators can only be 0, 2 or 3; got {} instead".format(nb_annotators))

    # extract sub-dataframe only with the columns of interest
    sub_df_comparative_dates = df_comparative_dates.loc[:, cols_of_interest]  # type: pd.DataFrame

    # cast comparative_date to int
    sub_df_comparative_dates = sub_df_comparative_dates.astype({"comparative_date": int, t2_label_name: int})
    if nb_annotators == 2:
        sub_df_comparative_dates = sub_df_comparative_dates.astype({"t2_label_A2": int})

    if binary_classification:
        # convert all non-stable labels to "unstable" (i.e. merge progression and response into one unique class) to make the task binary
        sub_df_comparative_dates[t2_label_name].replace({2: 1}, inplace=True)
        if nb_annotators == 2:
            sub_df_comparative_dates['t2_label_A2'].replace({2: 1}, inplace=True)

    # only pick cases where the two annotators agree
    if nb_annotators == 2:
        sub_df_comparative_dates = sub_df_comparative_dates.loc[sub_df_comparative_dates[t2_label_name] == sub_df_comparative_dates["t2_label_A2"]]

    # remove rows with unknown/not mentioned label
    df_rows_with_known_label = sub_df_comparative_dates.query("{} != 3".format(t2_label_name))

    # restart dataframe's indexes from 0 and make them increase without holes (i.e. missing rows)
    df_rows_with_known_label = df_rows_with_known_label.reset_index(drop=True)

    return df_rows_with_known_label


def plot_roc_curve(flat_y_test: list,
                   flat_y_pred_proba: list,
                   cv_folds: int,
                   out_path: str,
                   legend_label: str = "doc2vec",
                   plot: bool = True,
                   save: bool = True) -> Tuple[np.ndarray, np.ndarray, float]:
    """This function computes FPR, TPR and AUC. Then, it plots the ROC curve
    Args:
        flat_y_test: labels
        flat_y_pred_proba: predictions
        cv_folds: number of folds in the cross-validation
        out_path: path where we save the figure
        legend_label: label to use in the legend
        plot: if True, the ROC curve will be displayed
        save: if True, save the figure to disk
    Returns:
        fpr: false positive rates
        tpr: true positive rates
        auc_roc: area under the ROC curve
    """
    fpr, tpr, _ = roc_curve(flat_y_test, flat_y_pred_proba, pos_label=1)
    tpr[0] = 0.0  # ensure that first element is 0
    tpr[-1] = 1.0  # ensure that last element is 1
    auc_roc = auc(fpr, tpr)
    if plot:
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, color="b", label=r'{} (AUC = {:.2f})'.format(legend_label, auc_roc), lw=2, alpha=.8)
        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
        ax.set_title("ROC curve; {}-fold CV".format(cv_folds), weight="bold", fontsize=15)
        ax.set_xlabel('FPR (1- specificity)', fontsize=12)
        ax.set_ylabel('TPR (sensitivity)', fontsize=12)
        ax.legend(loc="lower right")
        if save:
            fig.savefig(out_path)  # save the full figure

    return fpr, tpr, auc_roc


def first_argmax(input_list: list) -> int:
    """This function returns the index of the max value. If there are duplicate max values in input_list,
    the index of the first maximum value found will be returned.
    Args:
        input_list: list for which we want to find the argmax
    Returns:
        idx_max: index corresponding to the maximum value
    """
    idx_max = input_list.index(max(input_list))

    return idx_max


def first_argmin(input_list: list) -> int:
    """This function returns the index of the min value. If there are duplicate min values in input_list,
    the index of the first min value found will be returned.
    Args:
        input_list: list for which we want to find the argmax
    Returns:
        idx_min: index corresponding to the minimum value
    """
    idx_min = input_list.index(min(input_list))

    return idx_min


def set_parameter_requires_grad(model: monai.networks.nets,
                                nb_output_classes: int,
                                network: str,
                                fc_nodes: tuple,
                                feature_extracting: bool,
                                pretrain_path: str) -> list:
    """This function sets the .requires_grad attribute of some parameters in the model to False.
    The params with requires_grad == False are those that we freeze (i.e. that we do not retrain).
    Args:
        model: model used for training
        nb_output_classes: number of output classes
        network: network used during training
        fc_nodes: number of neurons in the fully-connected layers
        feature_extracting: if True, it means we use the model as feature extractor and only fine-tune the last layer(s). If False,
                            the model is fine-tuned and all model parameters are updated
        pretrain_path: path to weights of a pre-trained model; if empty, the model will train from scratch
    Returns:
        params_to_update: parameters that we want to update during training
    """
    params_to_update = model.parameters()  # by default, we update all parameters

    # if we use the pre-trained CNN as feature extractor (i.e. only fine-tune last layer(s)) AND pretrain_path is not empty (i.e. we are loading pre-trained weights)
    if feature_extracting and pretrain_path:
        print("\nTransfer learning with Feature Extraction...")
        # first set all grads to False
        for param in model.parameters():
            param.requires_grad = False  # set requires_grad to False (i.e. freeze layer)

        if network == "seresnext50":
            # then, re-initialize the last fully-connected layer(s) such that, by default, requires_grad are set to True when the layer is created
            nb_filters = model.last_linear.in_features  # extract number of filters of last FC layer
            model.last_linear = nn.Linear(nb_filters, nb_output_classes)
        elif network == "customVGG":
            # then, re-initialize the last fully-connected layer(s) such that, by default, requires_grad are set to True when the layer is created
            model.fc1 = torch.nn.LazyLinear(out_features=fc_nodes[0])  # we use lazy linear to infer the input shape
            model.fc2 = torch.nn.Linear(in_features=fc_nodes[0], out_features=fc_nodes[1])
            model.fc3 = torch.nn.Linear(in_features=fc_nodes[1], out_features=fc_nodes[2])
            model.fc4 = torch.nn.Linear(in_features=fc_nodes[2], out_features=nb_output_classes)
        else:
            raise ValueError("Unknown network; only 'densenet121', 'efficientnet', 'seresnext50', 'customVGG' and 'seresnet50'  are allowed; got {} instead".format(network))

        params_to_update = []  # initialize as empty
        for param in model.parameters():
            if param.requires_grad:  # if requires_grad == True
                params_to_update.append(param)

    assert params_to_update, "params_to_update should not be empty"

    return params_to_update


def extract_validation_data(ext_train_subs: list,
                            x_ext_train: list,
                            y_ext_train: list,
                            percentage_val_subs: float) -> (list, list, list, list, list, list):
    """This function extracts validation volumes and labels from the training set.
    Args:
        ext_train_subs: unique subjects of training set
        x_ext_train: paths to difference volumes of training set
        y_ext_train: labels of difference volumes of training set
        percentage_val_subs: percentage of subjects to use for validation
    """
    assert not list_has_duplicates(ext_train_subs), "There should not be duplicates in subject list"
    nb_validation_subs = int(round_half_up(percentage_val_subs * len(ext_train_subs)))

    random.seed(123)  # set fixed random seed such that val subjects are always the same
    val_subs = random.sample(ext_train_subs, nb_validation_subs)
    assert not list_has_duplicates(val_subs), "There should not be duplicates in val subject list. We want to sample without replacement."

    # keep remaining subs for training
    int_train_subs = [sub for sub in ext_train_subs if sub not in val_subs]
    assert sorted(int_train_subs + val_subs) == sorted(ext_train_subs), "int_train_subs combined with val_subs should be equal to all ext_train_subs"

    x_int_train = [path for path in x_ext_train if any(sub in path for sub in int_train_subs)]
    idxs_int_train_volumes_paths = [idx for idx, value in enumerate(x_ext_train) if value in x_int_train]
    y_int_train = [y_ext_train[idx] for idx in idxs_int_train_volumes_paths]

    x_val = [path for path in x_ext_train if any(sub in path for sub in val_subs)]
    idxs_val_volumes_paths = [idx for idx, value in enumerate(x_ext_train) if value in x_val]
    y_val = [y_ext_train[idx] for idx in idxs_val_volumes_paths]

    return int_train_subs, val_subs, x_int_train, x_val, y_int_train, y_val


def add_automatically_annotated_data(automatically_annotated_data_path: str,
                                     x_train: list,
                                     y_train: list,
                                     df_rows_automatic_labels_path: str,
                                     ext_cv_folds: int,
                                     external_cv_fold_counter: int,
                                     split_across_folds: bool,
                                     val_subs: list,
                                     test_subs: list,
                                     shuffle=True):
    """This function combines the manually-annotated data with the added automatically-annotated data.
    Args:
        automatically_annotated_data_path: path to directory containing automatically-annotated difference maps
        x_train: paths to manually-annotated difference maps
        y_train: labels of manually-annotated difference maps
        df_rows_automatic_labels_path: path to dataframe containing info about the automatically-annotated data
        ext_cv_folds: number of cross-validation folds
        external_cv_fold_counter: cross-validation fold being run
        split_across_folds: if True, added data is split across cross-val splits; if False, all added data is merged with the training fold at each cross-val split
        val_subs: validation subjects; used to exclude these from the added subs
        test_subs: test subjects; used to exclude these from the added subs
        shuffle: if True, the mixed data (manual+automatic) is shuffled
    Returns:
        enlarged_x: paths to mixed (manual+automatic) difference maps
        enlarged_y: labels corresponding to the mixed (manual+automatic) difference maps
    """
    random.seed(123)  # set random seed for reproducibility
    df_rows_automatic_labels = pd.read_csv(df_rows_automatic_labels_path)  # type: pd.DataFrame

    all_subs, volume_differences_t2, classification_labels, _ = extract_volumes_and_labels_add_automatic_sub_ses(automatically_annotated_data_path,
                                                                                                                 df_rows_automatic_labels,
                                                                                                                 val_subs,
                                                                                                                 test_subs,
                                                                                                                 annotation_type="automatic")

    # if we want to split the automatically-annotated data across the training folds of the cross-validation
    if split_across_folds:
        # split added subjects in equal-sized groups. At every training fold we add a portion of the new subjects
        split_subs = split_list_equal_sized_groups(all_subs, n=ext_cv_folds)  # type: list

        one_split_of_subs = split_subs[external_cv_fold_counter-1]  # extract one split of subjects

    # if instead we want to merge all the added data to the training fold of each cross-validation split
    else:
        one_split_of_subs = all_subs

    # extract difference volumes and labels corresponding to the added subjects
    x_added = [volume for volume in volume_differences_t2 if any(sub in volume for sub in one_split_of_subs)]
    idxs_added_volumes = [idx for idx, value in enumerate(volume_differences_t2) if value in x_added]
    y_added = [classification_labels[idx] for idx in idxs_added_volumes]

    # merge manually-annotated data with the (added) automatically-annotated data
    enlarged_x = x_train + x_added
    enlarged_y = y_train + y_added

    if shuffle:
        enlarged_x, enlarged_y = shuffle_two_lists_with_same_order(enlarged_x, enlarged_y)

    return enlarged_x, enlarged_y


class CustomVGG(torch.nn.Module):  # inherit from torch.nn.Module
    def __init__(self,
                 conv_filters: tuple,
                 fc_nodes: tuple,
                 nb_output_classes: int) -> None:
        """In the constructor we instantiate the layers"""
        super().__init__()  # add __init__ of superclass (i.e. parent class)
        self.conv0_1 = torch.nn.Conv3d(in_channels=1, out_channels=conv_filters[0], kernel_size=(3, 3, 3), stride=(1, 1, 1), padding="same")
        self.conv1 = torch.nn.Conv3d(in_channels=conv_filters[0], out_channels=conv_filters[0], kernel_size=(3, 3, 3), stride=(1, 1, 1), padding="same")
        self.conv1_2 = torch.nn.Conv3d(in_channels=conv_filters[0], out_channels=conv_filters[1], kernel_size=(3, 3, 3), stride=(1, 1, 1), padding="same")
        self.conv2 = torch.nn.Conv3d(in_channels=conv_filters[1], out_channels=conv_filters[1], kernel_size=(3, 3, 3), stride=(1, 1, 1), padding="same")
        self.conv2_3 = torch.nn.Conv3d(in_channels=conv_filters[1], out_channels=conv_filters[2], kernel_size=(3, 3, 3), stride=(1, 1, 1), padding="same")
        self.conv3 = torch.nn.Conv3d(in_channels=conv_filters[2], out_channels=conv_filters[2], kernel_size=(3, 3, 3), stride=(1, 1, 1), padding="same")
        self.conv3_4 = torch.nn.Conv3d(in_channels=conv_filters[2], out_channels=conv_filters[3], kernel_size=(3, 3, 3), stride=(1, 1, 1), padding="same")
        self.conv4 = torch.nn.Conv3d(in_channels=conv_filters[3], out_channels=conv_filters[3], kernel_size=(3, 3, 3), stride=(1, 1, 1), padding="same")

        self.maxpool3d = torch.nn.MaxPool3d(kernel_size=(2, 2, 2), stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        self.batchnorm1 = torch.nn.BatchNorm3d(conv_filters[0], eps=1e-05, momentum=0.1)
        self.batchnorm2 = torch.nn.BatchNorm3d(conv_filters[1], eps=1e-05, momentum=0.1)
        self.batchnorm3 = torch.nn.BatchNorm3d(conv_filters[2], eps=1e-05, momentum=0.1)
        self.batchnorm4 = torch.nn.BatchNorm3d(conv_filters[3], eps=1e-05, momentum=0.1)
        self.flatten = torch.nn.Flatten()
        self.relu = torch.nn.ReLU()

        self.fc1 = torch.nn.LazyLinear(out_features=fc_nodes[0])  # we use lazy linear to infer the input shape
        self.fc2 = torch.nn.Linear(in_features=fc_nodes[0], out_features=fc_nodes[1])
        self.fc3 = torch.nn.Linear(in_features=fc_nodes[1], out_features=fc_nodes[2])
        self.fc4 = torch.nn.Linear(in_features=fc_nodes[2], out_features=nb_output_classes)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        """In the forward method we accept a Tensor of input data and we return a Tensor of output data."""
        x = self.conv0_1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.batchnorm1(x)
        x = self.maxpool3d(x)

        x = self.conv1_2(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.batchnorm2(x)
        x = self.maxpool3d(x)

        x = self.conv2_3(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.batchnorm3(x)
        x = self.maxpool3d(x)

        x = self.conv3_4(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.batchnorm4(x)
        x = self.maxpool3d(x)

        x = torch.flatten(x, 1)  # flatten

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x_out = self.sigmoid(x)

        return x_out


def select_model(network: str,
                 device: torch.device,
                 nb_output_classes: int,
                 conv_filters: tuple,
                 fc_nodes: tuple):
    """This function chooses a model from some of the monai networks.
    Args:
        network: network chosen by the user
        device: device used (either CPU or GPU)
        nb_output_classes: number of output classes
        conv_filters: number of filters in conv layers
        fc_nodes: number of neurons in the fully-connected layers
    """
    if network == "densenet121":
        model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=nb_output_classes).to(device)  # define network
        model_for_inference = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=nb_output_classes).to(device)  # define network
    elif network == "efficientnet":
        model = monai.networks.nets.EfficientNetBN(model_name='efficientnet-b0', spatial_dims=3, in_channels=1, num_classes=nb_output_classes).to(device)  # define network
        model_for_inference = monai.networks.nets.EfficientNetBN(model_name='efficientnet-b0', spatial_dims=3, in_channels=1, num_classes=nb_output_classes).to(device)  # define network
    elif network == "seresnext50":
        model = monai.networks.nets.SEResNext50(spatial_dims=3, in_channels=1, num_classes=nb_output_classes, layers=(2, 3, 4, 2)).to(device)  # define network
        model_for_inference = monai.networks.nets.SEResNext50(spatial_dims=3, in_channels=1, num_classes=nb_output_classes, layers=(2, 3, 4, 2)).to(device)  # define network
    elif network == "seresnet50":
        model = monai.networks.nets.SEResNet50(spatial_dims=3, in_channels=1, num_classes=nb_output_classes, layers=(2, 3, 4, 2)).to(device)  # define network
        model_for_inference = monai.networks.nets.SEResNet50(spatial_dims=3, in_channels=1, num_classes=nb_output_classes, layers=(2, 3, 4, 2)).to(device)  # define network
    elif network == "customVGG":
        model = CustomVGG(conv_filters, fc_nodes, nb_output_classes).to(device)  # define network
        model_for_inference = CustomVGG(conv_filters, fc_nodes, nb_output_classes).to(device)  # define network
    else:
        raise ValueError("Unknown network; only 'densenet121', 'efficientnet', 'seresnext50', 'customVGG' and 'seresnet50'  are allowed; got {} instead".format(network))

    return model, model_for_inference


def create_dir_if_not_exist(dir_to_create: str) -> None:
    """This function creates the input dir if it doesn't exist.
    Args:
        dir_to_create: directory that we want to create
    """
    if not os.path.exists(dir_to_create):  # if dir doesn't exist
        os.makedirs(dir_to_create, exist_ok=True)  # create it


def all_elements_in_list_are_identical(input_list: list) -> bool:
    """This function checks whether all the elements in input_list are identical. If they are, True is returned; otherwise, False is returned
    Args:
        input_list: input list for which we check that all elements are identical
    Returns:
        all_elements_are_identical: bool that indicates whether all elements are identical or not
    """
    all_elements_are_identical = all(x == input_list[0] for x in input_list)

    return all_elements_are_identical


def remove_val_and_test_subs_used_in_baseline(all_subs: list,
                                              fold_to_do: int,
                                              path_to_output_baseline_dir: str) -> list:
    """This function removes the validation and test subjects that were used in the baseline from all_subs.
    This function is only used when we do pre-training + fine-tuning/feature-extracting
    Args:
        all_subs: list of all subjects that we'll later split into train and val
        fold_to_do: which is the fold that was done in the baseline experiment
        path_to_output_baseline_dir: path to folder containing the outputs of the baseline experiment
    Returns:
        reduced_subs: the input list but without the val and test subjects that were used in the baseline experiment
    """
    out_dir_baseline = os.path.join(path_to_output_baseline_dir, "fold{}".format(fold_to_do))
    val_subs = load_list_from_disk_with_pickle(os.path.join(out_dir_baseline, "val_subs_split{}.pkl".format(fold_to_do)))
    test_subs = load_list_from_disk_with_pickle(os.path.join(out_dir_baseline, "test_subs_split{}.pkl".format(fold_to_do)))

    reduced_subs = [sub for sub in all_subs if sub not in val_subs+test_subs]  # remove validation and test subjects of the baseline experiment

    return reduced_subs


def main():
    pass


if __name__ == '__main__':
    main()
