import os
import sys
sys.path.append('/home/to5743/glioma_project/Glioma_Change_Detection_T2w/')  # this line is needed on the HPC cluster to recognize the dir as a python package
import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from monai.data import Dataset
from monai.transforms import AddChanneld, Compose, Resized, EnsureTyped, LoadImaged,\
    Lambda, RandFlipd, RandGaussianNoised, RandZoomd, Rand3DElasticd
from datetime import datetime
import getpass
from shutil import copyfile
from dataset_creation.utils_dataset_creation import load_config_file
from utils_tdinoto.utils_lists import extract_unique_elements, save_list_to_disk_with_pickle
from utils_tdinoto.utils_strings import str2bool
import optuna
from training_and_inference.utils_volume_classification import extract_volumes_and_labels,\
    extract_train_and_test_volumes_and_labels, training_loop_with_validation, extract_reports_with_known_t2_label_agreed_between_annotators,\
    set_parameter_requires_grad, extract_validation_data, select_model, run_inference, create_dir_if_not_exist, add_automatically_annotated_data


__author__ = "Tommaso Di Noto"
__version__ = "0.0.1"
__email__ = "tommydino@hotmail.it"
__status__ = "Prototype"


def classification_t2_volumes_difference_after_optuna_transfer_learning(df_comparative_dates_path: str,
                                                                        input_dir: str,
                                                                        batch_size: int,
                                                                        nb_workers: int,
                                                                        ext_cv_folds: int,
                                                                        nb_epochs: int,
                                                                        out_dir: str,
                                                                        automatically_annotated_data_path: str,
                                                                        split_added_data_across_folds: bool,
                                                                        val_interval: int,
                                                                        network: str,
                                                                        binary_classification: bool,
                                                                        annotation_type: str,
                                                                        legend_label: str,
                                                                        class_weights: list,
                                                                        nb_annotators: int,
                                                                        percentage_val_subs: float,
                                                                        fold_to_do: int,
                                                                        optuna_output_dir: str,
                                                                        study_name: str,
                                                                        patience: int,
                                                                        pretrain_path_above_0_75: str,
                                                                        pretrain_path_above_0_95: str) -> None:
    """This function extracts the image pairs where annotators agree for the T2 conclusion and invokes the actual classification function
    Args:
        df_comparative_dates_path: path to dataframe which contain the link between current and comparative exams
        input_dir: path where the HAD volume differences are saved
        batch_size: batch size
        nb_workers: number of workers to use in parallel operations
        ext_cv_folds: number of external cross-validation folds
        nb_epochs: number of training epochs
        out_dir: output folder
        automatically_annotated_data_path: path where the AAD volume differences are saved
        split_added_data_across_folds: whether to split AAD data across folds or not
        val_interval: epoch frequency with which we print/save validation metrics
        network: model to use for training and inference
        binary_classification: whether to perform binary classification or not
        annotation_type: used to distinguish between the two annotation schemes (manual and automatic)
        legend_label: label to use in the legend of figures
        class_weights: weights to use during training; used to penalize more for minority class
        nb_annotators: number of human annotators
        percentage_val_subs: percentage of subjects used for validation
        fold_to_do: cross-validation fold to do (used so that we can run multiple folds in parallel in the HPC cluster)
        optuna_output_dir: directory where we save (and load/resume) the optuna study
        study_name: optuna study name
        patience: number of epochs to wait for early-stopping if there is no improvement across epochs
        pretrain_path_above_0_75: path to weights of a pre-trained model
        pretrain_path_above_0_95: path to weights of a pre-trained model
    """
    date = datetime.today().strftime('%b_%d_%Y')  # type: str # save today's date
    out_dir = os.path.join(out_dir, "{}_{}_{}".format(legend_label,
                                                      network,
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

    all_subs, volume_differences_t2, classification_labels, median_shape_orig = extract_volumes_and_labels(input_dir,
                                                                                                           df_rows_with_known_t2_label,
                                                                                                           annotation_type)
    print("\nMedian shape: {}".format(median_shape_orig))

    nb_output_classes = len(extract_unique_elements(classification_labels))  # type: int # find number of unique classes

    # define monai data augmentation
    augmentation_transforms = Compose([RandFlipd(keys="volume", prob=0.2, spatial_axis=None),
                                       RandGaussianNoised(keys="volume", prob=0.2, mean=0.0, std=0.1),
                                       RandZoomd(keys="volume", prob=0.2, min_zoom=0.7, max_zoom=1.3),
                                       Rand3DElasticd(keys="volume", prob=0.2, sigma_range=(5, 7), magnitude_range=(50, 150), padding_mode="zeros")])

    # choose network, loss function and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # if available, set GPU to use during training loop; otherwise, use cpu
    conv_filters_vgg, fc_nodes_vgg = (4, 8, 16, 32), (256, 128, 32)

    # define loss function with weights to use during training; the higher the weight for one class the higher the loss;
    # we "force" the net to learn more on the minority class
    class_weights = torch.Tensor(class_weights).to(device)  # type: torch.Tensor # cast from list to tensor and set device
    loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)  # define loss function

    # --------------------------------------- BEGIN EXTERNAL CROSS-VALIDATION ---------------------------------------
    kf_ext = KFold(n_splits=ext_cv_folds, shuffle=True, random_state=123)  # create cross-validator object with fixed random seed for reproducibility
    external_cv_fold_counter = 0  # type: int # counter to keep track of cross-validation fold
    for ext_train_subs_idxs, test_subs_idxs in kf_ext.split(all_subs):
        external_cv_fold_counter += 1  # increment counter

        if external_cv_fold_counter == fold_to_do:

            assert os.path.exists(optuna_output_dir), "Optuna directory should exist cause we are performing inference"

            # create Relational DataBase (RDB) where we save the trials; as soon as the script finishes, a .db file is dumped in the specified directory (optuna_output_dir in this case)
            storage_name = "sqlite:///{}.db".format(os.path.join(optuna_output_dir, study_name))
            # since we will monitor the validation F1 score, we set the direction as "maximize"; load_if_exists=True is used to resume the study in case we had already done some trials
            study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True, direction="maximize")
            trials_performed_so_far = study.get_trials()  # type: list
            assert trials_performed_so_far, "list trials_performed_so_far should not be empty cause we are performing inference"

            print("\nTrials performed so far: {}".format(len(trials_performed_so_far)))
            print("Best trial:")
            best_trial = study.best_trial
            print("  Value (validation f1-score): ", best_trial.value)
            print("  Params: ")
            for key, value in best_trial.params.items():
                print("    {}: {}".format(key, value))

            # extract and apply best hyperparameters
            best_lr = best_trial.params["lr"]
            best_wd = best_trial.params["wd"]
            best_feature_extracting = best_trial.params["feature_extracting"]
            best_pretrain_path = best_trial.params["pretrain_path"]
            best_df_aad_path = best_trial.params["df_aad_path"]

            # if we want to load the weights of a pre-trained model (i.e.if pretrain_path is not empty)
            if best_pretrain_path:
                print("\nLoad weights from pre-trained network...")
                model, model_for_inference = select_model(network, device, nb_output_classes, conv_filters_vgg, fc_nodes_vgg)
                model.load_state_dict(torch.load(best_pretrain_path))  # load parameters corresponding to a pretrain model

            # if instead we want to train from scratch (i.e. without loading any weights from pre-trained models)
            else:
                print("\nTrain from scratch...")
                # re-instantiate the model for every trial such that we always re-start with random weights
                model, model_for_inference = select_model(network, device, nb_output_classes, conv_filters_vgg, fc_nodes_vgg)

            # choose median_shape (it can slightly vary depending on which AAD we use)
            if not best_pretrain_path:  # if pretrain_path is empty
                median_shape = median_shape_orig
            elif best_pretrain_path == pretrain_path_above_0_75:
                median_shape = (310, 387, 43)
            elif best_pretrain_path == pretrain_path_above_0_95:
                median_shape = (310, 387, 46)
            else:
                raise ValueError("Unknown pretrain_path")

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

            # set parameters to update during training (all params if we perform fine-tuning; only the last linear layer(s) if we perform feature_extracting)
            params_to_update = set_parameter_requires_grad(model,
                                                           nb_output_classes,
                                                           network,
                                                           fc_nodes_vgg,
                                                           feature_extracting=best_feature_extracting,
                                                           pretrain_path=best_pretrain_path)
            model.to(device)  # ensure model is loaded in the GPU (if running on the cluster)
            optimizer = torch.optim.Adam(params_to_update, lr=best_lr, weight_decay=best_wd)

            # ensure that multiple sessions of same subject are not split some in training and some in val/test
            ext_train_subs, test_subs, x_ext_train, y_ext_train, x_test, y_test = extract_train_and_test_volumes_and_labels(all_subs,
                                                                                                                            ext_train_subs_idxs,
                                                                                                                            test_subs_idxs,
                                                                                                                            volume_differences_t2,
                                                                                                                            classification_labels)

            # save list of test subjects to disk; we will use it later to avoid training on these during pre-training
            save_list_to_disk_with_pickle(test_subs, os.path.join(out_dir, "fold{}".format(external_cv_fold_counter)), "test_subs_split{}.pkl".format(external_cv_fold_counter))

            # select portion of training subs (and corresponding sessions) for validation
            int_train_subs, val_subs, x_int_train, x_val, y_int_train, y_val = extract_validation_data(ext_train_subs, x_ext_train, y_ext_train, percentage_val_subs)
            print("\nNb. ext_train_subs: {}; nb. diffmaps: {}".format(len(ext_train_subs), len(x_ext_train)))
            print("Nb. int_train_subs: {}; nb. diffmaps: {}".format(len(int_train_subs), len(x_int_train)))
            print("Nb. val_subs: {}; nb. diffmaps: {}".format(len(val_subs), len(x_val)))
            print("Nb. test_subs: {}; nb. diffmaps: {}".format(len(test_subs), len(x_test)))
            print("\nFirst ten val_subs: {} ...".format(sorted(val_subs)[:10]))
            print("\nFirst ten test_subs: {} ...\n".format(sorted(test_subs)[:10]))

            # save list of validation subjects to disk; we will use it later to avoid training on these during pre-training
            save_list_to_disk_with_pickle(val_subs, os.path.join(out_dir, "fold{}".format(external_cv_fold_counter)), "val_subs_split{}.pkl".format(external_cv_fold_counter))

            # if we want to add the automatically-annotated data (i.e. if automatically_annotated_data_path is not empty) AND pretrain_path is empty
            # either we mix HAD and AAD and train from scratch OR we do pretrain + fine-tuning/feature-extracting
            if automatically_annotated_data_path and not best_pretrain_path:

                print("\nLength x_int_train before adding AAD: {}".format(len(x_int_train)))
                x_int_train, y_int_train = add_automatically_annotated_data(automatically_annotated_data_path,
                                                                            x_int_train,
                                                                            y_int_train,
                                                                            best_df_aad_path,
                                                                            ext_cv_folds,
                                                                            external_cv_fold_counter,
                                                                            split_added_data_across_folds,
                                                                            val_subs,
                                                                            test_subs,
                                                                            shuffle=True)

                print("\nLength x_int_train after adding AAD: {}\n".format(len(x_int_train)))

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
            best_model_path = training_loop_with_validation(nb_epochs, model, int_train_loader, device, optimizer, loss_function,
                                                            val_loader, out_dir, date, external_cv_fold_counter, ext_cv_folds,
                                                            val_interval, patience, legend_label)

            # create test data loader
            test_files = [{"volume": volume_name, 'label': label_name} for volume_name, label_name in zip(x_test, y_test)]
            test_ds = Dataset(data=test_files, transform=val_transforms)
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=nb_workers, pin_memory=torch.cuda.is_available())

            # -------- run inference on test subjects
            model_for_inference.load_state_dict(torch.load(best_model_path, map_location=device))  # load best params according to validation results
            filenames, y_pred_binary_test, y_pred_probab_test, y_true_test = run_inference(model_for_inference, test_loader, device)

            # -------- save results to disk
            save_list_to_disk_with_pickle(filenames, os.path.join(out_dir, "fold{}".format(external_cv_fold_counter)), "filenames_fold{}.pkl".format(external_cv_fold_counter))
            save_list_to_disk_with_pickle(y_pred_binary_test, os.path.join(out_dir, "fold{}".format(external_cv_fold_counter)), "y_pred_binary_fold{}.pkl".format(external_cv_fold_counter))
            save_list_to_disk_with_pickle(y_pred_probab_test, os.path.join(out_dir, "fold{}".format(external_cv_fold_counter)), "y_pred_probabilistic_fold{}.pkl".format(external_cv_fold_counter))
            save_list_to_disk_with_pickle(y_true_test, os.path.join(out_dir, "fold{}".format(external_cv_fold_counter)), "y_true_fold{}.pkl".format(external_cv_fold_counter))
            print("\nSaved output lists at {}".format(os.path.join(out_dir, "fold{}".format(external_cv_fold_counter))))


def main():
    # the code inside here is run only when THIS script is run, and not just imported
    config_dict = load_config_file()  # load input config file with argparser

    # extract training args
    fold_to_do = config_dict['fold_to_do']
    network = config_dict['network']
    assert network in ("customVGG", "seresnext50"), "Only 'customVGG' and 'seresnext50' are allowed: found '{}' instead".format(network)
    batch_size = config_dict['batch_size']  # batch size used during training
    ext_cv_folds = config_dict['ext_cv_folds']  # number of external cross validation folds
    nb_epochs = config_dict['nb_epochs']  # number of training epochs
    val_interval = config_dict['val_interval']  # epoch frequency with which we print/save validation metrics
    binary_classification = str2bool(config_dict['binary_classification'])  # type: bool # whether to perform binary classification or not
    annotation_type = config_dict['annotation_type']  # type: str # used to distinguish between the two annotation schemes (manual and automatic)
    legend_label = config_dict['legend_label']  # type: str # label to use in the legend of figures
    class_weights = config_dict['class_weights']  # type: list # to use during training to weight more one class wrt another
    nb_annotators = config_dict['nb_annotators']  # type: int # number of human annotators
    percentage_val_subs = config_dict['percentage_val_subs']  # type: float

    df_comparative_dates_path = config_dict['df_comparative_dates_path']  # path to dataframe containing comparative dates and reports
    input_dir = config_dict['input_dir']  # path to folder containing HAD volume differences
    out_dir = config_dict['out_dir']  # path where we save model parameters
    automatically_annotated_data_path = config_dict['automatically_annotated_data_path']  # path to folder containing AAD volume differences
    split_added_data_across_folds = str2bool(config_dict['split_added_data_across_folds'])
    optuna_output_dir = config_dict['optuna_output_dir']  # directory where we save (and load/resume) the optuna study
    study_name = config_dict['study_name'].replace("foldX", "fold{}".format(fold_to_do))
    patience = config_dict['patience']
    pretrain_path_above_0_75 = config_dict['pretrain_path_above_0_75'].replace("foldX", "fold{}".format(fold_to_do))
    pretrain_path_above_0_95 = config_dict['pretrain_path_above_0_95'].replace("foldX", "fold{}".format(fold_to_do))

    # set number of jobs to run in parallel
    on_hpc_cluster = getpass.getuser() in ['to5743']  # type: bool # check if user is in list of authorized users
    if on_hpc_cluster:  # if we are running in the HPC
        assert torch.cuda.is_available(), "We expect to have a GPU when running on the HPC"
        nb_workers = 4  # type: int # number of jobs to run in parallel
    else:  # if instead we run the script locally
        nb_workers = 1  # type: int # number of jobs to run in parallel

    classification_t2_volumes_difference_after_optuna_transfer_learning(df_comparative_dates_path,
                                                                        input_dir,
                                                                        batch_size,
                                                                        nb_workers,
                                                                        ext_cv_folds,
                                                                        nb_epochs,
                                                                        out_dir,
                                                                        automatically_annotated_data_path,
                                                                        split_added_data_across_folds,
                                                                        val_interval,
                                                                        network,
                                                                        binary_classification,
                                                                        annotation_type,
                                                                        legend_label,
                                                                        class_weights,
                                                                        nb_annotators,
                                                                        percentage_val_subs,
                                                                        fold_to_do,
                                                                        optuna_output_dir,
                                                                        study_name,
                                                                        patience,
                                                                        pretrain_path_above_0_75,
                                                                        pretrain_path_above_0_95)


if __name__ == '__main__':
    main()
