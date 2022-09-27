import os
import sys
sys.path.append('/home/to5743/glioma_project/Glioma_Change_Detection_T2w/')  # this line is needed on the HPC cluster to recognize the dir as a python package
import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from monai.data import Dataset
import numpy as np
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
    extract_train_and_test_volumes_and_labels, training_loop_with_validation_optuna, extract_reports_with_known_t2_label_agreed_between_annotators,\
    set_parameter_requires_grad, extract_validation_data, add_automatically_annotated_data, select_model, create_dir_if_not_exist


__author__ = "Tommaso Di Noto"
__version__ = "0.0.1"
__email__ = "tommydino@hotmail.it"
__status__ = "Prototype"


class ObjectiveOptunaTransferLearning(object):
    def __init__(self,
                 pretrain_path_wad_above_0_75,
                 pretrain_path_wad_above_0_95,
                 nb_output_classes,
                 network,
                 fc_nodes_vgg,
                 conv_filters_vgg,
                 device,
                 external_cv_fold_counter,
                 ext_cv_folds,
                 all_subs,
                 ext_train_subs_idxs,
                 test_subs_idxs,
                 volume_differences_t2,
                 classification_labels,
                 out_dir,
                 percentage_val_subs,
                 data_path_wad,
                 df_rows_automatic_labels_path_above_0_75,
                 df_rows_automatic_labels_path_above_0_95,
                 split_added_data_across_folds,
                 augmentation_transforms,
                 nb_workers,
                 nb_epochs,
                 loss_function,
                 date,
                 val_interval,
                 batch_size,
                 patience,
                 legend_label,
                 median_shape_orig):
        self.pretrain_path_wad_above_0_75 = pretrain_path_wad_above_0_75
        self.pretrain_path_wad_above_0_95 = pretrain_path_wad_above_0_95
        self.nb_output_classes = nb_output_classes
        self.network = network
        self.fc_nodes_vgg = fc_nodes_vgg
        self.conv_filters_vgg = conv_filters_vgg
        self.device = device
        self.external_cv_fold_counter = external_cv_fold_counter
        self.ext_cv_folds = ext_cv_folds
        self.all_subs = all_subs
        self.ext_train_subs_idxs = ext_train_subs_idxs
        self.test_subs_idxs = test_subs_idxs
        self.volume_differences_t2 = volume_differences_t2
        self.classification_labels = classification_labels
        self.out_dir = out_dir
        self.percentage_val_subs = percentage_val_subs
        self.data_path_wad = data_path_wad
        self.df_rows_automatic_labels_path_above_0_75 = df_rows_automatic_labels_path_above_0_75
        self.df_rows_automatic_labels_path_above_0_95 = df_rows_automatic_labels_path_above_0_95
        self.split_added_data_across_folds = split_added_data_across_folds
        self.augmentation_transforms = augmentation_transforms
        self.nb_workers = nb_workers
        self.nb_epochs = nb_epochs
        self.loss_function = loss_function
        self.date = date
        self.val_interval = val_interval
        self.batch_size = batch_size
        self.patience = patience
        self.legend_label = legend_label
        self.median_shape_orig = median_shape_orig

    def __call__(self, trial):
        """Here we define the arguments that WILL BE TUNED with optuna"""

        # if pretrain_path is empty, we train from scratch; otherwise we do some transfer learning (either fine-tuning or feature extracting)
        pretrain_path_optuna = trial.suggest_categorical('pretrain_path', ["", self.pretrain_path_wad_above_0_75, self.pretrain_path_wad_above_0_95])
        # if feature_extracting is True, we do feature extracting (i.e. only fine-tune last linear layer(s)); otherwise we do fine-tuning (i.e. fine-tune all layers)
        feature_extracting_optuna = trial.suggest_categorical('feature_extracting', [True, False])
        df_automatic_labels_optuna = trial.suggest_categorical('df_aad_path', [self.df_rows_automatic_labels_path_above_0_75, self.df_rows_automatic_labels_path_above_0_95])
        learning_rate_optuna = trial.suggest_categorical('lr', [1e-4, 1e-5, 1e-6])
        weight_decay_optuna = trial.suggest_categorical('wd', [0., 0.01])

        dict_params = {'pretrain_path_optuna': pretrain_path_optuna,
                       'feature_extracting_optuna': feature_extracting_optuna,
                       'df_rows_automatic_labels_optuna': df_automatic_labels_optuna,
                       'learning_rate_optuna': learning_rate_optuna,
                       'weight_decay_optuna': weight_decay_optuna}

        # -------------------------- chunk of code that contains hyperparameters to tune
        print("\n---------------------------------------------------- Hyperparams for trial nb. {}:\n{}".format(trial.number, dict_params))

        # if we want to load the weights of a pre-trained model (i.e. if pretrain_path is not empty)
        if dict_params['pretrain_path_optuna']:
            print("\nLoad weights from pre-trained network...")
            model, _ = select_model(self.network, self.device, self.nb_output_classes, self.conv_filters_vgg, self.fc_nodes_vgg)
            model.load_state_dict(torch.load(dict_params['pretrain_path_optuna']))  # load parameters corresponding to a pretrain model

        # if instead we want to train from scratch (i.e. without loading any weights from pre-trained models)
        else:
            print("\nTrain from scratch...")
            # re-instantiate the model for every trial such that we always re-start with random weights
            model, _ = select_model(self.network, self.device, self.nb_output_classes, self.conv_filters_vgg, self.fc_nodes_vgg)

        if not dict_params['pretrain_path_optuna']:  # if pretrain_path is empty
            median_shape = self.median_shape_orig
        elif dict_params['pretrain_path_optuna'] == self.pretrain_path_wad_above_0_75:
            median_shape = (310, 387, 43)
        elif dict_params['pretrain_path_optuna'] == self.pretrain_path_wad_above_0_95:
            median_shape = (310, 387, 46)
        else:
            raise ValueError("Unknown pretrain_path")

        # define preprocessing and transforms for training volumes
        train_transforms = Compose([LoadImaged(keys="volume"),
                                    AddChanneld(keys="volume"),  # add channel; monai expects channel-first tensors
                                    Resized(keys="volume", spatial_size=median_shape),  # resize all volumes to median volume shape
                                    Lambda(lambda x: self.augmentation_transforms(x)),  # apply data augmentation(s)
                                    EnsureTyped(keys="volume")])  # ensure that input data is either a PyTorch Tensor or np array

        # define preprocessing and transforms for validation/test volumes
        val_transforms = Compose([LoadImaged(keys="volume"),
                                  AddChanneld(keys="volume"),  # add channel; monai expects channel-first tensors
                                  Resized(keys="volume", spatial_size=median_shape),  # resize all volumes to median volume shape
                                  EnsureTyped(keys="volume")])  # ensure that input data is either a PyTorch Tensor or np array

        # set parameters to update during training (all params if we perform fine-tuning; only the last linear layer(s) if we perform feature_extracting)
        params_to_update = set_parameter_requires_grad(model,
                                                       self.nb_output_classes,
                                                       self.network,
                                                       self.fc_nodes_vgg,
                                                       feature_extracting=dict_params['feature_extracting_optuna'],
                                                       pretrain_path=dict_params['pretrain_path_optuna'])
        model.to(self.device)  # ensure model is loaded in the GPU (if running on the cluster)

        # define optimizer with learning rate and weight decay
        optimizer = torch.optim.Adam(params_to_update, lr=dict_params['learning_rate_optuna'], weight_decay=dict_params['weight_decay_optuna'])

        print("\n\n-------------------------- External CV fold {}/{}".format(self.external_cv_fold_counter, self.ext_cv_folds))
        # ensure that multiple sessions of same subject are not split some in training and some in val/test
        ext_train_subs, test_subs, x_ext_train, y_ext_train, x_test, y_test = extract_train_and_test_volumes_and_labels(self.all_subs,
                                                                                                                        self.ext_train_subs_idxs,
                                                                                                                        self.test_subs_idxs,
                                                                                                                        self.volume_differences_t2,
                                                                                                                        self.classification_labels)

        # save list of test subjects to disk; we will use it later to avoid training on these during pre-training
        save_list_to_disk_with_pickle(test_subs, os.path.join(self.out_dir, "fold{}".format(self.external_cv_fold_counter)), "test_subs_split{}.pkl".format(self.external_cv_fold_counter))

        # select portion of training subs (and corresponding sessions) for validation
        int_train_subs, val_subs, x_int_train, x_val, y_int_train, y_val = extract_validation_data(ext_train_subs, x_ext_train, y_ext_train, self.percentage_val_subs)
        print("\nNb. ext_train_subs: {}; nb. diffmaps: {}".format(len(ext_train_subs), len(x_ext_train)))
        print("Nb. int_train_subs: {}; nb. diffmaps: {}".format(len(int_train_subs), len(x_int_train)))
        print("Nb. val_subs: {}; nb. diffmaps: {}".format(len(val_subs), len(x_val)))
        print("Nb. test_subs: {}; nb. diffmaps: {}".format(len(test_subs), len(x_test)))
        print("\nval_subs: {}".format(sorted(val_subs)))
        print("\ntest_subs: {}\n".format(sorted(test_subs)))

        # save list of validation subjects to disk; we will use it later to avoid training on these during pre-training
        save_list_to_disk_with_pickle(val_subs, os.path.join(self.out_dir, "fold{}".format(self.external_cv_fold_counter)), "val_subs_split{}.pkl".format(self.external_cv_fold_counter))

        # check labels' distribution
        unique_y_ext_train, cnt_y_ext_train = np.unique(y_ext_train, return_counts=True)
        unique_y_int_train, cnt_y_int_train = np.unique(y_int_train, return_counts=True)
        unique_y_val, cnt_y_val = np.unique(y_val, return_counts=True)
        unique_y_test, cnt_y_test = np.unique(y_test, return_counts=True)
        print("External training set: unique labels -> {}, counts -> {}".format(unique_y_ext_train, cnt_y_ext_train))
        print("Internal training set: unique labels -> {}, counts -> {}".format(unique_y_int_train, cnt_y_int_train))
        print("Validation set: unique labels -> {}, counts -> {}".format(unique_y_val, cnt_y_val))
        print("Test set: unique labels -> {}, counts -> {}\n".format(unique_y_test, cnt_y_test))

        # if we want to add the automatically-annotated data (i.e. if automatically_annotated_data_path is not empty) AND pretrain_path is empty
        # either we mix HAD and AAD and train from scratch OR we do pretrain + fine-tuning/feature-extracting
        if self.data_path_wad and not dict_params['pretrain_path_optuna']:
            print("\nLength x_int_train before adding AAD: {}".format(len(x_int_train)))
            x_int_train, y_int_train = add_automatically_annotated_data(self.data_path_wad,
                                                                        x_int_train,
                                                                        y_int_train,
                                                                        dict_params['df_rows_automatic_labels_optuna'],
                                                                        self.ext_cv_folds,
                                                                        self.external_cv_fold_counter,
                                                                        self.split_added_data_across_folds,
                                                                        val_subs,
                                                                        test_subs,
                                                                        shuffle=True)

            print("\nLength x_int_train after adding AAD: {}\n".format(len(x_int_train)))

        # create list of dictionaries
        int_train_files = [{"volume": volume_name, 'label': label_name} for volume_name, label_name in zip(x_int_train, y_int_train)]
        val_files = [{"volume": volume_name, 'label': label_name} for volume_name, label_name in zip(x_val, y_val)]

        # create a training data loader
        int_train_ds = Dataset(data=int_train_files, transform=train_transforms)
        int_train_loader = DataLoader(int_train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.nb_workers, pin_memory=torch.cuda.is_available())

        # create a validation data loader
        val_ds = Dataset(data=val_files, transform=val_transforms)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.nb_workers, pin_memory=torch.cuda.is_available())

        # start typical PyTorch training
        _, best_validation_auc = training_loop_with_validation_optuna(self.nb_epochs, model, int_train_loader, self.device, optimizer, self.loss_function,
                                                                      val_loader, self.out_dir, self.date, self.external_cv_fold_counter, self.ext_cv_folds,
                                                                      self.val_interval, self.patience, self.legend_label)

        return best_validation_auc


def classification_t2_volumes_difference_optuna(df_dates_and_labels_had: str,
                                                data_path_had: str,
                                                batch_size: int,
                                                nb_workers: int,
                                                ext_cv_folds: int,
                                                nb_epochs: int,
                                                out_dir: str,
                                                val_interval: int,
                                                network: str,
                                                binary_classification: bool,
                                                pretrain_path_wad_above_0_75: str,
                                                pretrain_path_wad_above_0_95: str,
                                                annotation_type: str,
                                                legend_label: str,
                                                class_weights: list,
                                                nb_annotators: int,
                                                percentage_val_subs: float,
                                                data_path_wad: str,
                                                df_dates_and_labels_wad_above_0_75: str,
                                                df_dates_and_labels_wad_above_0_95: str,
                                                split_added_data_across_folds: bool,
                                                fold_to_do: int,
                                                nb_optuna_trials: int,
                                                optuna_output_dir: str,
                                                patience: int) -> None:
    """This function extracts the image pairs where both annotators agree for the T2 conclusion and invokes the actual classification function
    Args:
        df_dates_and_labels_had: path to dataframe which contain the link between current and comparative exams
        data_path_had: path where the volume differences are saved
        batch_size: batch size
        nb_workers: number of workers to use in parallel operations
        ext_cv_folds: number of external cross-validation folds
        nb_epochs: number of training epochs
        out_dir: output folder
        val_interval: epoch frequency with which we print/save validation metrics
        network: model to use for training and inference
        binary_classification: whether to perform binary classification or not
        pretrain_path_wad_above_0_75: path to weights of a pre-trained model; if empty, the model will train from scratch
        pretrain_path_wad_above_0_95: path to weights of a pre-trained model; if empty, the model will train from scratch
        annotation_type: used to distinguish between the two annotation schemes (manual and automatic)
        legend_label: label to use in the legend of figures and optuna output dir
        class_weights: weights to use during training; used to penalize more for minority class
        nb_annotators: number of human annotators
        percentage_val_subs: percentage of subjects used for validation
        data_path_wad: path to automatically-annotated data that we will mix with the training data (i.e. we enlarge the training dataset)
        df_dates_and_labels_wad_above_0_75: path to dataframe containing current sess, comp sess and labels of the automatically-annotated samples
        df_dates_and_labels_wad_above_0_95: path to dataframe containing current sess, comp sess and labels of the automatically-annotated samples
        split_added_data_across_folds: if True, added data is split across cross-val splits; if False, all added data is merged with the training fold at each cross-val split
        fold_to_do: cross-validation fold to do (used so that we can run multiple folds in parallel in the HPC cluster)
        nb_optuna_trials: number of hyperparams combinations to try with optuna
        optuna_output_dir: directory where we save (and load/resume) the optuna study
        patience: number of epochs to wait for early-stopping if there is no improvement across epochs
    Returns:
        None
    """
    date = datetime.today().strftime('%b_%d_%Y')  # type: str # save today's date
    optuna_output_dir = os.path.join(optuna_output_dir, "{}_{}_{}".format(legend_label, network, date))  # modify optuna output dir
    date_hours_minutes = datetime.today().strftime('%b_%d_%Y_%Hh')  # type: str # save today's date
    out_dir = os.path.join(out_dir, "{}_{}_{}".format(legend_label,
                                                      network,
                                                      date_hours_minutes))  # type: str # update name of output directory

    create_dir_if_not_exist(out_dir)  # if output dir does not exist, create it

    # copy config file to output dir so that we know every time which were the input parameters
    config_file_path = sys.argv[2]  # type: str
    config_out_path = os.path.join(out_dir, "config_file.json")  # type: str
    if not os.path.exists(config_out_path):
        copyfile(config_file_path, config_out_path)

    # set names of columns of interest that we want to extract
    df_rows_with_known_t2_label = extract_reports_with_known_t2_label_agreed_between_annotators(df_dates_and_labels_had,
                                                                                                binary_classification,
                                                                                                nb_annotators,
                                                                                                annotation_type)  # type: pd.DataFrame

    all_subs, volume_differences_t2, classification_labels, median_shape_orig = extract_volumes_and_labels(data_path_had,
                                                                                                           df_rows_with_known_t2_label,
                                                                                                           annotation_type)

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

            # ---------------------------------------- OPTUNA HYPERTUNING -----------------------------------
            # we include inside ObjectiveOptuna all the code that contains the parameters to tune
            objective = ObjectiveOptunaTransferLearning(pretrain_path_wad_above_0_75,
                                                        pretrain_path_wad_above_0_95,
                                                        nb_output_classes,
                                                        network,
                                                        fc_nodes_vgg,
                                                        conv_filters_vgg,
                                                        device,
                                                        external_cv_fold_counter,
                                                        ext_cv_folds,
                                                        all_subs,
                                                        ext_train_subs_idxs,
                                                        test_subs_idxs,
                                                        volume_differences_t2,
                                                        classification_labels,
                                                        out_dir,
                                                        percentage_val_subs,
                                                        data_path_wad,
                                                        df_dates_and_labels_wad_above_0_75,
                                                        df_dates_and_labels_wad_above_0_95,
                                                        split_added_data_across_folds,
                                                        augmentation_transforms,
                                                        nb_workers,
                                                        nb_epochs,
                                                        loss_function,
                                                        date,
                                                        val_interval,
                                                        batch_size,
                                                        patience,
                                                        legend_label,
                                                        median_shape_orig)

            create_dir_if_not_exist(optuna_output_dir)  # create dir if it doesn't exist

            study_name = "optuna_best_params_fold{}_{}".format(fold_to_do, legend_label)  # unique identifier of the study.
            # create Relational DataBase (RDB) where we save the trials; as soon as the script finishes, a .db file is dumped in the specified directory (optuna_output_dir in this case)
            storage_name = "sqlite:///{}.db".format(os.path.join(optuna_output_dir, study_name))
            # since we will monitor the validation F1 score, we set the direction as "maximize"; load_if_exists=True is used to resume the study in case we had already done some trials
            study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True, direction="maximize")
            trials_performed_so_far = study.get_trials()  # type: list
            print("\nTrials performed so far = {}".format(len(trials_performed_so_far)))
            for idx, trial in enumerate(trials_performed_so_far):
                print("\nTrial {}: params = {}; objective = {}".format(idx, trial.params, trial.value))
            # start optimization of study (i.e. invoke __call__ method)
            study.optimize(objective, n_trials=nb_optuna_trials)

            print("\nBest trial:")
            best_trial = study.best_trial
            print("  Value (validation f1-score): ", best_trial.value)
            print("  Params: ")
            for key, value in best_trial.params.items():
                print("    {}: {}".format(key, value))


def main():
    # the code inside here is run only when THIS script is run, and not just imported
    config_dict = load_config_file()  # load input config file with argparser

    # extract training args
    fold_to_do = config_dict['fold_to_do']
    network = config_dict['network']
    assert network in ("customVGG", "seresnext50"), "Only 'customVGG' and 'seresnext50' are allowed: found '{}' instead".format(network)
    date_of_pretrain = config_dict['date_of_pretrain']
    df_dates_and_labels_had = config_dict['df_dates_and_labels_had']  # path to dataframe containing comparative dates and reports
    data_path_had = config_dict['data_path_had']  # path to folder containing volume differences
    data_path_wad = config_dict['data_path_wad']  # type: str # path to automatically-annotated data that we will mix with the training set
    out_dir = config_dict['out_dir']  # path where we save model parameters
    optuna_output_dir = config_dict['optuna_output_dir']  # directory where we save (and later load/resume) the optuna study
    df_dates_and_labels_wad_above_0_75 = config_dict['df_dates_and_labels_wad_above_0_75']  # type: str # path to dataframe containing info about the automatically-annotated data
    df_dates_and_labels_wad_above_0_95 = config_dict['df_dates_and_labels_wad_above_0_95']  # type: str # path to dataframe containing info about the automatically-annotated data
    pretrain_output_dir = config_dict['pretrain_output_dir']

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
    nb_optuna_trials = config_dict['nb_optuna_trials']  # type: int # number of hyperparms combinations to try with optuna
    patience = config_dict['patience']  # type: int

    pretrain_path_wad_above_0_75 = os.path.join(pretrain_output_dir,
                                                'pretrain_aad_above_0_75_{}_lr_1e-05_{}'.format(network, date_of_pretrain),
                                                "fold{}".format(fold_to_do),
                                                "best_model.pth")
    pretrain_path_wad_above_0_95 = os.path.join(pretrain_output_dir,
                                                'pretrain_aad_above_0_95_{}_lr_1e-05_{}'.format(network, date_of_pretrain),
                                                "fold{}".format(fold_to_do),
                                                "best_model.pth")
    assert os.path.exists(pretrain_path_wad_above_0_75), "Path {} does not exist".format(pretrain_path_wad_above_0_75)
    assert os.path.exists(pretrain_path_wad_above_0_95), "Path {} does not exist".format(pretrain_path_wad_above_0_95)
    split_added_data_across_folds = str2bool(config_dict['split_added_data_across_folds'])  # type: bool

    # set number of jobs to run in parallel
    on_hpc_cluster = getpass.getuser() in ['to5743']  # type: bool # check if user is in list of authorized users
    if on_hpc_cluster:  # if we are running in the HPC
        assert torch.cuda.is_available(), "We expect to have a GPU when running on the HPC"
        nb_workers = 4  # type: int # number of jobs to run in parallel
    else:  # if instead we run the script locally
        nb_workers = 1  # type: int # number of jobs to run in parallel

    classification_t2_volumes_difference_optuna(df_dates_and_labels_had,
                                                data_path_had,
                                                batch_size,
                                                nb_workers,
                                                ext_cv_folds,
                                                nb_epochs,
                                                out_dir,
                                                val_interval,
                                                network,
                                                binary_classification,
                                                pretrain_path_wad_above_0_75,
                                                pretrain_path_wad_above_0_95,
                                                annotation_type,
                                                legend_label,
                                                class_weights,
                                                nb_annotators,
                                                percentage_val_subs,
                                                data_path_wad,
                                                df_dates_and_labels_wad_above_0_75,
                                                df_dates_and_labels_wad_above_0_95,
                                                split_added_data_across_folds,
                                                fold_to_do,
                                                nb_optuna_trials,
                                                optuna_output_dir,
                                                patience)


if __name__ == '__main__':
    main()
