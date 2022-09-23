import os
import optuna
from dataset_creation.utils_dataset_creation import load_config_file


__author__ = "Tommaso Di Noto"
__version__ = "0.0.1"
__email__ = "tommydino@hotmail.it"
__status__ = "Prototype"


def print_best_hyperparams(optuna_output_dir,
                           cv_folds,
                           study_name,
                           legend_model):

    assert os.path.exists(optuna_output_dir), "Optuna directory should exist"

    print("\n----------- Best hyperparams for {}".format(legend_model))
    for fold in range(cv_folds):
        study_name = study_name.replace("foldX", "fold{}".format(fold + 1))
        # create Relational DataBase (RDB) where we save the trials; as soon as the script finishes, a .db file is dumped in the specified directory (optuna_output_dir in this case)
        storage_name = "sqlite:///{}.db".format(os.path.join(optuna_output_dir, study_name))
        # since we will monitor the validation F1 score, we set the direction as "maximize"; load_if_exists=True is used to resume the study in case we had already done some trials
        study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True, direction="maximize")
        trials_performed_so_far = study.get_trials()  # type: list
        assert trials_performed_so_far, "list trials_performed_so_far should not be empty"

        best_trial = study.best_trial
        print("Fold {}: {} trials performed; val AUC = {:.3f}; best params: {}".format(fold + 1, len(trials_performed_so_far), best_trial.value, best_trial.params))

        # reset study name so that in every iteration of the loop it is modified
        study_name = study_name.replace("fold{}".format(fold + 1), "foldX")


def main():
    # the code inside here is run only when THIS script is run, and not just imported
    config_dict = load_config_file()  # load input config file

    optuna_output_dir = config_dict['optuna_output_dir']
    cv_folds = config_dict['cv_folds']
    study_name = config_dict['study_name']
    legend_model = config_dict['legend_model']

    print_best_hyperparams(optuna_output_dir,
                           cv_folds,
                           study_name,
                           legend_model)


if __name__ == '__main__':
    main()
