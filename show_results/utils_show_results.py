import matplotlib.pyplot as plt
import os
import numpy as np
from typing import Tuple
from utils_tdinoto.utils_strings import keep_only_digits
from utils_tdinoto.utils_lists import load_list_from_disk_with_pickle, flatten_list
from sklearn.metrics import auc, roc_curve, precision_recall_curve, accuracy_score, confusion_matrix, recall_score, precision_score, f1_score


__author__ = "Tommaso Di Noto"
__version__ = "0.0.1"
__email__ = "tommydino@hotmail.it"
__status__ = "Prototype"


def plot_roc_curve_wo_std(tpr_vgg_baseline,
                          auc_vgg_baseline,
                          legend_label_vgg_baseline,
                          tpr_vgg_transfer_learning,
                          auc_vgg_transfer_learning,
                          legend_label_vgg_transfer_learning,
                          tpr_seresnext_baseline,
                          auc_seresnext_baseline,
                          legend_label_seresnext_baseline,
                          tpr_seresnext_transfer_learning,
                          auc_seresnext_transfer_learning,
                          legend_label_seresnext_transfer_learning,
                          tpr_vgg_baseline_brats,
                          auc_vgg_baseline_brats,
                          legend_label_vgg_baseline_brats_inference,
                          tpr_vgg_transfer_learning_brats,
                          auc_vgg_transfer_learning_brats,
                          legend_label_vgg_transfer_learning_brats_inference,
                          tpr_seresnext_baseline_brats,
                          auc_seresnext_baseline_brats,
                          legend_label_seresnext_baseline_brats_inference,
                          tpr_seresnext_transfer_learning_brats,
                          auc_seresnext_transfer_learning_brats,
                          legend_label_seresnext_transfer_learning_brats_inference,
                          cv_folds):

    fig, ax = plt.subplots()
    mean_fpr = np.linspace(0, 1, 100)
    # VGG-BASELINE
    ax.plot(mean_fpr, tpr_vgg_baseline, color="b", label=r'{} (AUC = {:.2f})'.format(legend_label_vgg_baseline, auc_vgg_baseline), lw=3, alpha=.8)
    # VGG-TRANSFER LEARNING
    ax.plot(mean_fpr, tpr_vgg_transfer_learning, color="g", label=r'{} (AUC = {:.2f})'.format(legend_label_vgg_transfer_learning, auc_vgg_transfer_learning), lw=3, alpha=.8)
    # SERESNEXT BASELINE
    ax.plot(mean_fpr, tpr_seresnext_baseline, color="k", label=r'{} (AUC = {:.2f})'.format(legend_label_seresnext_baseline, auc_seresnext_baseline), lw=3, alpha=.8)
    # SERESNEXT TRANSFER LEARNING
    ax.plot(mean_fpr, tpr_seresnext_transfer_learning, color="y", label=r'{} (AUC = {:.2f})'.format(legend_label_seresnext_transfer_learning, auc_seresnext_transfer_learning), lw=3, alpha=.8)

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='No Skill', alpha=.8)  # draw chance line
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
    ax.set_title("ROC curves; {} cross-validation folds".format(cv_folds), weight="bold", fontsize=15)
    ax.set_xlabel('FPR (1- specificity)', fontsize=12)
    ax.set_ylabel('TPR (sensitivity)', fontsize=12)
    ax.legend(loc="lower right")

    # ------------------------------------------ BRATS inference
    fig, ax = plt.subplots()
    # VGG-BASELINE
    ax.plot(mean_fpr, tpr_vgg_baseline_brats, color="b", label=r'{} (AUC = {:.2f})'.format(legend_label_vgg_baseline_brats_inference, auc_vgg_baseline_brats), lw=3, alpha=.8)
    # VGG-TRANSFER LEARNING
    ax.plot(mean_fpr, tpr_vgg_transfer_learning_brats, color="g", label=r'{} (AUC = {:.2f})'.format(legend_label_vgg_transfer_learning_brats_inference, auc_vgg_transfer_learning_brats), lw=3, alpha=.8)
    # SERESNEXT BASELINE
    ax.plot(mean_fpr, tpr_seresnext_baseline_brats, color="k", label=r'{} (AUC = {:.2f})'.format(legend_label_seresnext_baseline_brats_inference, auc_seresnext_baseline_brats), lw=3, alpha=.8)
    # SERESNEXT TRANSFER LEARNING
    ax.plot(mean_fpr, tpr_seresnext_transfer_learning_brats, color="y", label=r'{} (AUC = {:.2f})'.format(legend_label_seresnext_transfer_learning_brats_inference, auc_seresnext_transfer_learning_brats), lw=3, alpha=.8)

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='No Skill', alpha=.8)  # draw chance line
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
    ax.set_title("ROC curves; BraTS inference", weight="bold", fontsize=15)
    ax.set_xlabel('FPR (1- specificity)', fontsize=12)
    ax.set_ylabel('TPR (sensitivity)', fontsize=12)
    ax.legend(loc="lower right")


def extract_average_values_pr(nb_random_realizations: int,
                              recall_list: list,
                              precision_list: list,
                              aupr_values: list) -> Tuple[float, float, float, float, float, float]:
    """This function extracts the average values for the PR curve across the random realizations
    Args:
        nb_random_realizations: number of cross-validation runs that were performed
        recall_list: it contains the recall values across the random realizations
        precision_list: it contains the precision values across the random realizations
        aupr_values: it contains the AUPR values across the random realizations
    Returns:
        mean_prec: mean precision value across random realizations
        std_prec: standard deviation precision value across random realizations
        precisions_upper: upper bound (+1 std)
        precisions_lower: lower bound (+1 std)
        avg_aupr: mean AUPR value across random realizations
        std_aupr: standard deviation AUPR value across random realizations
    """
    precisions = []  # type: list
    mean_recall = np.linspace(0, 1, 100)

    # since across random iterations the recall and precision vectors have different length, we must interpolate
    for iteration in range(nb_random_realizations):
        # flip the vectors because np.interp expects increasing values of the x-axis
        rec_flip = np.flip(recall_list[iteration])
        prec_flip = np.flip(precision_list[iteration])

        interp_prec = np.interp(mean_recall, rec_flip, prec_flip)
        interp_prec[0] = 1.0
        precisions.append(interp_prec)

    mean_prec = np.mean(precisions, axis=0)
    std_prec = np.std(precisions, axis=0)
    precisions_upper = np.minimum(mean_prec + std_prec, 1)
    precisions_lower = np.maximum(mean_prec - std_prec, 0)

    # average also the AUPR
    avg_aupr = np.mean(aupr_values)
    std_aupr = np.std(aupr_values)

    return mean_prec, std_prec, precisions_upper, precisions_lower, avg_aupr, std_aupr


def extract_average_values_roc(nb_random_realizations: int,
                               fpr_list: list,
                               tpr_list: list,
                               auc_values: list) -> Tuple[float, float, float, float, float, float]:
    """This function extracts the average values for the ROC curve across the random realizations
        Args:
            nb_random_realizations: number of cross-validation runs that were performed
            fpr_list: it contains the FPR values across the random realizations
            tpr_list: it contains the TPR values across the random realizations
            auc_values: it contains the AUC values across the random realizations
        Returns:
            mean_tpr: mean TPR value across random realizations
            std_tpr: standard deviation TPR value across random realizations
            tprs_upper: upper bound (+1 std)
            tprs_lower: lower bound (+1 std)
            avg_auc: mean AUC value across random realizations
            std_auc: standard deviation AUC value across random realizations
        """
    tprs = []  # type: list
    mean_fpr = np.linspace(0, 1, 100)

    # since across random iterations the fpr and tpr vectors have different length, we must interpolate
    for iteration in range(nb_random_realizations):
        interp_tpr = np.interp(mean_fpr, fpr_list[iteration], tpr_list[iteration])
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

    # average also the AUC
    avg_auc = np.mean(auc_values)
    std_auc = np.std(auc_values)

    return mean_tpr, std_tpr, tprs_upper, tprs_lower, avg_auc, std_auc


def plot_avg_roc_curve_with_std(fpr_vgg_baseline,
                                tpr_vgg_baseline,
                                auc_vgg_baseline,
                                legend_label_vgg_baseline,
                                fpr_vgg_transfer_learning,
                                tpr_vgg_transfer_learning,
                                auc_vgg_transfer_learning,
                                legend_label_vgg_transfer_learning,
                                fpr_seresnext_baseline,
                                tpr_seresnext_baseline,
                                auc_seresnext_baseline,
                                legend_label_seresnext_baseline,
                                fpr_seresnext_transfer_learning,
                                tpr_seresnext_transfer_learning,
                                auc_seresnext_transfer_learning,
                                legend_label_seresnext_transfer_learning,
                                cv_folds):
    """This function plots the average ROC curve across the random realizations
    """
    # VGG-BASELINE
    mean_tpr_vgg_baseline, _, tpr_upper_vgg_baseline, tpr_lower_vgg_baseline,\
        mean_auc_vgg_baseline, _ = extract_average_values_roc(cv_folds, fpr_vgg_baseline, tpr_vgg_baseline, auc_vgg_baseline)

    # VGG-TRANSFER LEARNING
    mean_tpr_vgg_tl, _, tpr_upper_vgg_tl, tpr_lower_vgg_tl,\
        mean_auc_vgg_tl, _ = extract_average_values_roc(cv_folds, fpr_vgg_transfer_learning, tpr_vgg_transfer_learning, auc_vgg_transfer_learning)

    # SERESNEXT BASELINE
    mean_tpr_seresnext_baseline, _, tpr_upper_seresnext_baseline, tpr_lower_seresnext_baseline, \
        mean_auc_seresnext_baseline, _ = extract_average_values_roc(cv_folds, fpr_seresnext_baseline, tpr_seresnext_baseline, auc_seresnext_baseline)

    # SERESNEXT TRANSFER LEARNING
    mean_tpr_seresnext_tl, _, tpr_upper_seresnext_tl, tpr_lower_seresnext_tl, \
        mean_auc_seresnext_tl, _ = extract_average_values_roc(cv_folds, fpr_seresnext_transfer_learning, tpr_seresnext_transfer_learning, auc_seresnext_transfer_learning)

    fig, ax = plt.subplots()
    mean_fpr = np.linspace(0, 1, 100)
    # VGG-BASELINE
    ax.plot(mean_fpr, mean_tpr_vgg_baseline, color="b", label=r'{} (mean AUC = {:.2f})'.format(legend_label_vgg_baseline, mean_auc_vgg_baseline), lw=3, alpha=.8)
    ax.fill_between(mean_fpr, tpr_lower_vgg_baseline, tpr_upper_vgg_baseline, color='b', alpha=.2, label=r'$\pm$ 1 std. dev.')
    # VGG-TRANSFER LEARNING
    ax.plot(mean_fpr, mean_tpr_vgg_tl, color="g", label=r'{} (mean AUC = {:.2f})'.format(legend_label_vgg_transfer_learning, mean_auc_vgg_tl), lw=3, alpha=.8)
    ax.fill_between(mean_fpr, tpr_lower_vgg_tl, tpr_upper_vgg_tl, color='g', alpha=.2, label=r'$\pm$ 1 std. dev.')
    # SERESNEXT BASELINE
    ax.plot(mean_fpr, mean_tpr_seresnext_baseline, color="k", label=r'{} (mean AUC = {:.2f})'.format(legend_label_seresnext_baseline, mean_auc_seresnext_baseline), lw=3, alpha=.8)
    ax.fill_between(mean_fpr, tpr_lower_seresnext_baseline, tpr_upper_seresnext_baseline, color='k', alpha=.2, label=r'$\pm$ 1 std. dev.')
    # SERESNEXT TRANSFER LEARNING
    ax.plot(mean_fpr, mean_tpr_seresnext_tl, color="y", label=r'{} (mean AUC = {:.2f})'.format(legend_label_seresnext_transfer_learning, mean_auc_seresnext_tl), lw=3, alpha=.8)
    ax.fill_between(mean_fpr, tpr_lower_seresnext_tl, tpr_upper_seresnext_tl, color='y', alpha=.2, label=r'$\pm$ 1 std. dev.')

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='No Skill', alpha=.8)  # draw chance line
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
    ax.set_title("ROC curves; {} cross-validation folds".format(cv_folds), weight="bold", fontsize=15)
    ax.set_xlabel('FPR (1- specificity)', fontsize=12)
    ax.set_ylabel('TPR (sensitivity)', fontsize=12)
    ax.legend(loc="lower right")


def extract_roc_curve_params_across_folds_flattened(output_dir,
                                                    brats_inference=False):

    if not brats_inference:
        y_true_flat, y_pred_probab_flat = extract_flat_preds_and_ground_truth_from_cv_folds(output_dir)
    else:
        y_true_flat, y_pred_probab_flat = load_list_from_disk_with_pickle(os.path.join(output_dir, "y_true.pkl")), \
                                          load_list_from_disk_with_pickle(os.path.join(output_dir, "y_pred_probab_avg.pkl"))

    fpr, tpr, _ = roc_curve(y_true_flat, y_pred_probab_flat, pos_label=1)
    tpr[0] = 0.0  # ensure that first element is 0
    tpr[-1] = 1.0  # ensure that last element is 1
    auc_roc = auc(fpr, tpr)

    mean_fpr = np.linspace(0, 1, 100)
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0

    return mean_fpr, interp_tpr, auc_roc
        

def extract_roc_curve_params_across_folds(output_dir: str) -> Tuple[list, list, list]:
    """This function extracts the parameters of the ROC curve
    Args:
        output_dir: path to output directory
    Returns:
        fpr_across_folds: it contains the fpr values
        tpr_across_folds: it contains the tpr values
        auc_across_folds: it contains the auc values
    """
    files_in_output_dir = os.listdir(output_dir)
    folds = [file for file in files_in_output_dir if "fold" in file and os.path.isdir(os.path.join(output_dir, file))]
    assert len(folds) == 5, "We expect to have 5 folds"
    fpr_across_folds = []  # ground truth labels
    tpr_across_folds = []  # binary predictions
    auc_across_folds = []  # probabilistic predictions
    for fold in folds:
        fold_nb = keep_only_digits(fold)
        y_true = load_list_from_disk_with_pickle(os.path.join(output_dir, fold, "y_true_fold{}.pkl".format(fold_nb)))
        y_pred_probab = load_list_from_disk_with_pickle(os.path.join(output_dir, fold, "y_pred_probabilistic_fold{}.pkl".format(fold_nb)))

        fpr, tpr, _ = roc_curve(y_true, y_pred_probab, pos_label=1)
        tpr[0] = 0.0  # ensure that first element is 0
        tpr[-1] = 1.0  # ensure that last element is 1
        auc_roc = auc(fpr, tpr)

        fpr_across_folds.append(fpr)
        tpr_across_folds.append(tpr)
        auc_across_folds.append(auc_roc)

    return fpr_across_folds, tpr_across_folds, auc_across_folds


def extract_pr_curve_params_across_folds(output_dir: str) -> Tuple[list, list, list]:
    """This function extracts the parameters of the PR curve
    Args:
        output_dir: path to output directory
    Returns:
        recall_across_folds: it contains the recall values
        precision_across_folds: it contains the precision values
        aupr_across_folds: it contains the aupr values
    """
    files_in_output_dir = os.listdir(output_dir)
    folds = [file for file in files_in_output_dir if "fold" in file and os.path.isdir(os.path.join(output_dir, file))]
    assert len(folds) == 5, "We expect to have 5 folds"
    recall_across_folds = []  # ground truth labels
    precision_across_folds = []  # binary predictions
    aupr_across_folds = []  # probabilistic predictions
    for fold in folds:
        fold_nb = keep_only_digits(fold)
        y_true = load_list_from_disk_with_pickle(os.path.join(output_dir, fold, "y_true_fold{}.pkl".format(fold_nb)))
        y_pred_probab = load_list_from_disk_with_pickle(os.path.join(output_dir, fold, "y_pred_probabilistic_fold{}.pkl".format(fold_nb)))

        precision, recall, _ = precision_recall_curve(y_true, y_pred_probab)
        aupr = auc(recall, precision)

        recall_across_folds.append(recall)
        precision_across_folds.append(precision)
        aupr_across_folds.append(aupr)

    return recall_across_folds, precision_across_folds, aupr_across_folds


def find_average_no_skill_across_folds_pr_curve(out_dir: str, brats_inference=False) -> float:
    """This function finds the no-skill value of the PR curve across the cross-validation folds.
    """
    if not brats_inference:
        no_skill_across_realizations = []
        all_files = os.listdir(out_dir)
        folds = [item for item in all_files if os.path.isdir(os.path.join(out_dir, item)) and "fold" in item]
        for fold in folds:
            fold_nb = keep_only_digits(fold)
            y_test = load_list_from_disk_with_pickle(os.path.join(out_dir, fold, "y_true_fold{}.pkl".format(fold_nb)))
            y_test_np = np.asarray(y_test)
            no_skill = len(y_test_np[y_test_np == 1]) / len(y_test_np)  # count proportion of 1 over total length
            no_skill_across_realizations.append(no_skill)

        average_no_skill = np.mean(no_skill_across_realizations)

    else:
        y_test = load_list_from_disk_with_pickle(os.path.join(out_dir, "y_true.pkl"))
        y_test_np = np.asarray(y_test)
        average_no_skill = len(y_test_np[y_test_np == 1]) / len(y_test_np)

    return average_no_skill


def plot_avg_pr_curve_with_std(recall_vgg_baseline,
                               precision_vgg_baseline,
                               aupr_vgg_baseline,
                               legend_label_vgg_baseline,
                               recall_vgg_transfer_learning,
                               precision_vgg_transfer_learning,
                               aupr_vgg_transfer_learning,
                               legend_label_vgg_transfer_learning,
                               recall_seresnext_baseline,
                               precision_seresnext_baseline,
                               aupr_seresnext_baseline,
                               legend_label_seresnext_baseline,
                               recall_seresnext_transfer_learning,
                               precision_seresnext_transfer_learning,
                               aupr_seresnext_transfer_learning,
                               legend_label_seresnext_transfer_learning,
                               cv_folds,
                               avg_no_skill_across_folds):
    # VGG BASELINE
    mean_prec_vgg_baseline, _, precisions_upper_vgg_baseline, precisions_lower_vgg_baseline,\
        avg_aupr_vgg_baseline, _ = extract_average_values_pr(cv_folds, recall_vgg_baseline, precision_vgg_baseline, aupr_vgg_baseline)

    # VGG TRANSFER LEARNING
    mean_prec_vgg_tl, _, precisions_upper_vgg_tl, precisions_lower_vgg_tl, \
        avg_aupr_vgg_tl, _ = extract_average_values_pr(cv_folds, recall_vgg_transfer_learning, precision_vgg_transfer_learning, aupr_vgg_transfer_learning)

    # SERESNEXT BASELINE
    mean_prec_seresnext_baseline, _, precisions_upper_seresnext_baseline, precisions_lower_seresnext_baseline, \
        avg_aupr_seresnext_baseline, _ = extract_average_values_pr(cv_folds, recall_seresnext_baseline, precision_seresnext_baseline, aupr_seresnext_baseline)

    # SERESNEXT TRANSFER LEARNING
    mean_prec_seresnext_tl, _, precisions_upper_seresnext_tl, precisions_lower_seresnext_tl, \
        avg_aupr_seresnext_tl, _ = extract_average_values_pr(cv_folds, recall_seresnext_transfer_learning, precision_seresnext_transfer_learning, aupr_seresnext_transfer_learning)

    fig, ax = plt.subplots()
    mean_recall = np.linspace(0, 1, 100)
    # VGG BASELINE
    ax.plot(mean_recall, mean_prec_vgg_baseline, color="b", label=r'{} (mean AUPR = {:.2f})'.format(legend_label_vgg_baseline, avg_aupr_vgg_baseline), lw=3, alpha=.8)
    ax.fill_between(mean_recall, precisions_lower_vgg_baseline, precisions_upper_vgg_baseline, color='b', alpha=.2, label=r'$\pm$ 1 std. dev.')
    # VGG TRANSFER LEARNING
    ax.plot(mean_recall, mean_prec_vgg_tl, color="g", label=r'{} (mean AUPR = {:.2f})'.format(legend_label_vgg_transfer_learning, avg_aupr_vgg_tl), lw=3, alpha=.8)
    ax.fill_between(mean_recall, precisions_lower_vgg_tl, precisions_upper_vgg_tl, color='g', alpha=.2, label=r'$\pm$ 1 std. dev.')
    # SERESNEXT BASELINE
    ax.plot(mean_recall, mean_prec_seresnext_baseline, color="k", label=r'{} (mean AUPR = {:.2f})'.format(legend_label_seresnext_baseline, avg_aupr_seresnext_baseline), lw=3, alpha=.8)
    ax.fill_between(mean_recall, precisions_lower_seresnext_baseline, precisions_upper_seresnext_baseline, color='k', alpha=.2, label=r'$\pm$ 1 std. dev.')
    # SERESNEXT TRANSFER LEARNING
    ax.plot(mean_recall, mean_prec_seresnext_tl, color="y", label=r'{} (mean AUPR = {:.2f})'.format(legend_label_seresnext_transfer_learning, avg_aupr_seresnext_tl), lw=3, alpha=.8)
    ax.fill_between(mean_recall, precisions_lower_seresnext_tl, precisions_upper_seresnext_tl, color='y', alpha=.2, label=r'$\pm$ 1 std. dev.')

    ax.plot([0, 1], [avg_no_skill_across_folds, avg_no_skill_across_folds], linestyle='--', lw=2, color="r", label='No Skill', alpha=.8)
    ax.set_title("PR curves; {} cross-validation folds".format(cv_folds), weight="bold", fontsize=15)
    ax.set_xlabel('Recall (sensitivity)', fontsize=12)
    ax.set_ylabel('Precision (PPV)', fontsize=12)
    ax.legend(loc="center left")
    plt.show()


def extract_flat_preds_and_ground_truth_from_cv_folds(output_dir: str) -> Tuple[list, list]:
    files_in_output_dir = os.listdir(output_dir)
    folds = [file for file in files_in_output_dir if "fold" in file and os.path.isdir(os.path.join(output_dir, file))]
    assert len(folds) == 5, "We expect to have 5 folds"
    y_true_across_folds = []  # ground truth labels
    y_pred_probab_across_folds = []  # binary predictions
    for fold in folds:
        fold_nb = keep_only_digits(fold)
        y_true = load_list_from_disk_with_pickle(os.path.join(output_dir, fold, "y_true_fold{}.pkl".format(fold_nb)))
        y_pred_probab = load_list_from_disk_with_pickle(os.path.join(output_dir, fold, "y_pred_probabilistic_fold{}.pkl".format(fold_nb)))

        y_true_across_folds.append(y_true)
        y_pred_probab_across_folds.append(y_pred_probab)

    y_true_flat = flatten_list(y_true_across_folds)
    y_pred_probab_flat = flatten_list(y_pred_probab_across_folds)

    return y_true_flat, y_pred_probab_flat


def extract_pr_curve_params_across_folds_flattened(output_dir: str, brats_inference=False):

    if not brats_inference:
        y_true_flat, y_pred_probab_flat = extract_flat_preds_and_ground_truth_from_cv_folds(output_dir)
    else:
        y_true_flat, y_pred_probab_flat = load_list_from_disk_with_pickle(os.path.join(output_dir, "y_true.pkl")), \
                                          load_list_from_disk_with_pickle(os.path.join(output_dir, "y_pred_probab_avg.pkl"))

    precision, recall, _ = precision_recall_curve(y_true_flat, y_pred_probab_flat)
    aupr = auc(recall, precision)

    mean_recall = np.linspace(0, 1, 100)

    # flip the vectors because np.interp expects increasing values of the x-axis
    rec_flip = np.flip(recall)
    prec_flip = np.flip(precision)

    interp_prec = np.interp(mean_recall, rec_flip, prec_flip)
    interp_prec[0] = 1.0

    return mean_recall, interp_prec, aupr


def plot_pr_curve_wo_std(precision_vgg_baseline,
                         aupr_vgg_baseline,
                         legend_label_vgg_baseline,
                         precision_vgg_transfer_learning,
                         aupr_vgg_transfer_learning,
                         legend_label_vgg_transfer_learning,
                         precision_seresnext_baseline,
                         aupr_seresnext_baseline,
                         legend_label_seresnext_baseline,
                         precision_seresnext_transfer_learning,
                         aupr_seresnext_transfer_learning,
                         legend_label_seresnext_transfer_learning,
                         cv_folds,
                         avg_no_skill_across_folds,
                         precision_vgg_baseline_brats,
                         aupr_vgg_baseline_brats,
                         legend_label_vgg_baseline_brats_inference,
                         precision_vgg_transfer_learning_brats,
                         aupr_vgg_transfer_learning_brats,
                         legend_label_vgg_transfer_learning_brats_inference,
                         precision_seresnext_baseline_brats,
                         aupr_seresnext_baseline_brats,
                         legend_label_seresnext_baseline_brats_inference,
                         precision_seresnext_transfer_learning_brats,
                         aupr_seresnext_transfer_learning_brats,
                         legend_label_seresnext_transfer_learning_brats_inference,
                         avg_no_skill_brats_inference):

    fig, ax = plt.subplots()
    mean_recall = np.linspace(0, 1, 100)
    # VGG BASELINE
    ax.plot(mean_recall, precision_vgg_baseline, color="b", label=r'{} (AUPR = {:.2f})'.format(legend_label_vgg_baseline, aupr_vgg_baseline), lw=3, alpha=.8)
    # VGG TRANSFER LEARNING
    ax.plot(mean_recall, precision_vgg_transfer_learning, color="g", label=r'{} (AUPR = {:.2f})'.format(legend_label_vgg_transfer_learning, aupr_vgg_transfer_learning), lw=3, alpha=.8)
    # SERESNEXT BASELINE
    ax.plot(mean_recall, precision_seresnext_baseline, color="k", label=r'{} (AUPR = {:.2f})'.format(legend_label_seresnext_baseline, aupr_seresnext_baseline), lw=3, alpha=.8)
    # SERESNEXT TRANSFER LEARNING
    ax.plot(mean_recall, precision_seresnext_transfer_learning, color="y", label=r'{} (AUPR = {:.2f})'.format(legend_label_seresnext_transfer_learning, aupr_seresnext_transfer_learning), lw=3, alpha=.8)

    ax.plot([0, 1], [avg_no_skill_across_folds, avg_no_skill_across_folds], linestyle='--', lw=2, color="r", label='No Skill', alpha=.8)
    ax.set_title("PR curves; {} cross-validation folds".format(cv_folds), weight="bold", fontsize=15)
    ax.set_xlabel('Recall (sensitivity)', fontsize=12)
    ax.set_ylabel('Precision (PPV)', fontsize=12)
    ax.legend(loc="upper right")

    # BRATS inference
    fig, ax = plt.subplots()
    # VGG BASELINE
    ax.plot(mean_recall, precision_vgg_baseline_brats, color="b", label=r'{} (AUPR = {:.2f})'.format(legend_label_vgg_baseline_brats_inference, aupr_vgg_baseline_brats), lw=3, alpha=.8)
    # VGG TRANSFER LEARNING
    ax.plot(mean_recall, precision_vgg_transfer_learning_brats, color="g", label=r'{} (AUPR = {:.2f})'.format(legend_label_vgg_transfer_learning_brats_inference, aupr_vgg_transfer_learning_brats), lw=3, alpha=.8)
    # SERESNEXT BASELINE
    ax.plot(mean_recall, precision_seresnext_baseline_brats, color="k", label=r'{} (AUPR = {:.2f})'.format(legend_label_seresnext_baseline_brats_inference, aupr_seresnext_baseline_brats), lw=3, alpha=.8)
    # SERESNEXT TRANSFER LEARNING
    ax.plot(mean_recall, precision_seresnext_transfer_learning_brats, color="y", label=r'{} (AUPR = {:.2f})'.format(legend_label_seresnext_transfer_learning_brats_inference, aupr_seresnext_transfer_learning_brats), lw=3, alpha=.8)
    ax.plot([0, 1], [avg_no_skill_brats_inference, avg_no_skill_brats_inference], linestyle='--', lw=2, color="r", label='No Skill', alpha=.8)
    ax.set_title("PR curves; BraTS inference".format(cv_folds), weight="bold", fontsize=15)
    ax.set_xlabel('Recall (sensitivity)', fontsize=12)
    ax.set_ylabel('Precision (PPV)', fontsize=12)
    ax.legend(loc="center left")
    plt.show()


def compute_observed_diff_aupr(y_probab_model_1, y_true, y_probab_model_2, test_statistic):

    if test_statistic == "aupr":
        # compute AUPR model 1
        precision_model1, recall_model1, _ = precision_recall_curve(y_true, y_probab_model_1)
        aupr_model1 = auc(recall_model1, precision_model1)

        # compute AUPR model 2
        precision_model2, recall_model2, _ = precision_recall_curve(y_true, y_probab_model_2)
        aupr_model2 = auc(recall_model2, precision_model2)

        observed_diff = aupr_model2 - aupr_model1
    elif test_statistic == "auc":
        # compute AUC model 1
        fpr_model_1, tpr_model_1, _ = roc_curve(y_true, y_probab_model_1, pos_label=1)
        tpr_model_1[0] = 0.0  # ensure that first element is 0
        tpr_model_1[-1] = 1.0  # ensure that last element is 1
        auc_roc_model_1 = auc(fpr_model_1, tpr_model_1)

        # compute AUC model 2
        fpr_model_2, tpr_model_2, _ = roc_curve(y_true, y_probab_model_2, pos_label=1)
        tpr_model_2[0] = 0.0  # ensure that first element is 0
        tpr_model_2[-1] = 1.0  # ensure that last element is 1
        auc_roc_model_2 = auc(fpr_model_2, tpr_model_2)

        observed_diff = auc_roc_model_2 - auc_roc_model_1
    else:
        raise ValueError("Only 'aupr' and 'auc' are accepted as test statistic; got {} instead".format(test_statistic))

    return observed_diff


def one_permutation_matching_labels(y_probab_model_1,
                                    y_true,
                                    y_probab_model_2,
                                    test_statistic):
    # mix all predictions together
    all_y_preds = y_probab_model_1 + y_probab_model_2

    # duplicate ground truth vector (such that we can keep a correspondence between predictions and labels)
    y_true_duplicated = y_true + y_true

    # randomly draw one resample
    idxs_random_sample_1 = sorted(list(np.random.choice(list(range(len(all_y_preds))), len(y_true), replace=False)))
    random_y_pred_1 = [all_y_preds[idx] for idx in idxs_random_sample_1]
    corresponding_y_true_1 = [y_true_duplicated[idx] for idx in idxs_random_sample_1]  # use same indexes also for the labels

    # create second resample by taking remaining
    idxs_random_sample_2 = sorted([idx for idx in list(range(len(all_y_preds))) if idx not in idxs_random_sample_1])
    random_y_pred_2 = [all_y_preds[idx] for idx in idxs_random_sample_2]
    corresponding_y_true_2 = [y_true_duplicated[idx] for idx in idxs_random_sample_2]

    if test_statistic == "aupr":
        # compute AUPR for random resample 1
        precision_sample_1, recall_sample_1, _ = precision_recall_curve(corresponding_y_true_1, random_y_pred_1)
        aupr_sample_1 = auc(recall_sample_1, precision_sample_1)

        # compute AUPR for random resample 2
        precision_sample_2, recall_sample_2, _ = precision_recall_curve(corresponding_y_true_2, random_y_pred_2)
        aupr_sample_2 = auc(recall_sample_2, precision_sample_2)

        permuted_difference = aupr_sample_2 - aupr_sample_1
    elif test_statistic == "auc":
        # compute AUC for random resample 1
        fpr_sample_1, tpr_sample_1, _ = roc_curve(corresponding_y_true_1, random_y_pred_1, pos_label=1)
        tpr_sample_1[0] = 0.0  # ensure that first element is 0
        tpr_sample_1[-1] = 1.0  # ensure that last element is 1
        auc_roc_sample_1 = auc(fpr_sample_1, tpr_sample_1)

        # compute AUC for random resample 2
        fpr_sample_2, tpr_sample_2, _ = roc_curve(corresponding_y_true_2, random_y_pred_2, pos_label=1)
        tpr_sample_2[0] = 0.0  # ensure that first element is 0
        tpr_sample_2[-1] = 1.0  # ensure that last element is 1
        auc_roc_sample_2 = auc(fpr_sample_2, tpr_sample_2)

        permuted_difference = auc_roc_sample_2 - auc_roc_sample_1
    else:
        raise ValueError("Only 'aupr' and 'auc' are accepted as test statistic; got {} instead".format(test_statistic))

    return permuted_difference


def run_permutations(nb_permutations: int,
                     y_probab_model_1: list,
                     y_true: list,
                     y_probab_model_2: list,
                     test_statistic: str) -> list:
    permuted_auc_differences = []  # will contain the randomly permuted AUC values
    for i in range(nb_permutations):
        if i % 2000 == 0:  # only print every XX permutation
            print("Permutation {}".format(i))
        one_permuted_auc_difference = one_permutation_matching_labels(y_probab_model_1, y_true, y_probab_model_2, test_statistic)
        permuted_auc_differences.append(one_permuted_auc_difference)  # append to external list

    return permuted_auc_differences


def permutation_tests(output_dir_model1: str,
                      legend_label_model1: str,
                      output_dir_model2: str,
                      legend_label_model2: str,
                      nb_permutations: int,
                      test_statistic: str,
                      figure_nb: int,
                      brats_inference: bool = False) -> None:

    print("\n------------------------ {} vs. {}".format(legend_label_model1, legend_label_model2))
    print("------ test statistic: {}; nb. permutations: {}".format(test_statistic, nb_permutations))

    if not brats_inference:  # if we're not doing inference on BraTS
        y_true_model1, y_pred_probab_model1 = extract_flat_preds_and_ground_truth_from_cv_folds(output_dir_model1)
        y_true_model2, y_pred_probab_model2 = extract_flat_preds_and_ground_truth_from_cv_folds(output_dir_model2)
    else:  # if instead we are evaluating results of the inference on BraTS
        y_true_model1, y_pred_probab_model1 = load_list_from_disk_with_pickle(os.path.join(output_dir_model1, "y_true.pkl")),\
                                              load_list_from_disk_with_pickle(os.path.join(output_dir_model1, "y_pred_probab_avg.pkl"))
        y_true_model2, y_pred_probab_model2 = load_list_from_disk_with_pickle(os.path.join(output_dir_model2, "y_true.pkl")),\
                                              load_list_from_disk_with_pickle(os.path.join(output_dir_model2, "y_pred_probab_avg.pkl"))

    assert y_true_model1 == y_true_model2, "Ground truth vectors should be identical"

    # compute observed difference
    observed_difference = compute_observed_diff_aupr(y_pred_probab_model1, y_true_model1, y_pred_probab_model2, test_statistic)

    # compute permuted differences
    permuted_differences = run_permutations(nb_permutations, y_pred_probab_model1, y_true_model1, y_pred_probab_model2, test_statistic)

    # show distribution
    plt.figure(figure_nb)
    hist = np.histogram(permuted_differences, bins=20)
    _ = plt.hist(permuted_differences, bins=20)
    plt.title("Test statistic: {}".format(test_statistic))

    # add vertical line representing the observed difference
    plt.axvline(observed_difference, 0, hist[0][int(len(hist) / 2)], linewidth=4, color="r")

    # count number of samples above the observed value
    count_samples_above_observed_value = (np.asarray(permuted_differences) > observed_difference).sum()

    # compute p-value: it corresponds to the proportion of values above the observed wrt the total
    p_value = count_samples_above_observed_value / len(permuted_differences)

    if p_value <= 0.05 or p_value >= 0.95:
        print("Significant difference. p-value = {}".format(p_value))
    else:
        print("NON-significant difference. p-value = {}".format(p_value))


def classification_metrics(y_true: np.ndarray,
                           y_pred: np.ndarray,
                           binary: bool = True) -> Tuple[np.ndarray, float, float, float, float, float, float, float, float, float]:
    """This function computes some standard classification metrics for a binary problem
    Args:
        y_true: ground truth labels
        y_pred: predicted labels
        binary: indicates whether the classification is binary (i.e. two classes) or not (i.e. more classes)
    Returns:
        conf_mat: confusion matrix
        acc: accuracy
        rec: recall (i.e. sensitivity, or true positive rate)
        spec: specificity (i.e. true negative rate)
        prec: precision (i.e. positive predictive value)
        npv: negative predictive value
        f1: F1-score (i.e. harmonic mean of precision and recall)
    """
    conf_mat = confusion_matrix(y_true, y_pred)  # type: np.ndarray
    acc, rec_macro, spec, prec_macro, npv, f1_macro, rec_weighted, prec_weighted, f1_weighted = 0., 0., 0., 0., 0., 0., 0., 0., 0.  # initialize all metrics to None
    if binary:
        assert conf_mat.shape == (2, 2), "Confusion Matrix does not correspond to a binary task"
        tn = conf_mat[0][0]
        fn = conf_mat[1][0]
        fp = conf_mat[0][1]

        acc = accuracy_score(y_true, y_pred)
        rec_macro = recall_score(y_true, y_pred)
        spec = tn / (tn + fp)
        prec_macro = precision_score(y_true, y_pred)
        npv = tn / (tn + fn)
        f1_macro = f1_score(y_true, y_pred)

    else:
        acc = accuracy_score(y_true, y_pred)
        rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
        prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro')

        rec_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        prec_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted')

    return conf_mat, acc, rec_macro, spec, prec_macro, npv, f1_macro, rec_weighted, prec_weighted, f1_weighted


def plot_roc_curve(flat_y_test: list,
                   flat_y_pred_proba: list,
                   cv_folds: int,
                   embedding_label: str = "doc2vec",
                   plot: bool = True) -> Tuple[np.ndarray, np.ndarray, float]:
    """This function computes FPR, TPR and AUC. Then, it plots the ROC curve
    Args:
        flat_y_test: labels
        flat_y_pred_proba: predictions
        cv_folds: number of folds in the cross-validation
        embedding_label: embedding algorithm that was used
        plot: if True, the ROC curve will be displayed
    """
    fpr, tpr, _ = roc_curve(flat_y_test, flat_y_pred_proba, pos_label=1)
    tpr[0] = 0.0  # ensure that first element is 0
    tpr[-1] = 1.0  # ensure that last element is 1
    auc_roc = auc(fpr, tpr)
    if plot:
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, color="b", label=r'{} (AUC = {:.2f})'.format(embedding_label, auc_roc), lw=2, alpha=.8)
        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
        ax.set_title("ROC curve; {}-fold CV".format(cv_folds), weight="bold", fontsize=15)
        ax.set_xlabel('FPR (1- specificity)', fontsize=12)
        ax.set_ylabel('TPR (sensitivity)', fontsize=12)
        ax.legend(loc="lower right")
    return fpr, tpr, auc_roc


def plot_pr_curve(flat_y_test: list,
                  flat_y_pred_proba: list,
                  cv_folds: int,
                  embedding_label: str = "doc2vec",
                  plot: bool = True) -> Tuple[np.ndarray, np.ndarray, float]:
    """This function computes precision, recall and AUPR. Then, it plots the PR curve
    Args:
        flat_y_test (list): labels
        flat_y_pred_proba (list): predictions
        cv_folds (int): number of folds in the cross-validation
        embedding_label (str): embedding algorithm that was used
        plot (bool): if True, the ROC curve will be displayed
    """
    precision, recall, _ = precision_recall_curve(flat_y_test, flat_y_pred_proba)
    aupr = auc(recall, precision)
    if plot:
        fig, ax = plt.subplots()
        ax.plot(recall, precision, color="g", label=r'{} (AUPR = {:.2f})'.format(embedding_label, aupr))
        ax.set_title("PR curve; {}-fold CV".format(cv_folds), weight="bold", fontsize=15)
        ax.set_xlabel('Recall (sensitivity)', fontsize=12)
        ax.set_ylabel('Precision (PPV)', fontsize=12)
        ax.legend(loc="lower left")
    return recall, precision, aupr
