from dataset_creation.utils_dataset_creation import load_config_file
from utils_tdinoto.utils_strings import str2bool
from show_results.utils_show_results import plot_avg_roc_curve_with_std, extract_roc_curve_params_across_folds,\
    extract_pr_curve_params_across_folds, plot_avg_pr_curve_with_std, find_average_no_skill_across_folds_pr_curve, \
    extract_roc_curve_params_across_folds_flattened, plot_roc_curve_wo_std, extract_pr_curve_params_across_folds_flattened, plot_pr_curve_wo_std


__author__ = "Tommaso Di Noto"
__version__ = "0.0.1"
__email__ = "tommydino@hotmail.it"
__status__ = "Prototype"


def plot_curves(cv_folds,
                output_dir_vgg_baseline,
                legend_label_vgg_baseline,
                output_dir_vgg_transfer_learning,
                legend_label_vgg_transfer_learning,
                output_dir_seresnext_baseline,
                legend_label_seresnext_baseline,
                output_dir_seresnext_transfer_learning,
                legend_label_seresnext_transfer_learning,
                output_dir_vgg_baseline_brats_inference,
                legend_label_vgg_baseline_brats_inference,
                output_dir_vgg_transfer_learning_brats_inference,
                legend_label_vgg_transfer_learning_brats_inference,
                output_dir_seresnext_baseline_brats_inference,
                legend_label_seresnext_baseline_brats_inference,
                output_dir_seresnext_transfer_learning_brats_inference,
                legend_label_seresnext_transfer_learning_brats_inference,
                plot_with_standard_deviation):

    if plot_with_standard_deviation:  # if we want the plots to have a shaded area as confidence interval
        # extract params of ROC curve (fpr, tpr, auc)
        fpr_vgg_baseline, tpr_vgg_baseline, auc_vgg_baseline = extract_roc_curve_params_across_folds(output_dir_vgg_baseline)
        fpr_vgg_transfer_learning, tpr_vgg_transfer_learning, auc_vgg_transfer_learning = extract_roc_curve_params_across_folds(output_dir_vgg_transfer_learning)
        fpr_seresnext_baseline, tpr_seresnext_baseline, auc_seresnext_baseline = extract_roc_curve_params_across_folds(output_dir_seresnext_baseline)
        fpr_seresnext_transfer_learning, tpr_seresnext_transfer_learning, auc_seresnext_transfer_learning = extract_roc_curve_params_across_folds(output_dir_seresnext_transfer_learning)

        # plot ROC curves
        plot_avg_roc_curve_with_std(fpr_vgg_baseline,
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
                                    cv_folds)

        # extract params of PR curve (precision, recall, aupr)
        recall_vgg_baseline, precision_vgg_baseline, aupr_vgg_baseline = extract_pr_curve_params_across_folds(output_dir_vgg_baseline)
        recall_vgg_transfer_learning, precision_vgg_transfer_learning, aupr_vgg_transfer_learning = extract_pr_curve_params_across_folds(output_dir_vgg_transfer_learning)
        recall_seresnext_baseline, precision_seresnext_baseline, aupr_seresnext_baseline = extract_pr_curve_params_across_folds(output_dir_seresnext_baseline)
        recall_seresnext_transfer_learning, precision_seresnext_transfer_learning, aupr_seresnext_transfer_learning = extract_pr_curve_params_across_folds(output_dir_seresnext_transfer_learning)

        # since we only use the ground truth, we can use any model, cause the ground truth is always the same
        avg_no_skill_across_folds = find_average_no_skill_across_folds_pr_curve(output_dir_vgg_baseline)

        # plot PR curves
        plot_avg_pr_curve_with_std(recall_vgg_baseline,
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
                                   avg_no_skill_across_folds)

    else:  # if instead we only plot the curves without any confidence interval

        # extract params of ROC curve (fpr, tpr, auc)
        _, tpr_vgg_baseline, auc_vgg_baseline = extract_roc_curve_params_across_folds_flattened(output_dir_vgg_baseline)
        _, tpr_vgg_transfer_learning, auc_vgg_transfer_learning = extract_roc_curve_params_across_folds_flattened(output_dir_vgg_transfer_learning)
        _, tpr_seresnext_baseline, auc_seresnext_baseline = extract_roc_curve_params_across_folds_flattened(output_dir_seresnext_baseline)
        _, tpr_seresnext_transfer_learning, auc_seresnext_transfer_learning = extract_roc_curve_params_across_folds_flattened(output_dir_seresnext_transfer_learning)

        # BRATS inference
        _, tpr_vgg_baseline_brats, auc_vgg_baseline_brats = extract_roc_curve_params_across_folds_flattened(output_dir_vgg_baseline_brats_inference,
                                                                                                            brats_inference=True)
        _, tpr_vgg_transfer_learning_brats, auc_vgg_transfer_learning_brats = extract_roc_curve_params_across_folds_flattened(output_dir_vgg_transfer_learning_brats_inference,
                                                                                                                              brats_inference=True)
        _, tpr_seresnext_baseline_brats, auc_seresnext_baseline_brats = extract_roc_curve_params_across_folds_flattened(output_dir_seresnext_baseline_brats_inference,
                                                                                                                        brats_inference=True)
        _, tpr_seresnext_transfer_learning_brats, auc_seresnext_transfer_learning_brats = extract_roc_curve_params_across_folds_flattened(output_dir_seresnext_transfer_learning_brats_inference,
                                                                                                                                          brats_inference=True)

        plot_roc_curve_wo_std(tpr_vgg_baseline,
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
                              cv_folds)

        # extract params of PR curve (precision, recall, aupr)
        _, precision_vgg_baseline, aupr_vgg_baseline = extract_pr_curve_params_across_folds_flattened(output_dir_vgg_baseline)
        _, precision_vgg_transfer_learning, aupr_vgg_transfer_learning = extract_pr_curve_params_across_folds_flattened(output_dir_vgg_transfer_learning)
        _, precision_seresnext_baseline, aupr_seresnext_baseline = extract_pr_curve_params_across_folds_flattened(output_dir_seresnext_baseline)
        _, precision_seresnext_transfer_learning, aupr_seresnext_transfer_learning = extract_pr_curve_params_across_folds_flattened(output_dir_seresnext_transfer_learning)

        # BRATS inference
        _, precision_vgg_baseline_brats, aupr_vgg_baseline_brats = extract_pr_curve_params_across_folds_flattened(output_dir_vgg_baseline_brats_inference,
                                                                                                                  brats_inference=True)
        _, precision_vgg_transfer_learning_brats, aupr_vgg_transfer_learning_brats = extract_pr_curve_params_across_folds_flattened(output_dir_vgg_transfer_learning_brats_inference,
                                                                                                                                    brats_inference=True)
        _, precision_seresnext_baseline_brats, aupr_seresnext_baseline_brats = extract_pr_curve_params_across_folds_flattened(output_dir_seresnext_baseline_brats_inference,
                                                                                                                              brats_inference=True)
        _, precision_seresnext_transfer_learning_brats, aupr_seresnext_transfer_learning_brats = extract_pr_curve_params_across_folds_flattened(output_dir_seresnext_transfer_learning_brats_inference,
                                                                                                                                                brats_inference=True)

        # since we only use the ground truth, we can use any model, cause the ground truth is always the same
        avg_no_skill_across_folds = find_average_no_skill_across_folds_pr_curve(output_dir_vgg_baseline)
        avg_no_skill_brats_inference = find_average_no_skill_across_folds_pr_curve(output_dir_vgg_baseline, brats_inference=True)

        plot_pr_curve_wo_std(precision_vgg_baseline,
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
                             avg_no_skill_brats_inference)


def main():
    # the code inside here is run only when THIS script is run, and not just imported
    config_dict = load_config_file()  # load input config file

    # input args
    cv_folds = config_dict['cv_folds']
    plot_with_standard_deviation = str2bool(config_dict['plot_with_standard_deviation'])

    # VGG baseline
    output_dir_vgg_baseline = config_dict['output_dir_vgg_baseline']
    legend_label_vgg_baseline = config_dict['legend_label_vgg_baseline']

    # VGG transfer learning
    output_dir_vgg_transfer_learning = config_dict['output_dir_vgg_transfer_learning']
    legend_label_vgg_transfer_learning = config_dict['legend_label_vgg_transfer_learning']

    # SEResNeXt baseline
    output_dir_seresnext_baseline = config_dict['output_dir_seresnext_baseline']
    legend_label_seresnext_baseline = config_dict['legend_label_seresnext_baseline']

    # SEResNeXt transfer learning
    output_dir_seresnext_transfer_learning = config_dict['output_dir_seresnext_transfer_learning']
    legend_label_seresnext_transfer_learning = config_dict['legend_label_seresnext_transfer_learning']

    # BRAT-TCIA inference
    # ------ VGG baseline
    output_dir_vgg_baseline_brats_inference = config_dict['output_dir_vgg_baseline_brats_inference']
    legend_label_vgg_baseline_brats_inference = config_dict['legend_label_vgg_baseline_brats_inference']
    # ------ VGG transfer learning
    output_dir_vgg_transfer_learning_brats_inference = config_dict['output_dir_vgg_transfer_learning_brats_inference']
    legend_label_vgg_transfer_learning_brats_inference = config_dict['legend_label_vgg_transfer_learning_brats_inference']
    # ------ SEResNeXt baseline
    output_dir_seresnext_baseline_brats_inference = config_dict['output_dir_seresnext_baseline_brats_inference']
    legend_label_seresnext_baseline_brats_inference = config_dict['legend_label_seresnext_baseline_brats_inference']
    # ------ SEResNeXt transfer learning
    output_dir_seresnext_transfer_learning_brats_inference = config_dict['output_dir_seresnext_transfer_learning_brats_inference']
    legend_label_seresnext_transfer_learning_brats_inference = config_dict['legend_label_seresnext_transfer_learning_brats_inference']

    plot_curves(cv_folds,
                output_dir_vgg_baseline,
                legend_label_vgg_baseline,
                output_dir_vgg_transfer_learning,
                legend_label_vgg_transfer_learning,
                output_dir_seresnext_baseline,
                legend_label_seresnext_baseline,
                output_dir_seresnext_transfer_learning,
                legend_label_seresnext_transfer_learning,
                output_dir_vgg_baseline_brats_inference,
                legend_label_vgg_baseline_brats_inference,
                output_dir_vgg_transfer_learning_brats_inference,
                legend_label_vgg_transfer_learning_brats_inference,
                output_dir_seresnext_baseline_brats_inference,
                legend_label_seresnext_baseline_brats_inference,
                output_dir_seresnext_transfer_learning_brats_inference,
                legend_label_seresnext_transfer_learning_brats_inference,
                plot_with_standard_deviation)


if __name__ == '__main__':
    main()
