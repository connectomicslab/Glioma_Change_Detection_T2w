from dataset_creation.utils_dataset_creation import load_config_file
from show_results.utils_show_results import permutation_tests
import matplotlib.pyplot as plt


""" Code inspired by the great tutorial https://www.jwilber.me/permutationtest/ """


__author__ = "Tommaso Di Noto"
__version__ = "0.0.1"
__email__ = "tommydino@hotmail.it"
__status__ = "Prototype"


def experiments_transfer_learning_vs_baseline(output_dir_vgg_baseline,
                                              legend_label_vgg_baseline,
                                              output_dir_vgg_transfer_learning,
                                              legend_label_vgg_transfer_learning,
                                              output_dir_seresnext_baseline,
                                              legend_label_seresnext_baseline,
                                              output_dir_seresnext_transfer_learning,
                                              legend_label_seresnext_transfer_learning,
                                              nb_permutations,
                                              test_statistic):
    # VGG baseline vs. VGG transfer learning
    permutation_tests(output_dir_vgg_baseline,
                      legend_label_vgg_baseline,
                      output_dir_vgg_transfer_learning,
                      legend_label_vgg_transfer_learning,
                      nb_permutations,
                      test_statistic,
                      figure_nb=1)

    # SERESNEXT baseline vs. SERESNEXT transfer learning
    permutation_tests(output_dir_seresnext_baseline,
                      legend_label_seresnext_baseline,
                      output_dir_seresnext_transfer_learning,
                      legend_label_seresnext_transfer_learning,
                      nb_permutations,
                      test_statistic,
                      figure_nb=2)


def experiments_vgg_vs_seresnext(output_dir_vgg_baseline,
                                 legend_label_vgg_baseline,
                                 output_dir_vgg_transfer_learning,
                                 legend_label_vgg_transfer_learning,
                                 output_dir_seresnext_baseline,
                                 legend_label_seresnext_baseline,
                                 output_dir_seresnext_transfer_learning,
                                 legend_label_seresnext_transfer_learning,
                                 nb_permutations,
                                 test_statistic):

    # VGG baseline vs. SERESNEXT baseline
    permutation_tests(output_dir_vgg_baseline,
                           legend_label_vgg_baseline,
                           output_dir_seresnext_baseline,
                           legend_label_seresnext_baseline,
                           nb_permutations,
                           test_statistic,
                           figure_nb=3)

    # VGG transfer learning vs. SERESNEXT transfer learning
    permutation_tests(output_dir_vgg_transfer_learning,
                      legend_label_vgg_transfer_learning,
                      output_dir_seresnext_transfer_learning,
                      legend_label_seresnext_transfer_learning,
                      nb_permutations,
                      test_statistic,
                      figure_nb=4)

    plt.show()


def experiments_brats_inference(output_dir_vgg_baseline_brats_inference,
                                legend_label_vgg_baseline_brats_inference,
                                output_dir_vgg_transfer_learning_brats_inference,
                                legend_label_vgg_transfer_learning_brats_inference,
                                output_dir_seresnext_baseline_brats_inference,
                                legend_label_seresnext_baseline_brats_inference,
                                output_dir_seresnext_transfer_learning_brats_inference,
                                legend_label_seresnext_transfer_learning_brats_inference,
                                nb_permutations,
                                test_statistic):

    # SEResNeXt baseline vs. SEResNeXt TL
    permutation_tests(output_dir_seresnext_baseline_brats_inference,
                      legend_label_seresnext_baseline_brats_inference,
                      output_dir_seresnext_transfer_learning_brats_inference,
                      legend_label_seresnext_transfer_learning_brats_inference,
                      nb_permutations,
                      test_statistic,
                      figure_nb=5,
                      brats_inference=True)

    # SEResNeXt baseline vs. VGG baseline
    permutation_tests(output_dir_seresnext_baseline_brats_inference,
                      legend_label_seresnext_baseline_brats_inference,
                      output_dir_vgg_baseline_brats_inference,
                      legend_label_vgg_baseline_brats_inference,
                      nb_permutations,
                      test_statistic,
                      figure_nb=5,
                      brats_inference=True)


def run_permutation_tests(nb_permutations: int,
                          output_dir_vgg_baseline: str,
                          legend_label_vgg_baseline: str,
                          output_dir_vgg_transfer_learning: str,
                          legend_label_vgg_transfer_learning: str,
                          output_dir_seresnext_baseline: str,
                          legend_label_seresnext_baseline: str,
                          output_dir_seresnext_transfer_learning: str,
                          legend_label_seresnext_transfer_learning: str,
                          output_dir_vgg_baseline_brats_inference: str,
                          legend_label_vgg_baseline_brats_inference: str,
                          output_dir_vgg_transfer_learning_brats_inference: str,
                          legend_label_vgg_transfer_learning_brats_inference: str,
                          output_dir_seresnext_baseline_brats_inference: str,
                          legend_label_seresnext_baseline_brats_inference: str,
                          output_dir_seresnext_transfer_learning_brats_inference: str,
                          legend_label_seresnext_transfer_learning_brats_inference: str,
                          test_statistic: str) -> None:

    # TRANSFER LEARNING vs. BASELINE experiments
    experiments_transfer_learning_vs_baseline(output_dir_vgg_baseline,
                                              legend_label_vgg_baseline,
                                              output_dir_vgg_transfer_learning,
                                              legend_label_vgg_transfer_learning,
                                              output_dir_seresnext_baseline,
                                              legend_label_seresnext_baseline,
                                              output_dir_seresnext_transfer_learning,
                                              legend_label_seresnext_transfer_learning,
                                              nb_permutations,
                                              test_statistic)

    # VGG vs. SERESNEXT experiments
    experiments_vgg_vs_seresnext(output_dir_vgg_baseline,
                                 legend_label_vgg_baseline,
                                 output_dir_vgg_transfer_learning,
                                 legend_label_vgg_transfer_learning,
                                 output_dir_seresnext_baseline,
                                 legend_label_seresnext_baseline,
                                 output_dir_seresnext_transfer_learning,
                                 legend_label_seresnext_transfer_learning,
                                 nb_permutations,
                                 test_statistic)

    # # BRAT-TCIA INFERENCE
    experiments_brats_inference(output_dir_vgg_baseline_brats_inference,
                                legend_label_vgg_baseline_brats_inference,
                                output_dir_vgg_transfer_learning_brats_inference,
                                legend_label_vgg_transfer_learning_brats_inference,
                                output_dir_seresnext_baseline_brats_inference,
                                legend_label_seresnext_baseline_brats_inference,
                                output_dir_seresnext_transfer_learning_brats_inference,
                                legend_label_seresnext_transfer_learning_brats_inference,
                                nb_permutations,
                                test_statistic)


def main():
    # the code inside here is run only when THIS script is run, and not just imported
    config_dict = load_config_file()  # load input config file

    nb_permutations = config_dict['nb_permutations']
    test_statistic = config_dict['test_statistic']
    assert test_statistic in ("aupr", "auc"), "Only 'aupr' and 'auc' are accepted as test statistic; got {} instead".format(test_statistic)
    
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

    run_permutation_tests(nb_permutations,
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
                          test_statistic)


if __name__ == '__main__':
    main()
