# Transfer learning with weak labels from radiology reports: application to glioma change detection

<p align="center">
  <img src="https://github.com/connectomicslab/Glioma_Change_Detection_T2w/blob/master/figures/t2w_difference_maps_four_cases_Jul_22_2022.png" />
</p>


This repository contains the code used for the [paper](https://arxiv.org/abs/2210.09698): 
"Transfer learning with weak labels from radiology reports: application to glioma change detection". Please cite the paper if you are using either our code or one of our labeled datasets (in-house or BRATS-2015).

## Data
You can find the data used for this project at [this Zenodo link](https://zenodo.org/record/7214605#.Y05Ug3VBzIE), and the corresponding longitudinal labels inside the dataframes located in the [extra_files](https://github.com/connectomicslab/Glioma_Change_Detection_T2w/tree/master/extra_files) directory. The column of interest for the labels is "t2_label". Remember that 0=stable difference map, while 1=unstable difference map.

## Installation & conda environment
1) Clone the repository
2) Create a conda environment using the `environment.yml` file located inside the `install` directory. If you are not familiar with conda environments, 
please check out the [official documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). Alternatively,
feel free to use your favorite [IDE](https://en.wikipedia.org/wiki/Integrated_development_environment) such as [PyCharm](https://www.jetbrains.com/pycharm/download/#section=linux) or [Visual Studio](https://visualstudio.microsoft.com/downloads/) to set up the environment.

## How to run python scripts
All scripts in this project are run with json configuration files and the [argparse](https://docs.python.org/3/library/argparse.html) package. If you're not familiar with this, please refer to [this guide](https://towardsdatascience.com/three-ways-to-parse-arguments-in-your-python-code-aba092e8ad73) (or similar ones).

## GPU needed!
Most scripts in this project require you to have a GPU, with at least 4 GB of VRAM.

# 1) Optuna hypertuning
In order to find the best hyperparameters to use in the final classification, we first run hypertuning via [Optuna](https://optuna.org/). This is the most time-consuming step. If you want to skip this step, you can already go to [Training/Inference AFTER Optuna](https://github.com/connectomicslab/Glioma_Change_Detection_T2w#2-traininginference-after-optuna). If instead you want to re-run hypertuning yourself, keep on reading here. As explained in the paper, we ran 4 experiments: VGG-Baseline, VGG-Transfer-Learning, SEResNeXt-Baseline, and SEResNeXt-Transfer-Learning. The scripts used to find the best hyperparameters with Optuna are `classify_T2w_diffmaps_optuna_baseline.py` (for VGG-Baseline and SEResNeXt-Baseline) and `classify_T2w_diffmaps_optuna_transfer_learning.py` (for VGG-Transfer-Learning and SEResNeXt-Transfer-Learning). Both scripts are located inside the
[training_and_inference](https://github.com/connectomicslab/Glioma_Change_Detection_T2w/tree/master/training_and_inference) directory. The configuration files needed to run the hyperparameter search with Optuna (both for the Baseline and the Transfer Learning experiments) can be found inside the [optuna_hypertuning](https://github.com/connectomicslab/Glioma_Change_Detection_T2w/tree/master/training_and_inference/config_files_train_and_inf/optuna_hypertuning) directory. Let's start by looking how to run the Baseline experiments.

### 1.1) Baseline
The Baseline experiments are the ones for which we only use the Human-Annotated Dataset (**HAD** in the paper). Since every cross-validation fold is computationally intensive, we run each fold independently. Thus, for each of the five cross-validation folds, we must run:

#### Optuna Baseline (cross-validation fold 1)
```python
classify_T2w_diffmaps_optuna_baseline.py --config config_files_train_and_inf/optuna_hypertuning/optuna_baseline_f1.json
```
#### Optuna Baseline (cross-validation fold 2)
```python
classify_T2w_diffmaps_optuna_baseline.py --config config_files_train_and_inf/optuna_hypertuning/optuna_baseline_f2.json
```
#### and so on for folds 3, 4, and 5 (each time, change the config file accordingly)

---
Please be aware that the 5 cross-validation folds must be run twice: once for the VGG model setting the argument `network` (inside the config file) to `"customVGG"`, and once for the SEResNeXt model, changing the argument `network` to `"seresnext50"`.

### 1.2) Transfer Learning (TL)
The TL experiments are the ones for which we use both the Human-Annotated Dataset (**HAD** in the paper) and the Weakly-Annotated Dataset (**WAD** in the paper). If we want to carry out the TL experiments, we must first run pre-training on **WAD**, because in some TL configurations (fine-tuning and feature extracting) we will load weights from the model trained on it. In turn, to perform pre-training, we must have first run the Baseline experiments because we want to exclude from pre-training all validation and test patients that were used for the Baseline. The configuration files to run pre-training are located inside the [pretrain_wad](https://github.com/connectomicslab/Glioma_Change_Detection_T2w/tree/master/training_and_inference/config_files_train_and_inf/pretrain_wad) directory. In the config file of the pre-training script, we will need to specify `path_to_output_baseline_dir` which is exactly the output directory where we saved results of the Baseline experiment. Once the Baseline experiments have been run, we can launch pre-training. If you want to skip pre-training, we also provide the weights of the four pre-trained models which are inside the directory [pretraining](https://github.com/connectomicslab/Glioma_Change_Detection_T2w/tree/master/extra_files/pretraining). In this case, you can skip directly to section [Transfer-Learning (cross-validation fold 1)](https://github.com/connectomicslab/Glioma_Change_Detection_T2w#transfer-learning-cross-validation-fold-1). If instead you want to re-run pretraining, keep on reading here. Two separate pre-trainings have to be run, one for the WAD dataset with the report classifier probability > 0.75 and one for the WAD dataset with the report classifier probability > 0.95 (see hyperparameter fraction_of_WAD in the paper for more details). For instance, if we want to run the pre-training
for WAD > 0.75 (again for each fold separately):

#### Pre-training WAD > 0.75 (cross-validation fold 1)
```python
classify_T2w_diffmaps_pretrain.py --config config_files_train_and_inf/pretrain_wad/pretrain_wad_above_0_75_f1.json
```
#### and so on for folds 2, 3, 4, and 5 (each time, change the config file accordingly)
As for the Baseline experiments, you must run the 5 pre-training folds for the VGG model (setting the argument `network` to `"customVGG"`) and for the SEResNeXt model (setting the argument `network` to `"seresnext50"`).

---

Similarly, to run pre-training for WAD > 0.95:
#### Pre-training WAD > 0.95 (cross-validation fold 1)
```python
classify_T2w_diffmaps_pretrain.py --config config_files_train_and_inf/pretrain_wad/pretrain_wad_above_0_95_f1.json
```
#### and so on for folds 2, 3, 4, and 5 (each time, change the config file accordingly)

Once pre-training has been run (both for WAD > 0.75, and WAD > 0.95), we can proceed with the actual Optuna hyperparameter search for the TL experiments.
After running the two pre-training scripts (WAD > 0.75, and WAD > 0.95), 4 directories will be generated: they will be named
 * `pretrain_aad_above_0_75_customVGG_le_1e-05_MM_DD_YYYY`
 * `pretrain_aad_above_0_75_seresnext50_le_1e-05_MM_DD_YYYY`
 * `pretrain_aad_above_0_95_customVGG_le_1e-05_MM_DD_YYYY`
 * `pretrain_aad_above_0_95_seresnext50_le_1e-05_MM_DD_YYYY`

In the config files of the optuna TL experiments, make sure to set the argument `date_of_pretrain` to the `MM_DD_YYYY` of the pre-training output directories.
**N.B.** the dates for the same network should be identical (e.g. we expect the same date for `pretrain_aad_above_0_75_customVGG` and
`pretrain_aad_above_0_95_customVGG`). Then, as for the other experiments, we must run each cross-validation fold separately with:
#### Transfer-Learning (cross-validation fold 1)
```python
classify_T2w_diffmaps_optuna_transfer_learning.py --config config_files_train_and_inf/optuna_hypertuning/optuna_transfer_learning_f1.json
```
#### and so on for folds 2, 3, 4, and 5 (each time, change the config file accordingly)

As for the Baseline, if we want to run the TL experiments for the VGG model, we have to set the argument `network` to `"customVGG"`, while if we want
to run the TL experiments for the SEResNeXt model, we have to set `network` to `"seresnext50"` inside the config files. Also, make sure to set the correct
`date_of_pretrain` argument. If using the already-computed pre-training directories (those in the [pretraining](https://github.com/connectomicslab/Glioma_Change_Detection_T2w/tree/master/extra_files/pretraining) directory), `date_of_pretrain` should be
set to `Apr_19_2022` for the VGG and to `May_23_2022` for the SEResNeXt.

# 2) Training/Inference AFTER Optuna
Once the best hyperparameters have been found with Optuna, we can finally run the last training/inference to obtain classification results.
If you used the optuna output directories already provided by us, you can find them inside [optuna_output_dirs](https://github.com/connectomicslab/Glioma_Change_Detection_T2w/tree/master/extra_files/optuna_output_dirs). If instead you ran hypertuning yourself, you should have some Optuna output directories saved in the `optuna_output_dir` path that you specified in the config files of the Optuna experiments. These output directories should be named:
* `optuna_baseline_customVGG_MM_DD_YYYY`
* `optuna_baseline_seresnext50_MM_DD_YYYY`
* `optuna_transfer_learning_customVGG_MM_DD_YYYY`
* `optuna_transfer_learning_seresnext50_MM_DD_YYYY`

To run the final inference, we can use the scripts `classify_T2w_diffmaps_after_optuna_baseline.py` (for VGG-Baseline and SEResNeXt-Baseline) and 
`classify_T2w_diffmaps_after_optuna_transfer_learning.py` (for VGG-TL and SEResNeXt-TL). These scripts are located inside the [training_and_inference](https://github.com/connectomicslab/Glioma_Change_Detection_T2w/tree/master/training_and_inference) directory. The configuration files for the final training/inference are located inside the [after_optuna](https://github.com/connectomicslab/Glioma_Change_Detection_T2w/tree/master/training_and_inference/config_files_train_and_inf/after_optuna) directory. Inside the configuration files, you must set the argument `optuna_output_dir` either corresponding to the one of the paths provided by us (those in [optuna_output_dirs](https://github.com/connectomicslab/Glioma_Change_Detection_T2w/tree/master/extra_files/optuna_output_dirs)) or corresponding to one of the folders that were generated when you ran the scripts above.
### 2.1) Baseline
For the Baseline experiment, we'd run (again for each cross-validation fold):
```python
classify_T2w_diffmaps_after_optuna_baseline.py --config config_files_train_and_inf/after_optuna/config_t2_difference_baseline_after_optuna_f1.json
```
#### and so on for folds 2, 3, 4, and 5 (each time, change the config file accordingly)
As usual, you must run the 5 folds both for the VGG model (setting the argument `network` to `"customVGG"`) and for the SEResNeXt model (setting the argument `network` to `"seresnext50"`)

### 2.2) Transfer-Learning (TL)
For the TL experiment, we'd run (again for each cross-validation fold):
```python
classify_T2w_diffmaps_after_optuna_transfer_learning.py --config config_files_train_and_inf/after_optuna/config_t2_difference_tl_after_optuna_f1.json
```
#### and so on for folds 2, 3, 4, and 5 (each time, change the config file accordingly)
As usual, you must run the 5 folds both for the VGG model (setting the argument `network` to `"customVGG"`) and for the SEResNeXt model (setting the argument `network` to `"seresnext50"`)

# 3) Inference on longitudinal BraTS-2015
To assess model generalizability, we also ran inference on some longitudinal subjects from the BraTS-2015 dataset. To replicate this experiment, you can use the script `inference_longitudinal_brats_tcia_subs.py` located inside the [training_and_inference](https://github.com/connectomicslab/Glioma_Change_Detection_T2w/tree/master/training_and_inference) directory. The config file to use is `config_inference_brats_tcia.json` that is located inside the [brats_inference](https://github.com/connectomicslab/Glioma_Change_Detection_T2w/tree/master/training_and_inference/config_files_train_and_inf/brats_inference) directory. To perform inference on the BraTS patients, you must run:
```python
inference_longitudinal_brats_tcia_subs.py --config config_files_train_and_inf/brats_inference/config_inference_brats_tcia.json
```
There are two important arguments that you should change inside `config_inference_brats_tcia.json` which are `network` and `experiment`, depending on which model you want to use for inferece: 
* if you want to use the VGG-Baseline model, set `network` to `"customVGG"` and `experiment` to `"baseline"`
* if you want to use the VGG-TL model, set `network` to `"customVGG"` and `experiment` to `"transfer_learning"`
* if you want to use the SEResNeXt-Baseline model, set `network` to `"seresnext50"` and `experiment` to `"baseline"`
* if you want to use the SEResNeXt-TL model, set `network` to `"seresnext50"` and `experiment` to `"transfer_learning"`
