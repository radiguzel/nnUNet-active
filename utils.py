import os
from tabnanny import check
import numpy as np
import pickle
import nibabel as nib
from time import time, sleep
from datetime import datetime
import yaml
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.paths import (
    network_training_output_dir,
    preprocessing_output_dir,
    default_plans_identifier,
)
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name


def dump_yaml(
    task,
    strategy_name,
    strategy,
    network_trainer,
    trainer,
    trainer_epoch_counter,
    val_acc,
    results_dir,
):
    """
    Dump the results to a yaml file.
    :param task: Task id, (e.g. 501)
    :param strategy_name: Stragegy name (e.g. entropySampling)
    :param strategy: Strategy object, it holds the selected keys, booleans for if similarity is used or not, etc.
    :param network_trainer: 
    :param trainer: Trainer object, it holds the batch size, maximum number of epochs, etc.
    :param trainer_epoch_counter: At the end of each iteration's training, the epoch number is saved. (e.g. 99, 199,...,999)
    :param val_acc: Validation accuracy of each iteration is saved.
    :param results_dir: The directory where the yaml file should be saved in.
    :return:
    """

    timestamp = time()
    dt_object = datetime.fromtimestamp(timestamp)
    timestamp = datetime.now()
    config_dict = {}
    config_dict["task"] = task
    config_dict["strategyName"] = strategy_name
    config_dict["validationAccuracy"] = list(map(float, val_acc))
    config_dict["selectedKeys"] = str(strategy.sel_keys)
    config_dict["strategyIsContentS"] = strategy.is_content_s
    config_dict["strategyUseSimilarity"] = strategy.use_similarity
    # config_dict['unselectedKeys'] = str(strategy.unsel_keys)
    config_dict["lengthSelectedKeys"] = strategy.len_sel_keys
    config_dict["lengthUnselectedKeys"] = strategy.len_unsel_keys
    config_dict["splitFiles"] = strategy.splits_file
    config_dict["shortlistKeysArray"] = strategy.shortlist_keys_arr
    config_dict["validationRawDir"] = strategy.validation_raw_dir
    config_dict["queryStep"] = strategy.query_step
    config_dict["trainerName"] = network_trainer
    config_dict["trainerEpochCounter"] = trainer_epoch_counter
    config_dict["trainerAllTrLosses"] = list(map(float, trainer.all_tr_losses))
    config_dict["trainerAllValEvalMetrics"] = list(
        map(float, trainer.all_val_eval_metrics)
    )
    config_dict["trainerAllValLosses"] = list(map(float, trainer.all_val_losses))
    config_dict["trainerBatchSize"] = trainer.batch_size
    config_dict["trainerBestMATrLossForPatience"] = float(
        trainer.best_MA_tr_loss_for_patience
    )
    config_dict[
        "trainerBestEpochBasedOnMATrLoss"
    ] = trainer.best_epoch_based_on_MA_tr_loss
    config_dict["trainerBestValEvalCriterionMA"] = float(
        trainer.best_val_eval_criterion_MA
    )
    config_dict["trainerClasses"] = trainer.classes
    config_dict["trainerInitialLearningRate"] = trainer.initial_lr
    config_dict["trainerInitialLearningRateSchedulerEps"] = trainer.lr_scheduler_eps
    config_dict[
        "trainerInitialLearningRateSchedulerPatience"
    ] = trainer.lr_scheduler_patience
    config_dict["trainerInitialLearningRateThreshold"] = trainer.lr_threshold
    config_dict["trainerInitialLearningRate"] = trainer.initial_lr
    config_dict["trainerMaxNumEpochs"] = trainer.max_num_epochs
    config_dict["trainerNumberBatchesPerEpoch"] = trainer.num_batches_per_epoch
    config_dict["trainerNumberClasses"] = trainer.num_classes
    config_dict["trainerNumberInputChannels"] = trainer.num_input_channels
    config_dict["trainerNumberValBatchesPerEpoch"] = trainer.num_val_batches_per_epoch
    config_dict[
        "trainerOversampleForegroundPercent"
    ] = trainer.oversample_foreground_percent
    config_dict["trainerSaveEvery"] = trainer.save_every
    config_dict["trainerTrainLossMA"] = float(trainer.train_loss_MA)
    config_dict["trainerTrainLossMAAlpha"] = trainer.train_loss_MA_alpha
    config_dict["trainerTrainLossMAEps"] = trainer.train_loss_MA_eps
    config_dict["trainerValEvalCriterionMA"] = float(trainer.val_eval_criterion_MA)
    config_dict["trainerValEvalCriterionAlpha"] = trainer.val_eval_criterion_alpha
    config_dict["trainerWeightDecay"] = trainer.weight_decay
    config_dict["trainerPatience"] = trainer.patience
    config_dict["time"] = timestamp

    config_file = os.path.join(
        results_dir,
        "config_%s_%d_%d_%d_%02.0d_%02.0d.yaml"
        % (
            strategy_name,
            timestamp.year,
            timestamp.month,
            timestamp.day,
            timestamp.hour,
            timestamp.minute,
        ),
    )
    with open(config_file, "w") as f:
        data = yaml.dump(config_dict, f)


def save_active_checkpoint(
    fname, i, task, strategy_name, strategy, val_acc, trainer_epoch_counter
):
    """
    Save the checkpoint as a yaml file in case the code is stopped. It enables to start from where it left off.
    :param fname: The name of the folder to save the checkpoint
    :param i: Active run iteration number
    :param task: Task id, (e.g. 501)
    :param strategy_name: Stragegy name (e.g. entropySampling)
    :param strategy: Strategy object, it holds the selected keys, booleans for if similarity is used or not, etc.
    :param val_acc: Validation accuracy of each iteration until iteration i is saved.
    :param trainer_epoch_counter: At the end of each iteration's training, the epoch number is saved. (e.g. 99, 199,...,999)
    :return:
    """
    timestamp = time()
    dt_object = datetime.fromtimestamp(timestamp)
    timestamp = datetime.now()

    checkpoint_dict = {}
    checkpoint_dict["task"] = task
    checkpoint_dict["strategyName"] = strategy_name
    checkpoint_dict["strategy"] = strategy
    checkpoint_dict["ActiveRunNumber"] = i
    checkpoint_dict["time"] = timestamp
    checkpoint_dict["validationAccuracy"] = val_acc
    checkpoint_dict["trainerEpochCounter"] = trainer_epoch_counter
    print(
        f"active checkpoint is saved: iteration number:{i}, task:{task}, strategy name:{strategy_name}"
    )
    checkpoint_file = os.path.join(fname, "latest_active_checkpoint.yaml")
    with open(checkpoint_file, "w") as f:
        data = yaml.dump(checkpoint_dict, f)
    return


def load_active_checkpoint(fname, task, strategy_name):
    """
    Load the active checkpoint.
    :param fname: The name of the folder to load the checkpoint.
    :param task: Task id (e.g. 501)
    :param strategy_name: Strategy name (e.g. entropySampling)
    :return: the latest iteration number in active learning, validation accuracy so far, trainer epochs and the strategy object, in order. If the first item is 0 it means there is no checkpoint found.
    """
    if isfile(join(fname, "latest_active_checkpoint.yaml")):
        with open(join(fname, "latest_active_checkpoint.yaml"), "r") as stream:
            try:
                checkpoint_dict = yaml.load(stream, Loader=yaml.Loader)
            except yaml.YAMLError as exc:
                print(exc)
        if task != checkpoint_dict["task"]:
            raise RuntimeError("Checkpoint task id is not the same!")
        if strategy_name != checkpoint_dict["strategyName"]:
            raise RuntimeError("Checkpoint strategy name is not the same!")
        print(
            "active checkpoint is loaded. iteration number:%s, task:%s, strate name: %s"
            % (
                checkpoint_dict["ActiveRunNumber"],
                checkpoint_dict["task"],
                checkpoint_dict["strategyName"],
            )
        )
        return (
            checkpoint_dict["ActiveRunNumber"] + 1,
            checkpoint_dict["validationAccuracy"],
            checkpoint_dict["trainerEpochCounter"],
            checkpoint_dict["strategy"],
        )
    else:
        print("Active checkpoint is not found")
    return 0, [], [], []  # this means there is no checkpoint found


def delete_active_checkpoint(fname):
    """
    Delete the active checkpoint. This function is used before saving the latest checkpoint. A safe measure.
    :return:
    """
    if isfile(join(fname, "latest_active_checkpoint.yaml")):
        os.remove(join(fname, "latest_active_checkpoint.yaml"))
        print("active checpoint is deleted successfully!")
    return
