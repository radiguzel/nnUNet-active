from nnunet.run.run_training_func import run_training_func
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.paths import (
    network_training_output_dir,
    preprocessing_output_dir,
    default_plans_identifier,
)
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name
import numpy as np
import os
from collections import OrderedDict
import utils
from nnunet.training.dataloading.dataset_loading import (
    load_dataset,
    get_case_identifiers,
)
from nnunet.query_strategies.active_configuration import get_active_configuration
import argparse

# some constants to save the results
splits_filename = "splits_final.pkl"
splits_filename_orig = "splits_final_orig.pkl"
results_dir = "results_directory"


def main():
    """
    Run deep active learning on a task with a given query strategy. Task id and query strategy should be given as arguments e.g. "run_active 501 entropySampling".
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("task", help="task id")
    parser.add_argument("strategy", help="query strategy")
    args = parser.parse_args()
    task = args.task
    strategy_name = args.strategy
    network = "3d_fullres"
    network_trainer = "nnUNetTrainerV2"
    plans_identifier = "nnUNetPlansv2.1"
    fold = "0"
    print(f"Projects/nnUNet,task:{task}, strategy:{strategy_name}\n")
    (
        validation_summary_dir,
        validation_raw_dir,
        strategy_class,
    ) = get_active_configuration(
        strategy_name, network, task, network_trainer, plans_identifier
    )

    if strategy_class is None:
        raise RuntimeError("Could not find stragey class in nnunet.query_strategies")

    if not task.startswith("Task"):
        task_id = int(task)
        task_n = convert_id_to_task_name(task_id)
    dataset_directory = join(preprocessing_output_dir, task_n)
    splits_file_orig = join(dataset_directory, splits_filename_orig)
    splits_file = join(dataset_directory, splits_filename)
    last_layer_directory = join(dataset_directory, "last_layer")

    folder_with_preprocessed_data = join(
        dataset_directory, "nnUNetData_plans_v2.1_stage0"
    )

    case_identifiers = get_case_identifiers(folder_with_preprocessed_data)
    keys = np.sort(case_identifiers)
    strategy = strategy_class(
        keys,
        splits_file,
        splits_file_orig,
        validation_summary_dir,
        validation_raw_dir,
        last_layer_directory,
    )

    val = False
    fold = "0"
    c = False
    val_acc = []
    trainer_epoch_counter = []
    # os.system("nnUNet_plan_and_preprocess -t 501")
    start_n = 0
    (
        start_n,
        val_acc,
        trainer_epoch_counter,
        strategy_loaded,
    ) = utils.load_active_checkpoint(dataset_directory, task, strategy_name)
    if start_n > 0:
        c = True
        strategy = strategy_loaded

    if not strategy.was_initialized:
        strategy.initialize()
    strategy.use_similarity = False  # use similarity or not
    strategy.is_content_s = False  # Use the content distance or not
    strategy.is_dynamic_shortlist = False  # shortlist size is dynamic or not
    N = 10  # 2
    for i in range(start_n, N):
        print("*train-run: %d *\n*# of train els: %d*" % (i + 1, strategy.len_sel_keys))
        # Train the model on the labelled dataset
        trainer_o = run_training_func(network, network_trainer, task, fold, val, c)
        val = True
        fold = "1"
        c = True
        print("*val-run: %d *\n*# of val elem: %d*" % (i + 1, strategy.len_unsel_keys))
        # Run inference
        _ = run_training_func(network, network_trainer, task, fold, val, c)
        # Run inference to save the output and the last layer of the encoder.
        _ = run_training_func(
            network,
            network_trainer,
            task,
            fold,
            val,
            c,
            save_l=True,
            last_layer_directory=last_layer_directory,
        )
        strategy.query()  # Select the next samples from the unlabelled dataset
        val = False
        fold = "0"

        # Save checkpoint
        trainer_epoch_counter.append(trainer_o.epoch)
        validation_result_raw = load_json(join(validation_summary_dir, "summary.json"))[
            "results"
        ]
        validation_size = len(validation_result_raw["all"])
        print(f"validation size: {validation_size}")
        classes = validation_result_raw["mean"].keys()
        classes_dice = []
        for cl in classes:
            classes_dice.append(validation_result_raw["mean"][cl]["Dice"])
        val_acc.append(np.mean(classes_dice))

        utils.save_active_checkpoint(
            dataset_directory,
            i,
            task,
            strategy_name,
            strategy,
            val_acc,
            trainer_epoch_counter,
        )  # Save active checkpoint at each iteration.
    utils.delete_active_checkpoint(
        dataset_directory
    )  # After the active learning is finished, delete the active checpoint!
    print(val_acc)
    utils.dump_yaml(
        task,
        strategy_name,
        strategy,
        network_trainer,
        trainer_o,
        trainer_epoch_counter,
        val_acc,
        results_dir,
    )


if __name__ == "__main__":
    main()
