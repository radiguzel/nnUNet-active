import argparse
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.run.default_configuration import get_default_configuration
from nnunet.paths import default_plans_identifier
from nnunet.run.load_pretrained_weights import load_pretrained_weights
from nnunet.training.cascade_stuff.predict_next_stage import predict_next_stage
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.training.network_training.nnUNetTrainerCascadeFullRes import (
    nnUNetTrainerCascadeFullRes,
)
from nnunet.training.network_training.nnUNetTrainerV2_CascadeFullRes import (
    nnUNetTrainerV2CascadeFullRes,
)
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name


def run_training_func(
    network,
    network_trainer,
    task,
    fold,
    val=False,
    c=True,
    save_l=False,
    last_layer_directory=None,
):
    """
    :param network: (e.g. 3d_fullres)
    :param network_trainer: (e.g. nnUNetTrainerV2)
    :param task: Task id (e.g. 501)
    :param val: If True, validation is run only.
    :param c: If True, continue training from the latest checkpoint.
    :param save_l: save the last layer of the encoder
    :param last_layer_directory: The directory where the last layer of the encoder should be saved.
    :return trainer: trainer object is returned to be used for checkpoints.
    """

    disable_saving = False
    pretrained_weights = None
    npz = True
    val_disable_overwrite = True
    disable_next_stage_pred = False

    validation_only = val  # args.validation_only
    plans_identifier = default_plans_identifier
    find_lr = False
    disable_postprocessing_on_folds = False
    use_compressed_data = False
    decompress_data = not use_compressed_data
    deterministic = False
    valbest = False
    fp32 = False
    run_mixed_precision = not fp32
    val_folder = "validation_raw"

    if not task.startswith("Task"):
        task_id = int(task)
        task = convert_id_to_task_name(task_id)

    if fold == "all":
        pass
    else:
        fold = int(fold)

    (
        plans_file,
        output_folder_name,
        dataset_directory,
        batch_dice,
        stage,
        trainer_class,
    ) = get_default_configuration(network, task, network_trainer, plans_identifier)

    if trainer_class is None:
        raise RuntimeError(
            "Could not find trainer class in nnunet.training.network_training"
        )

    if network == "3d_cascade_fullres":
        assert issubclass(
            trainer_class, (nnUNetTrainerCascadeFullRes, nnUNetTrainerV2CascadeFullRes)
        ), (
            "If running 3d_cascade_fullres then your "
            "trainer class must be derived from "
            "nnUNetTrainerCascadeFullRes"
        )
    else:
        assert issubclass(
            trainer_class, nnUNetTrainer
        ), "network_trainer was found but is not derived from nnUNetTrainer"

    trainer = trainer_class(
        plans_file,
        fold,
        output_folder=output_folder_name,
        dataset_directory=dataset_directory,
        batch_dice=batch_dice,
        stage=stage,
        unpack_data=decompress_data,
        deterministic=deterministic,
        fp16=run_mixed_precision,
    )
    if disable_saving:
        # whether or not to save the final checkpoint
        trainer.save_final_checkpoint = False
        # whether or not to save the best checkpoint according to
        trainer.save_best_checkpoint = False
        # self.best_val_eval_criterion_MA
        # whether or not to save checkpoint_latest. We need that in case
        trainer.save_intermediate_checkpoints = True
        # the training chashes
        # if false it will not store/overwrite _latest but separate files each
        trainer.save_latest_only = True

    trainer.initialize(not validation_only)
    if find_lr:
        trainer.find_lr()
    else:
        if not validation_only:
            if c:
                # -c was set, continue a previous training and ignore pretrained weights
                trainer.load_latest_checkpoint()
                trainer.max_num_epochs += trainer.epoch
                print(
                    f"*trainer epoch:{trainer.epoch}\ntrainer max num epoch:{trainer.max_num_epochs}*"
                )
            elif (not c) and (pretrained_weights is not None):
                # we start a new training. If pretrained_weights are set, use them
                load_pretrained_weights(trainer.network, pretrained_weights)
            else:
                # new training without pretrained weights, do nothing
                pass
            trainer.run_training()
        else:
            if valbest:
                trainer.load_best_checkpoint(train=False)
            else:
                trainer.load_final_checkpoint(train=False)
        trainer.network.eval()

        if save_l:
            assert last_layer_directory != None
            if not isdir(last_layer_directory):
                os.mkdir(last_layer_directory)
            # predict validation
            trainer.network.save_last_encoder = True
            trainer.network.last_layer_directory = last_layer_directory
            trainer.validate(
                do_mirroring=False,
                use_sliding_window=False,
                save_softmax=npz,
                validation_folder_name=val_folder,
                run_postprocessing_on_folds=not disable_postprocessing_on_folds,
                overwrite=val_disable_overwrite,
            )
            trainer.network.save_last_encoder = False
            trainer.network.last_layer_directory = None
        else:
            # predict validation
            trainer.validate(
                save_softmax=npz,
                validation_folder_name=val_folder,
                run_postprocessing_on_folds=not disable_postprocessing_on_folds,
                overwrite=val_disable_overwrite,
            )

        if network == "3d_lowres" and not disable_next_stage_pred:
            print("predicting segmentations for the next stage of the cascade")
            predict_next_stage(
                trainer,
                join(
                    dataset_directory, trainer.plans["data_identifier"] + "_stage%d" % 1
                ),
            )
    return trainer
