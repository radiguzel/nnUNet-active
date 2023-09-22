import nnunet
from nnunet.paths import network_training_output_dir, preprocessing_output_dir, default_plans_identifier
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name
from nnunet.training.model_restore import recursive_find_python_class

def get_active_configuration(strategy_name, network, task, network_trainer, plans_identifier,
                            search_in=(nnunet.__path__[0], "query_strategies"),
                            base_module='nnunet.query_strategies'):
    """
    Configure the validation directories and strategy class.
    :param strategy_name: Stragegy name (e.g. entropySampling)
    :param network: Network (e.g. 3d_fullres)
    :param task: Task id, (e.g. 501)
    :param network_trainer: Network Trainer (e.g. nnUNetTrainerV2)
    :param plans_identifier: Plans identifier (e.g. nnUNetPlansv2.1)
    :returnvalidation_summary_dir: Inference values (output) are saved to this directory. (For the validation set)
    :return validation_raw_dir: Inference values (output) are saved to this directory. (For all the unlabelled samples)
    :return strategy_class: Strategy class
    """
    if not task.startswith("Task"):
        task_id = int(task)
        task_fullname = convert_id_to_task_name(task_id)
    
    validation_summary_dir = join(network_training_output_dir, network, task_fullname, network_trainer + "__" + plans_identifier, 'fold_0', 'validation_raw')
    validation_raw_dir = join(network_training_output_dir, network, task_fullname, network_trainer + "__" + plans_identifier, 'fold_1', 'validation_raw')
    
    
    strategy_class = recursive_find_python_class([join(*search_in)], strategy_name,
                                                current_module=base_module)
    return validation_summary_dir, validation_raw_dir, strategy_class