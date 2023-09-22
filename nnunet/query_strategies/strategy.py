import numpy as np
from collections import OrderedDict
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.paths import (
    network_training_output_dir,
    preprocessing_output_dir,
    default_plans_identifier,
)
import shutil
from nnunet.run.run_training_func import run_training_func
from sklearn.metrics.pairwise import cosine_similarity
import pathlib
from . import similarity
import math


class Strategy:
    def __init__(
        self,
        keys,
        splits_file,
        splits_file_orig,
        validation_summary_dir,
        validation_raw_dir,
        last_layer_directory,
        query_step=5,
        shortlist_size=30,
        use_similarity=True,
        is_dynamic_shortlist=False,
    ):
        """
        Strategy class is used to keep track of the acive learning steps and is a superclass of the sampling strategies. It provides with methods that is same across all the sampling strategies.
        :param keys: keys refer to training samples' ids.
        :param splits_file_orig: The original splits file that shows the training and validation set keys
        :param splits_file: The splits file that shows the training and validation set keys. This will be used in active learning
        :param validation_summary_dir: Inference values (output) are saved to this directory. (For the validation set)
        :param validation_raw_dir: Inference values (output) are saved to this directory. (For all the unlabelled samples)
        :param last_layer_directory: Values on the abstraction layer (Last layer of the encoder part) are saved to this directory.
        :param query_step: The number of samples that will be added to labelled dataset. In each active learning iteration, the
        :param use_similarity: Set true if similarity will be used to choose the samples to be queried in two stages.
        :param shortlist_size: Initial shortlist size.
        :param is_dynamic_shortlist: Set True if the shortlist size will reduce at each iteration.
        :return:

        What needs to be overrided:
        - __init__
        - query
        """

        self.keys = keys
        self.splits_file = splits_file
        self.splits_file_orig = splits_file_orig
        self.validation_summary_dir = validation_summary_dir
        self.validation_raw_dir = validation_raw_dir
        self.last_layer_directory = last_layer_directory
        self.query_step = query_step
        self.shortlist_size = shortlist_size
        self.use_similarity = use_similarity
        self.is_dynamic_shortlist = is_dynamic_shortlist

        self.shortlist_keys_arr = []
        self.splits = None
        self.sel_keys = None
        self.unsel_keys = None
        self.len_sel_keys = None
        self.len_unsel_keys = None
        self.was_initialized = False
        self.is_content_s = True

    def query(self):
        # this should be overrided in the subclass. This is used to choose the best samples among the unlabelled dataset.
        pass

    def initialize(self, proportion_training=0.8):
        """
        Initialization of the strategy object, original training and validation set is configured randomly with a seed from the keys so that the order of the keys are deterministic. Two splits file are generated. The first one is the original splits file for training and validation set, (when proportion_training=0.8) the size of the training set becomes 208 and the size of the validation set becomes 52. The second one is the splits file for active learning. The training set is initialized with the size of query step (as default, 5) and the validation set is the same with original one (52). So far, fold 0 was only used. As a second fold in the splits file, there is no training set and the validation set is the unlaballed dataset, (initially 203). Fold 1 is used to run inference and save the output values to estimate uncertainty and also save the last layer of the encoder for find the similarity between the images.
        :param proportion_training: The proportion of the keys used for training set in original splits file.
        :return:
        """
        rnd = np.random.RandomState(seed=12345 + 2)
        idx_tr = rnd.choice(
            len(self.keys), int(len(self.keys) * proportion_training), replace=False
        )
        idx_val = [i for i in range(len(self.keys)) if i not in idx_tr]
        tr_keys = [self.keys[i] for i in idx_tr]
        val_keys = [self.keys[i] for i in idx_val]

        splits = []
        splits.append(OrderedDict())

        splits[-1]["train"] = tr_keys
        splits[-1]["val"] = val_keys
        # splits_file_orig will be saved here to create splits_file
        save_pickle(splits, self.splits_file_orig)

        splits_o = load_pickle(self.splits_file_orig)

        rnd = np.random.RandomState(seed=12345 + 2)
        keys = np.sort(list(splits_o[0]["train"]))
        idx_sel = rnd.choice(len(keys), self.query_step, replace=False)
        sel_keys = [keys[i] for i in idx_sel]
        splits = splits_o
        splits[0]["train"] = sel_keys
        splits = [splits.pop(0)]

        idx_unsel = [i for i in range(len(keys)) if i not in idx_sel]
        unsel_keys = [keys[i] for i in idx_unsel]
        splits.append(OrderedDict())
        splits[1]["train"] = []
        splits[1]["val"] = unsel_keys
        write_pickle(splits, self.splits_file)
        self.splits = splits
        self.len_sel_keys = len(sel_keys)
        self.unsel_keys = unsel_keys
        self.len_unsel_keys = len(unsel_keys)
        self.was_initialized = True
        print("Strategy is initialized\n")

    def update(self):
        """
        Update the splits_file and delete the files inside validation_raw_dir and last_layer_directory. These files are used to choose the samples. They are not needed anymore, so they can be deleted safely.
        :return:
        """
        self.update_pickle()
        self.delete_dir(self.validation_raw_dir)
        self.delete_dir(self.last_layer_directory)

    def update_pickle(self):
        """
        Update splits_file so that the selected keys are added to the training set in fold 0 and the validation set in fold 1 is updated with the updated unselected keys (unlabelled dataset).
        :return:
        """
        self.unsel_keys = [e for e in self.unsel_keys if e not in self.sel_keys]
        self.splits[0]["train"].extend(self.sel_keys)
        self.sel_keys = self.splits[0]["train"]
        self.len_sel_keys = len(self.sel_keys)
        self.len_unsel_keys = len(self.unsel_keys)
        self.splits[1]["val"] = self.unsel_keys
        write_pickle(self.splits, self.splits_file)

    def delete_dir(self, target_dir):
        """
        Delete all the files inside the target directory (target_dir).
        :param target_dir: Target directory to be deleted.
        :return:
        """
        with os.scandir(target_dir) as entries:
            for entry in entries:
                if entry.is_dir() and not entry.is_symlink():
                    shutil.rmtree(entry.path)
                else:
                    os.remove(entry.path)
        print(f"{target_dir} folder is deleted")

    def read_last_layer(self):
        """
        Last layer of the encoder is saved after inference of the each unlabelled samples. If is_content_s is set to False, the mean of the last layer is saved.
        :return last_layer: The values of the last layer of the encoder.
        :return last_layer_keys: The corresponding keys are saved to keep track of the value and key pairs.
        """
        last_layer = []
        last_layer_keys = []
        file_list = list(pathlib.Path(self.last_layer_directory).glob("*.npy"))
        for idx, filename in enumerate(file_list):
            last_layer.append(np.load(filename))
            last_layer_keys.append(os.path.basename(filename)[:-4])
        if self.is_content_s:
            min_shape = similarity.get_min_shape(last_layer)
            last_layer = similarity.crop_S(last_layer, min_shape)
        else:
            last_layer = similarity.mean_S(last_layer)
        return last_layer, last_layer_keys

    def select_keys(self, S_c, S_u):
        """
        Select the samples S_a out of S_c that represents S_u. S_a is a subset of S_c, that is a subset of S_u.
        :param S_u: Unlabelled dataset
        :param S_c: Shortlist samples
        :return S_a_idxs: Samples to be queried
        """
        S_a = []
        S_a_idxs = []
        for _ in range(self.query_step):
            max_sim = -1
            max_arg = -1
            for idx, sc in enumerate(S_c):
                S_a.append(sc)
                group_similarity = similarity.group_to_group_similarity(
                    S_a, S_u, self.is_content_s
                )
                print(group_similarity)
                if group_similarity > max_sim:
                    max_sim = group_similarity
                    max_arg = idx
                    print(max_sim)
                del S_a[-1]
            S_a.append(S_c[max_arg])
            S_a_idxs.append(max_arg)
            print(max_arg)
        print(S_a_idxs)
        return S_a_idxs

    def select_among_shortlist_keys(self):
        """
        Among shortlist samples, the final samples to be queried is selected. If use_similarity is set to False, the similarity functions are not used and the first query_step (5) samples are returned. If is_dynamic_shortlist is set to True, shortlist_size is decreased by 20 percent. The aim is to find a set of samples, S_a, out of S_c such that represents S_u as much as possible. S_a is a subset of S_c, that is a subset of S_u.
        :return S_a_keys: The keys of the samples to be queried, It has the size of query_step.
        """
        if self.is_dynamic_shortlist:
            self.shortlist_size = max(
                self.query_step, math.ceil(self.shortlist_size * 0.8)
            )
        if not self.use_similarity:
            return self.shortlist_keys[: self.query_step]
        S_u, S_u_keys = self.read_last_layer()
        S_c = []
        S_c_idxs = []
        S_c_keys = []
        for idx, key in enumerate(self.shortlist_keys):
            S_c_idxs.append(S_u_keys.index(key))
            S_c_keys.append(key)
            S_c.append(S_u[S_c_idxs[idx]])
        # S_c_idxs = np.searchsorted(self.last_layer_idxs, self.shortlist_keys)

        S_a_idxs = self.select_keys(S_c, S_u)
        S_a_keys = list(map(S_c_keys.__getitem__, S_a_idxs))
        return S_a_keys
