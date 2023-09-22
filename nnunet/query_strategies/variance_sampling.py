from .strategy import Strategy
import pathlib
import numpy as np


class varianceSampling(Strategy):
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
        varianceSampling class is a subclass of Strategy. It is a variance-based sampling strategy.
        """
        super(varianceSampling, self).__init__(
            keys,
            splits_file,
            splits_file_orig,
            validation_summary_dir,
            validation_raw_dir,
            last_layer_directory,
            query_step,
            shortlist_size,
            use_similarity,
            is_dynamic_shortlist,
        )

    def query(self):
        """
        Select the samples from the unlabelled dataset based on variance uncertainty estimation.
        :return:
        """
        file_list = list(pathlib.Path(self.validation_raw_dir).glob("*.npz"))
        U = np.zeros(len(file_list))
        idxs = []
        for idx, filename in enumerate(file_list):
            b = np.load(filename)
            probs = b["softmax"]
            var_probs = np.var(probs, axis=0)
            U[idx] = var_probs.sum(dtype=np.float64) / var_probs.size
            filename_npz = str(filename).split("/")[-1]
            idxs.append(filename_npz[:-4])
        idxs_sorted = [idxs[i] for i in np.argsort(U)]
        self.shortlist_keys = idxs_sorted[: self.shortlist_size]
        self.shortlist_keys_arr.append(self.shortlist_keys)
        self.sel_keys = self.select_among_shortlist_keys()
        self.update()
