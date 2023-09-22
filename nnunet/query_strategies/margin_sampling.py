from .strategy import Strategy
import pathlib
import numpy as np


class marginSampling(Strategy):
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
        marginSampling class is a subclass of Strategy. It is a margin-based sampling strategy.
        """
        super(marginSampling, self).__init__(
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
        Select the samples from the unlabelled dataset based on margin uncertainty estimation.
        :return:
        """
        file_list = list(pathlib.Path(self.validation_raw_dir).glob("*.npz"))
        U = np.zeros(len(file_list))
        idxs = []
        for idx, filename in enumerate(file_list):
            b = np.load(filename)
            probs = b["softmax"]
            probs_sorted = np.sort(probs, axis=0)
            probs_marg = probs_sorted[-1, :, :, :] - probs_sorted[-2, :, :, :]
            U[idx] = probs_marg.sum(dtype=np.float64) / probs_marg.size
            filename_npz = str(filename).split("/")[-1]
            idxs.append(filename_npz[:-4])
        idxs_sorted = [idxs[i] for i in np.argsort(U)]
        self.shortlist_keys = idxs_sorted[: self.shortlist_size]
        self.shortlist_keys_arr.append(self.shortlist_keys)
        self.sel_keys = self.select_among_shortlist_keys()
        self.update()
