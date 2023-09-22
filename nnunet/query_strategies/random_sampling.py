from .strategy import Strategy
import numpy as np


class randomSampling(Strategy):
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
        randomSampling class is a subclass of Strategy. Samples are chosen randomly.
        """
        super(randomSampling, self).__init__(
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
        Random sampling. Used as baseline model
        """
        rnd = np.random.RandomState()
        keys = np.sort(list(self.unsel_keys))
        idx_sel = rnd.choice(len(keys), self.shortlist_size, replace=False)
        self.shortlist_keys = [keys[i] for i in idx_sel]
        self.shortlist_keys_arr.append(self.shortlist_keys)
        self.sel_keys = self.select_among_shortlist_keys()
        self.update()
