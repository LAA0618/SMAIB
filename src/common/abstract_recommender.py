import os
import numpy as np
import torch
import torch.nn as nn


class AbstractRecommender(nn.Module):
    r"""Base class for all models
    """
    def pre_epoch_processing(self):
        pass

    def post_epoch_processing(self):
        pass

    def calculate_loss(self, interaction):
        r"""Calculate the training loss for a batch data.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Training loss, shape: []
        """
        raise NotImplementedError

    def predict(self, interaction):
        r"""Predict the scores between users and items.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Predicted scores for given users and items, shape: [batch_size]
        """
        raise NotImplementedError

    def full_sort_predict(self, interaction):
        r"""full sort prediction function.
        Given users, calculate the scores between users and all candidate items.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Predicted scores for given users and all candidate items,
            shape: [n_batch_users * n_candidate_items]
        """
        raise NotImplementedError
    #
    # def __str__(self):
    #     """
    #     Model prints with number of trainable parameters
    #     """
    #     model_parameters = filter(lambda p: p.requires_grad, self.parameters())
    #     params = sum([np.prod(p.size()) for p in model_parameters])
    #     return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = self.parameters()
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)


class GeneralRecommender(AbstractRecommender):
    """This is a abstract general recommender. All the general model should implement this class.
    The base general recommender class provide the basic dataset and parameters information.
    """
    def _config_value(self, config, key, default=None):
        try:
            value = config[key]
        except Exception:
            return default
        if value is None:
            return default
        if isinstance(value, str) and value.lower() in ['none', 'null']:
            return default
        return value

    def _config_bool(self, config, key, default=False):
        value = self._config_value(config, key, default)
        if isinstance(value, str):
            return value.lower() in ['true', '1', 'yes', 'y']
        return bool(value)

    def _as_item_ids(self, values):
        if values is None:
            return np.array([], dtype=np.int64)
        values = np.asarray(values, dtype=np.int64).reshape(-1)
        values = values[(values >= 0) & (values < self.n_items)]
        return np.unique(values)

    def _load_missing_items(self, config, dataset_path):
        explicit_file = self._config_value(config, 'missing_items_file', '')
        candidates = []
        if explicit_file:
            candidates.append(explicit_file if os.path.isabs(explicit_file) else os.path.join(dataset_path, explicit_file))
        ratio = self._config_value(config, 'missing_ratio', 0.0)
        ratio_names = [str(ratio)]
        try:
            ratio_float = float(ratio)
            ratio_names.extend([
                ('%.3f' % ratio_float).rstrip('0').rstrip('.'),
                ('%.6f' % ratio_float).rstrip('0').rstrip('.'),
            ])
        except Exception:
            ratio_float = 0.0
        for name in dict.fromkeys(ratio_names):
            candidates.append(os.path.join(dataset_path, 'missing_items_{}.npy'.format(name)))
        candidates.append(os.path.join(dataset_path, 'missing_items.npy'))

        for file_path in candidates:
            if file_path and os.path.isfile(file_path):
                loaded = np.load(file_path, allow_pickle=True)
                if isinstance(loaded, np.ndarray) and loaded.shape == ():
                    loaded = loaded.item()
                if isinstance(loaded, dict):
                    both = self._as_item_ids(loaded.get('all', loaded.get('both', loaded.get('vt', loaded.get('tv')))))
                    visual = self._as_item_ids(loaded.get('v', loaded.get('visual', loaded.get('vision', loaded.get('image')))))
                    text = self._as_item_ids(loaded.get('t', loaded.get('text', loaded.get('txt'))))
                    visual = np.unique(np.concatenate([visual, both]))
                    text = np.unique(np.concatenate([text, both]))
                    return visual, text, file_path
                return self._as_item_ids(loaded), self._as_item_ids(loaded), file_path
        return None, None, None

    def _build_missing_availability(self, config, dataset_path):
        image_available = torch.ones(self.n_items, dtype=torch.bool, device=self.device)
        text_available = torch.ones(self.n_items, dtype=torch.bool, device=self.device)
        if not self._config_bool(config, 'missing_modal', False):
            return image_available, text_available

        visual_missing, text_missing, missing_file = self._load_missing_items(config, dataset_path)
        if visual_missing is None and text_missing is None:
            ratio = float(self._config_value(config, 'missing_ratio', 0.0))
            seed = int(self._config_value(config, 'missing_seed', 2026))
            rng = np.random.default_rng(seed)
            missing_num = int(self.n_items * ratio)
            sampled = rng.choice(np.arange(self.n_items), size=missing_num, replace=False) if missing_num > 0 else np.array([], dtype=np.int64)
            one_third = len(sampled) // 3
            two_third = 2 * len(sampled) // 3
            visual_missing = np.concatenate([sampled[:one_third], sampled[two_third:]])
            text_missing = np.concatenate([sampled[one_third:two_third], sampled[two_third:]])
        visual_missing = self._as_item_ids(visual_missing)
        text_missing = self._as_item_ids(text_missing)

        if visual_missing.size > 0:
            image_available[torch.as_tensor(visual_missing, dtype=torch.long, device=self.device)] = False
        if text_missing.size > 0:
            text_available[torch.as_tensor(text_missing, dtype=torch.long, device=self.device)] = False
        source = missing_file if missing_file is not None else 'deterministic fallback'
        print('>>>>>Missing modality protocol: image_missing={}, text_missing={}, source={}'.format(
            int((~image_available).sum().item()), int((~text_available).sum().item()), source))
        return image_available, text_available

    def _apply_missing_feature_fill(self, feat, available, config):
        if feat is None or available is None or available.all():
            return feat
        fill_mode = str(self._config_value(config, 'missing_feature_fill', 'mean')).lower()
        feat = feat.clone()
        if fill_mode == 'zero':
            fill_value = torch.zeros(feat.size(1), dtype=feat.dtype, device=feat.device)
        else:
            source = feat[available]
            if source.numel() == 0:
                source = feat
            fill_value = source.mean(dim=0)
        feat[~available] = fill_value
        return feat
    def __init__(self, config, dataloader):
        super(GeneralRecommender, self).__init__()

        # load dataset info
        self.USER_ID = config['USER_ID_FIELD']
        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.NEG_ITEM_ID = config['NEG_PREFIX'] + self.ITEM_ID
        self.n_users = dataloader.dataset.get_user_num()
        self.n_items = dataloader.dataset.get_item_num()

        # load parameters info
        self.batch_size = config['train_batch_size']
        self.device = config['device']

        # load encoded features here
        self.v_feat, self.t_feat = None, None
        if not config['end2end'] and config['is_multimodal_model']:
            dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
            # if file exist?
            v_feat_file_path = os.path.join(dataset_path, config['vision_feature_file'])
            t_feat_file_path = os.path.join(dataset_path, config['text_feature_file'])
            if os.path.isfile(v_feat_file_path):
                self.v_feat = torch.from_numpy(np.load(v_feat_file_path, allow_pickle=True)).type(torch.FloatTensor).to(
                    self.device)
            if os.path.isfile(t_feat_file_path):
                self.t_feat = torch.from_numpy(np.load(t_feat_file_path, allow_pickle=True)).type(torch.FloatTensor).to(
                    self.device)

            assert self.v_feat is not None or self.t_feat is not None, 'Features all NONE'

            self.image_available, self.text_available = self._build_missing_availability(config, dataset_path)
            if self.v_feat is not None:
                self.v_feat = self._apply_missing_feature_fill(self.v_feat, self.image_available, config)
            if self.t_feat is not None:
                self.t_feat = self._apply_missing_feature_fill(self.t_feat, self.text_available, config)
