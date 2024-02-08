import numpy as np
import torch as ch
import torchvision

import src.modeling as modeling


class PreprocessWrapper(ch.nn.Module):
    def __init__(self, model, preprocess):
        super().__init__()
        self.model = model
        self.preprocess = preprocess

    def forward(self, x):
        x = self.preprocess(x)
        return self.model(x)


class NormalizeWrapper(PreprocessWrapper):
    def __init__(self, model, mean, std):
        preprocess = torchvision.transforms.Normalize(mean, std)
        super().__init__(model, preprocess)


class ModelManager:
    def __init__(self, num_classes, group=None, num_copies=None):
        self.num_classes = num_classes
        self.num_copies = num_copies
        self._group = group

    def train_and_save(self, path, overwrite=False, seed=None):
        raise NotImplementedError

    def get_normalization_params(self, model):
        raise NotImplementedError

    def _load_without_normalization(self, path):
        raise NotImplementedError

    def load(self, path):
        model = self._load_without_normalization(path)
        normalization_params = self.get_normalization_params(model)
        return NormalizeWrapper(model, *normalization_params).cuda()

    def trained(self, path):
        raise NotImplementedError

    def train_and_load(self, path, overwrite=False, seed=None):
        if not self.trained(path) or overwrite:
            self.train_and_save(path, overwrite=overwrite, seed=seed)
        return self.load(path)

    @property
    def group(self):
        return self._group


class TrainingMixin:
    def __init__(
        self,
        ffcv_dataset,
        indices,
        sample_size=1.0,
        sampling_fn=None,
        seed=0,
        loader_kwargs=None,
        val_ffcv_datasets=None,
        val_indices=None,
        reweighting_ffcv_dataset=None,
        reweighting_indices=None,
        reweighting_groups=None,
        reweighting_relevant_groups=None,
        reweighting_c=0.2,
        linear_probe_config=None,
        linear_probe_only=False,
        verbose_epochs=False,
        eval_every=1,
        checkpoint_every=None,
        save_every=None,
    ):
        self.ffcv_dataset = ffcv_dataset
        self.indices = indices
        self.sample_size = sample_size
        self.sampling_fn = sampling_fn
        self.seed = seed
        self.loader_kwargs = loader_kwargs or {}
        self.val_ffcv_datasets = val_ffcv_datasets
        self.val_indices = val_indices
        self.reweighting_ffcv_dataset = reweighting_ffcv_dataset
        self.reweighting_indices = reweighting_indices
        self.reweighting_groups = reweighting_groups
        self.reweighting_relevant_groups = reweighting_relevant_groups
        self.reweighting_c = reweighting_c
        self.linear_probe_config = linear_probe_config
        self.linear_probe_only = linear_probe_only
        self.verbose_epochs = verbose_epochs
        self.eval_every = eval_every
        self.checkpoint_every = checkpoint_every
        self.save_every = save_every

    def construct_model(self):
        raise NotImplementedError

    def train_and_save(self, path, overwrite=False, seed=None):
        model = self.construct_model()
        normalization_params = self.get_normalization_params(model)
        if self.indices is None:
            assert self.sample_size == 1.0
            sampled_indices = None
        else:
            # Note: this is not the same as self.seed or seed (because self.seed might be 0)
            seed = self.seed if self.seed is not None else seed
            assert seed is not None
            if self.sampling_fn is None:
                sampled_indices = np.random.RandomState(seed).choice(
                    self.indices,
                    size=int(self.sample_size * len(self.indices)),
                    replace=False
                )
            else:
                sampled_indices = self.sampling_fn(self.indices, self.sample_size, seed)
        modeling.populate_config(self.config)
        standard_loader = modeling.make_loader(
            self.ffcv_dataset,
            indices=sampled_indices,
            train=True,
            normalization_params=normalization_params,
            **self.loader_kwargs,
        )
        if self.val_ffcv_datasets is None:
            val_loaders = None
        else:
            val_loaders = {}
            for key in self.val_ffcv_datasets:
                val_loaders[key] = modeling.make_loader(
                    self.val_ffcv_datasets[key],
                    indices=self.val_indices[key],
                    train=False,
                    normalization_params=normalization_params,
                    **self.loader_kwargs,
                )
        metrics = {}
        assert not self.linear_probe_only or self.linear_probe_config is not None
        if self.linear_probe_config is not None:
            assert self.linear_probe_config["training"]["freeze_features"]
            modeling.populate_config(self.linear_probe_config)
            linear_probe_loader = modeling.make_loader(
                self.ffcv_dataset,
                indices=sampled_indices,
                train=True,
                normalization_params=normalization_params,
                **self.loader_kwargs,
            )
            save_path = path / "linear_probe"
            metrics["linear_probe"] = modeling.train(
                model,
                linear_probe_loader,
                save_path=save_path,
                val_loaders=val_loaders,
                verbose_epochs=self.verbose_epochs,
                eval_every=self.eval_every,
                checkpoint_every=self.checkpoint_every,
                save_every=self.save_every,
                overwrite=overwrite,
            )
            modeling.populate_config({"training": {"freeze_features": False}})
        if not self.linear_probe_only:
            modeling.populate_config(self.config)
            save_path = path
            if self.reweighting_ffcv_dataset is not None:
                save_path = path / "base"
            metrics["standard"] = modeling.train(
                model,
                standard_loader,
                val_loaders=val_loaders,
                save_path=save_path,
                verbose_epochs=self.verbose_epochs,
                eval_every=self.eval_every,
                checkpoint_every=self.checkpoint_every,
                save_every=self.save_every,
                overwrite=overwrite,
            )
        if self.reweighting_ffcv_dataset is not None:
            reweighting_loader = modeling.make_loader(
                self.reweighting_ffcv_dataset,
                indices=self.reweighting_indices,
                train=False,
                normalization_params=normalization_params,
                **self.loader_kwargs,
            )
            modeling.reweight(
                model,
                reweighting_loader, 
                self.reweighting_groups, 
                self.reweighting_relevant_groups, 
                c=self.reweighting_c,
            )
            modeling.save_model(path / "final.pt", model)
        return model, metrics


class SimpleModelManager(TrainingMixin, ModelManager):
    def __init__(self, ffcv_dataset, indices, config, group=None, num_copies=None, **kwargs):
        ModelManager.__init__(self, ffcv_dataset.num_classes, group=group, num_copies=num_copies)
        TrainingMixin.__init__(self, ffcv_dataset, indices, **kwargs)
        self.config = config

    def construct_model(self):
        modeling.populate_config(self.config)
        return modeling.construct_model(self.ffcv_dataset.num_classes)

    def get_normalization_params(self, model):
        return self.ffcv_dataset.normalization_mean, self.ffcv_dataset.normalization_std

    def _load_without_normalization(self, path):
        modeling.populate_config(self.config)
        return modeling.load_model(path / "final.pt", self.num_classes)

    def trained(self, path):
        return (path / "final.pt").exists()


class CLIPBaseModelManager(ModelManager):
    def __init__(self, ffcv_dataset, config, group=None, num_copies=None, zero_shot_init=True):
        super().__init__(ffcv_dataset.num_classes, group=group, num_copies=num_copies)
        self.ffcv_dataset = ffcv_dataset
        self.zero_shot_init = zero_shot_init
        self.config = config

    def get_normalization_params(self, model):
        mean = np.array(model.model.preprocessor.transforms[-1].mean) * 255
        std = np.array(model.model.preprocessor.transforms[-1].std) * 255
        return mean, std

    def _construct_model(self):
        modeling.populate_config(self.config)
        return modeling.construct_model(
            self.num_classes,
            class_names=self.ffcv_dataset.label_names if self.zero_shot_init else None,
            templates=self.ffcv_dataset.templates if self.zero_shot_init else None,
        )


class CLIPZeroShotModelManager(CLIPBaseModelManager):
    def train_and_save(self, path, seed=None):
        return None, []

    def _load_without_normalization(self, path):
        return self._construct_model()

    def trained(self, path):
        return True


class CLIPFinetunedModelManager(TrainingMixin, CLIPBaseModelManager):
    def __init__(self, ffcv_dataset, indices, config, group=None, num_copies=None, zero_shot_init=True, **kwargs):
        CLIPBaseModelManager.__init__(self, ffcv_dataset, config, group=group, num_copies=num_copies, zero_shot_init=zero_shot_init)
        TrainingMixin.__init__(self, ffcv_dataset, indices, **kwargs)

    def construct_model(self):
        return self._construct_model()

    def _load_without_normalization(self, path):
        modeling.populate_config(self.config)
        return modeling.load_model(
            path / "final.pt",
            self.num_classes,
            class_names=self.ffcv_dataset.label_names if self.zero_shot_init else None,
            templates=self.ffcv_dataset.templates if self.zero_shot_init else None,
        )

    def trained(self, path):
        return (path / "final.pt").exists()


class TimmBaseModelManager(ModelManager):
    def __init__(self, num_classes, config, group=None, num_copies=None):
        super().__init__(num_classes, group=group, num_copies=num_copies)
        self.config = config

    def get_normalization_params(self, model):
        if isinstance(model, PreprocessWrapper):
            model = model.model
        mean = np.array(model.transform.transforms[-1].mean) * 255
        std = np.array(model.transform.transforms[-1].std) * 255
        return mean, std

    def _construct_model(self):
        modeling.populate_config(self.config)
        return modeling.construct_model(self.num_classes)


class TimmModelManager(TimmBaseModelManager):
    def __init__(self, config, group=None, num_copies=None):
        super().__init__(1_000, config, group=group, num_copies=num_copies)

    def _load_without_normalization(self, path):
        model = self._construct_model()
        resize, crop = model.transform.transforms[:2]
        assert isinstance(resize, torchvision.transforms.Resize)
        assert isinstance(crop, torchvision.transforms.CenterCrop)
        preprocess = torchvision.transforms.Compose([resize, crop])
        return PreprocessWrapper(model, preprocess)
    
    def trained(self, path):
        return True

    def train_and_save(self, path, seed=None):
        pass


class TimmFinetunedModelManager(TrainingMixin, TimmBaseModelManager):
    def __init__(self, ffcv_dataset, indices, config, group=None, num_copies=None, **kwargs):
        TimmBaseModelManager.__init__(self, ffcv_dataset.num_classes, config, group=group, num_copies=num_copies)
        TrainingMixin.__init__(self, ffcv_dataset, indices, **kwargs)

    def construct_model(self):
        return self._construct_model()

    def _load_without_normalization(self, path):
        modeling.populate_config(self.config)
        return modeling.load_model(path / "final.pt", self.num_classes)

    def trained(self, path):
        return (path / "final.pt").exists()


class IntermediateEpochModelManager(ModelManager):
    def __init__(self, manager, epoch, group=None):
        assert isinstance(manager, SimpleModelManager)
        self.manager = manager
        self.epoch = epoch
        super().__init__(manager.num_classes, group=group or manager.group)

    def get_normalization_params(self, model):
        return self.manager.get_normalization_params(model)

    def train_and_save(self, path, seed=None):
        pass

    def _load_without_normalization(self, path):
        modeling.populate_config(self.manager.config)
        return modeling.load_model(path / f"epoch={self.epoch}.pt", self.num_classes)

    def trained(self, path):
        return (path / f"epoch={self.epoch}.pt").exists()
