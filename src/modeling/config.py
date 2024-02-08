from fastargs import Param, Section, get_current_config


Section("training", "training hyperparameters").params(
    optimizer=Param(str, "optimizer to use", default="sgd"),
    lr=Param(float, "learning rate to use", default=0.5),
    lr_schedule=Param(str, "learning rate schedule to use", default="triangle"),
    gamma=Param(float, "gamma for step learning rate schedule", default=0.1),
    step_size=Param(int, "step size for step learning rate schedule", default=30),
    weight_decay=Param(float, "l2 weight decay", default=5e-4),
    momentum=Param(float, "momentum for SGD", default=0.9),
    warmup_epochs=Param(float, "number of warmup epochs", default=5),
    epochs=Param(int, "number of epochs to run for", default=24),
    batch_size=Param(int, "batch size", default=512),
    label_smoothing=Param(float, "label smoothing", default=0.1),
    use_scaler=Param(bool, "whether to use gradient scaler", default=True),
    clip_grad=Param(bool, "whether to clip gradient", default=False),
    grad_clip_norm=Param(float, "gradient clipping norm", default=1),
    image_dtype=Param(str, "torch data type to use", default="float16"),
    decoder=Param(str, "Decoder to use", default="simple"),
    augmentation=Param(str, "Augmentation to use", default=""),
    freeze_features=Param(bool, "Whether to freeze features", default=False),
    num_workers=Param(int, "number of workers", default=10),
)


Section("evaluation", "evaluation hyperparameters").params(
    lr_tta=Param(
        bool,
        "Test time augmentation by averaging with horizontally flipped version",
        default=True,
    ),
)


Section("model", "model initialization").params(
    model_name=Param(str, "name of model", default="timm_resnet18"),
    pretrained=Param(str, "name of pretrained weights", default=None),
    resize=Param(int, "Size to resize to", default=None),
)


def populate_config(config_):
    assert config_ is not None
    config = get_current_config()
    config.collect(config_)
