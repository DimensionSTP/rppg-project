from hydra.utils import instantiate
from omegaconf import DictConfig

from pytorch_lightning import Trainer, seed_everything

from ..utils.setup import SetUp
from ..tuners.rhythm_tuner import RhythmTuner


def train(
    config: DictConfig,
) -> None:

    if "seed" in config:
        seed_everything(config.seed)

    setup = SetUp(config)

    train_loader = setup.get_train_loader()
    val_loader = setup.get_val_loader()
    architecture = setup.get_architecture()
    callbacks = setup.get_callbacks()
    logger = setup.get_wandb_logger()

    logged_hparams = {}
    for key, value in config.architecture.items():
        if key != "_target_" and key != "model":
            logged_hparams[key] = value
    if "model" in config.architecture:
        for key, value in config.architecture["model"].items():
            if key != "_target_":
                logged_hparams[key] = value
    logged_hparams["batch_size"] = config.batch_size
    logged_hparams["epoch"] = config.epoch
    logged_hparams["seed"] = config.seed
    for key, value in config.trainer.items():
        if key != "_target_":
            logged_hparams[key] = value
    logger.log_hyperparams(logged_hparams)

    trainer: Trainer = instantiate(
        config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )

    try:
        trainer.fit(
            model=architecture,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )
        logger.experiment.alert(
            title="Training Complete",
            text="Training process has successfully finished.",
            level="INFO",
        )
    except Exception as e:
        logger.experiment.alert(
            title="Training Error",
            text="An error occurred during training",
            level="ERROR",
        )
        raise e


def test(
    config: DictConfig,
) -> None:

    if "seed" in config:
        seed_everything(config.seed)

    setup = SetUp(config)

    test_loader = setup.get_test_loader()
    architecture = setup.get_architecture()
    callbacks = setup.get_callbacks()
    logger = setup.get_wandb_logger()

    logged_hparams = {}
    for key, value in config.architecture.items():
        if key != "_target_" and key != "model":
            logged_hparams[key] = value
    if "model" in config.architecture:
        for key, value in config.architecture["model"].items():
            if key != "_target_":
                logged_hparams[key] = value
    logged_hparams["batch_size"] = config.batch_size
    logged_hparams["epoch"] = config.epoch
    logged_hparams["seed"] = config.seed
    for key, value in config.trainer.items():
        if key != "_target_":
            logged_hparams[key] = value
    logger.log_hyperparams(logged_hparams)

    trainer: Trainer = instantiate(
        config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )

    try:
        trainer.test(
            model=architecture,
            dataloaders=test_loader,
            ckpt_path=config.ckpt_path,
        )
        logger.experiment.alert(
            title="Testing Complete",
            text="Testing process has successfully finished.",
            level="INFO",
        )
    except Exception as e:
        logger.experiment.alert(
            title="Testing Error",
            text="An error occurred during testing",
            level="ERROR",
        )
        raise e


def predict(
    config: DictConfig,
) -> None:

    if "seed" in config:
        seed_everything(config.seed)

    setup = SetUp(config)

    predict_loader = setup.get_test_loader()
    architecture = setup.get_architecture()
    callbacks = setup.get_callbacks()
    logger = setup.get_wandb_logger()

    logged_hparams = {}
    for key, value in config.architecture.items():
        if key != "_target_" and key != "model":
            logged_hparams[key] = value
    if "model" in config.architecture:
        for key, value in config.architecture["model"].items():
            if key != "_target_":
                logged_hparams[key] = value
    logged_hparams["batch_size"] = config.batch_size
    logged_hparams["epoch"] = config.epoch
    logged_hparams["seed"] = config.seed
    for key, value in config.trainer.items():
        if key != "_target_":
            logged_hparams[key] = value
    logger.log_hyperparams(logged_hparams)

    trainer: Trainer = instantiate(
        config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )

    try:
        trainer.predict(
            model=architecture,
            dataloaders=predict_loader,
            ckpt_path=config.ckpt_path,
        )
        logger.experiment.alert(
            title="Prediction Complete",
            text="Prediction process has successfully finished.",
            level="INFO",
        )
    except Exception as e:
        logger.experiment.alert(
            title="Prediction Error",
            text="An error occurred during prediction",
            level="ERROR",
        )
        raise e


def tune(
    config: DictConfig,
) -> None:

    if "seed" in config:
        seed_everything(config.seed)

    setup = SetUp(config)

    train_loader = setup.get_train_loader()
    val_loader = setup.get_val_loader()
    logger = setup.get_wandb_logger()

    tuner: RhythmTuner = instantiate(
        config.tuner, train_loader=train_loader, val_loader=val_loader, logger=logger
    )
    tuner()