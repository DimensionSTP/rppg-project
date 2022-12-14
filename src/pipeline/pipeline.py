from hydra.utils import instantiate
from omegaconf import DictConfig

from pytorch_lightning import Trainer, seed_everything

from ..utils.setup import SetUp


def train(config: DictConfig):

    if "seed" in config:
        seed_everything(config.seed)

    setup = SetUp(config)

    train_loader = setup.get_train_loader()
    val_loader = setup.get_val_loader()
    # dataset_module = setup.get_dataset_module()
    architecture_module = setup.get_architecture_module()
    callbacks = setup.get_callbacks()
    logger = setup.get_wandb_logger()

    trainer: Trainer = instantiate(
        config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )

    trainer.fit(
        model=architecture_module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
    # trainer.fit(model=architecture_module, datamodule=dataset_module)


def test(config: DictConfig):

    if "seed" in config:
        seed_everything(config.seed)

    setup = SetUp(config)

    test_loader = setup.get_test_loader()
    # dataset_module = setup.get_dataset_module()
    architecture_module = setup.get_architecture_module()
    callbacks = setup.get_callbacks()
    logger = setup.get_wandb_logger()

    trainer: Trainer = instantiate(
        config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )

    trainer.test(
        model=architecture_module, dataloaders=test_loader, ckpt_path=config.ckpt_path
    )
    # trainer.test(
    #     model=architecture_module, datamodule=dataset_module, ckpt_path=config.ckpt_path
    # )


def predict(config: DictConfig):

    if "seed" in config:
        seed_everything(config.seed)

    setup = SetUp(config)

    test_loader = setup.get_test_loader()
    architecture_module = setup.get_architecture_module()
    callbacks = setup.get_callbacks()
    logger = setup.get_wandb_logger()

    trainer: Trainer = instantiate(
        config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )

    trainer.predict(
        model=architecture_module, dataloaders=test_loader, ckpt_path=config.ckpt_path
    )
