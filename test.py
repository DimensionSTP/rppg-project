import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.utilities.distributed import rank_zero_info

from src.utils.setup import SetUp


def test(config: DictConfig):

    if "seed" in config:
        seed_everything(config.seed)

    setup = SetUp(config)

    dataset_module = setup.get_dataset_module()
    architecture_module = setup.get_architecture_module()
    callbacks = setup.get_callbacks()
    logger = setup.get_wandb_logger()

    trainer: Trainer = instantiate(
        config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )

    trainer.test(
        model=architecture_module, datamodule=dataset_module, ckpt_path=config.ckpt_path
    )


@hydra.main(config_path="configs/", config_name="test.yaml")
def main(config):
    rank_zero_info(OmegaConf.to_yaml(config))
    return test(config)


if __name__ == "__main__":
    main()
