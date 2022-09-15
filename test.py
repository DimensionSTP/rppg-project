import hydra
from omegaconf import OmegaConf

from pytorch_lightning.utilities.distributed import rank_zero_info

from src.engine.engine import test


@hydra.main(config_path="configs/", config_name="test.yaml")
def main(config):
    rank_zero_info(OmegaConf.to_yaml(config))
    return test(config)


if __name__ == "__main__":
    main()
