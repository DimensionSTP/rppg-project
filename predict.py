import hydra
from omegaconf import OmegaConf

from pytorch_lightning.utilities.distributed import rank_zero_info

from src.engine.engine import predict


@hydra.main(config_path="configs/", config_name="customized_basic_rhythm_predict.yaml")
def main(config):
    rank_zero_info(OmegaConf.to_yaml(config))
    return predict(config)


if __name__ == "__main__":
    main()
