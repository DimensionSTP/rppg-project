import json

import hydra
from omegaconf import OmegaConf, DictConfig

from pytorch_lightning.utilities.distributed import rank_zero_info

from src.pipelines.pipeline import predict


@hydra.main(config_path="configs/", config_name="rhythm_predict.yaml")
def main(config: DictConfig,) -> None:
    if config.is_tuned:
        params = json.load(open(config.tuned_hparams_path, "rt", encoding="UTF-8"))
        config = OmegaConf.merge(config, params)
    rank_zero_info(OmegaConf.to_yaml(config))
    return predict(config)


if __name__ == "__main__":
    main()
