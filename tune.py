from omegaconf import DictConfig
import hydra

from src.pipeline.pipeline import tune


@hydra.main(config_path="configs/", config_name="custom_rhythm_rhythm_tune.yaml")
def main(config: DictConfig,) -> None:
    return tune(config)


if __name__ == "__main__":
    main()