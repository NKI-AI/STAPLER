import dotenv
import hydra
import mlflow
from omegaconf import DictConfig

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)

@hydra.main(
    config_path="../config",
    config_name="train_5_fold.yaml",
    version_base="1.2",
)
def main(config: DictConfig):

    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from stapler.entrypoints import stapler_entrypoint
    from stapler.utils.io_utils import extras, print_config, validate_config

    # Validate config -- Fails if there are mandatory missing values
    validate_config(config)

    # Applies optional utilities
    extras(config)

    if config.get("print_config"):
        print_config(config, resolve=True)

    result_dict = {}
    for i in range(5):
        result_dict[i] = stapler_entrypoint(config, i)


if __name__ == "__main__":

    main()
