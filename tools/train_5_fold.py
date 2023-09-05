import dotenv
import hydra
from omegaconf import DictConfig

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)

@hydra.main(
    config_path="../config",
    config_name="train_5_fold.yaml",
    version_base="1.3",
)
def main(config: DictConfig):
    # needed to import stapler
    import sys
    sys.path.append('../')

    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from stapler.entrypoints import stapler_entrypoint
    from stapler.utils.io_utils import extras, print_config, validate_config, ensemble_5_fold_output

    # Validate config -- Fails if there are mandatory missing values
    validate_config(config)

    # Applies optional utilities
    extras(config)

    if config.get("print_config"):
        print_config(config, resolve=True)

    result_dict = {}
    for i in range(5):
        result_dict[i] = stapler_entrypoint(config, i)

    if config.test_after_training:
        ensemble_5_fold_output(output_path=config.paths.output_dir, test_dataset_path=config.datamodule.test_data_path)


if __name__ == "__main__":

    main()
