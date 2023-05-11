"""Here we want to place any Input/Output utilities"""
import os
import json
import logging
import warnings
import pandas as pd
from typing import Any, Sequence
import pytorch_lightning as pl
import rich
import rich.syntax
import rich.tree
from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf.errors import InterpolationKeyError
from pytorch_lightning.utilities import rank_zero_only


def get_pylogger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    logging_levels = ("debug", "info", "warning", "error", "exception", "fatal", "critical")
    for level in logging_levels:
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


@rank_zero_only
def log_hyperparameters(
        config: DictConfig,
        model: pl.LightningModule,
        trainer: pl.Trainer,
) -> None:
    """Controls which config parts are saved by Lightning loggers.
    Additionaly saves:
    - number of model parameters
    """

    if not trainer.logger:
        return

    hparams = {}

    # choose which parts of hydra config will be saved to loggers
    hparams["model"] = config["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    hparams["model/params/non_trainable"] = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    hparams["datamodule"] = config["datamodule"]
    hparams["trainer"] = config["trainer"]

    if "seed" in config:
        hparams["seed"] = config["seed"]
    if "callbacks" in config:
        hparams["callbacks"] = config["callbacks"]

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)


def debug_function(x: float):
    """
    Function to use for debugging (e.g. github workflow testing)
    :param x:
    :return: x^2
    """
    return x ** 2


def validate_config(cfg: Any):
    if isinstance(cfg, ListConfig):
        for x in cfg:
            validate_config(x)
    elif isinstance(cfg, DictConfig):
        for name, v in cfg.items():
            if name == "hydra":
                print("Skipped validating hydra native configs")
                continue
            try:
                validate_config(v)
            except InterpolationKeyError:
                print(f"Skipped validating {name}: {v}")
                continue


@rank_zero_only
def print_config(
        config: DictConfig,
        fields: Sequence[str] = (
                "trainer",
                "model",
                "experiment",
                "datamodule",
                "callbacks",
                "logger",
                "test_after_training",
                "seed",
                "name",
        ),
        resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.
    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    with open("config_tree.log", "w") as fp:
        rich.print(tree, file=fp)


def extras(config: DictConfig) -> None:
    """A couple of optional utilities, controlled by main config file:
    - disabling warnings
    - forcing debug friendly configuration
    - verifying experiment name is set when running in experiment mode
    Modifies DictConfig in place.
    Args:
        config (DictConfig): Configuration composed by Hydra.
    """

    logger = get_pylogger(__name__)

    # disable python warnings if <config.ignore_warnings=True>
    if config.get("ignore_warnings"):
        logger.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # verify experiment name is set when running in experiment mode
    if config.get("enforce_tags") and (not config.get("tags") or config.get("tags") == ["dev"]):
        logger.info(
            "Running in experiment mode without tags specified"
            "Use `python run.py experiment=some_experiment tags=['some_tag',...]`, or change it in the experiment yaml"
        )
        logger.info("Exiting...")
        exit()

    # force debugger friendly configuration if <config.trainer.fast_dev_run=True>
    # debuggers don't like GPUs and multiprocessing
    if config.trainer.get("fast_dev_run"):
        logger.info("Forcing debugger friendly configuration! <config.trainer.fast_dev_run=True>")
        if config.trainer.get("gpus"):
            config.trainer.gpus = 0
        if config.datamodule.get("pin_memory"):
            config.datamodule.pin_memory = False
        if config.datamodule.get("num_workers"):
            config.datamodule.num_workers = 0


def save_output(output, path, append):
    logger = get_pylogger(__name__)
    logger.info("Saving test results to json at {}".format(path))
    json_path = os.path.join(path, f"test_results{append}.json")
    with open(json_path, "w") as f:
        json.dump(output, f)
    logger.info(f"Test results saved to {json_path}")


def ensemble_5_fold_output(output_path, test_dataset_path):
    logger = get_pylogger(__name__)
    logger.info("Checking 5-fold test results from {}".format(output_path))
    df_test = pd.read_csv(test_dataset_path)

    pred_cls_mean = [0] * len(df_test)
    for i in range(5):
        with open(os.path.join(output_path, f'test_results{i}.json')) as f:
            data = json.load(f)
            pred_cls = []
            label_cls = []
            for batch in data:
                pred_cls.append(batch['preds_cls'])
                label_cls.append(batch['labels_cls'])
            pred_cls = [item for sublist in pred_cls for item in sublist]
            label_cls = [item for sublist in label_cls for item in sublist]
            if df_test['label_true_pair'].tolist() != label_cls:
                raise ValueError('Labels from the test dataset are not same as the labels in the model output')
            pred_cls_mean = [x + y / 5 for x, y in zip(pred_cls_mean, pred_cls)]

    df_test['pred_cls'] = pred_cls_mean
    df_test.to_csv(os.path.join(output_path, f'predictions_5_fold_ensamble.csv'))
    logger.info(f"Ensembled test results saved to {output_path}")
