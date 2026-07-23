import json
import wandb
import logging
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def upload(sweep_id: Optional[str] = None):
    """Upload the best config from the given sweep to the "best_experiment" wandb Artifact.

    Args:
        sweep_id (Optional[str], optional): Sweep ID to look for the best config. If None, it will look for the last sweep in the cached last_sweep_metadata.json file. Defaults to None.
    """

    if sweep_id is None:
        # last_sweep_metadata = utils.load_json("last_sweep_metadata.json")
        # sweep_id = last_sweep_metadata["sweep_id"]

        # logger.info(f"Loading sweep_id from last_sweep_metadata.json with {sweep_id=}")
        return

    api = wandb.Api()
    sweep = wandb.sweep(
        f"{os.getenv("WANDB_ENTITY")}/{os.getenv("WANDB_PROJECT")}/{sweep_id}"
    )
    best_run = sweep.best_run()

    with wandb.init(
        name="best_experiment",
        job_type="hpo",
        group="train",
        run_id=best_run.id,
        resume="must",
    ) as run:
        run.use_artifact("config:latest")

        best_config = dict(run.config)

        logger.info(f"Best run {best_run.name}")
        logger.info("Best run config:")
        logger.info(best_config)
        logger.info(
            f"Best run = {best_run.name} with results {dict(run.summary['validation'])}"
        )

        config_path = "training_pipeline/config/best_config.json"
        with open(config_path, "w") as f:
            json.dump(best_config, f, indent=4)

        artifact = wandb.Artifact(
            name="best_config",
            type="model",
            metadata={"results": {"validation": dict(run.summary["validation"])}},
        )
        artifact.add_file(str(config_path))
        run.log_artifact(artifact)