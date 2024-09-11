import asyncio
import logging
import subprocess
import typing
from datetime import datetime
from logging import FileHandler

import typer
from rich.logging import RichHandler
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from sotopia.database.logs import AnnotationForEpisode, EpisodeLog
from sotopia.generation_utils.generate import LLM_Name
from sotopia.server import aevaluate_one_episode

# date and message only
FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"

process = subprocess.Popen(
    ["git", "rev-parse", "HEAD"], shell=False, stdout=subprocess.PIPE
)
git_head_hash = process.communicate()[0].strip()

logging.basicConfig(
    level=15,
    format=FORMAT,
    datefmt="[%X]",
    handlers=[
        RichHandler(),
        FileHandler(
            datetime.now().strftime(
                f"./logs/%H_%M_%d_%m_%Y_{str(git_head_hash.decode('utf-8'))}.log"
            )
        ),
    ],
)
app = typer.Typer()


def run_async_server_in_batch_aevaluate(
    batch_size: int = 10,
    model: LLM_Name = "gpt-4",
    reeval_list: list[str] = [],
    tag: str | None = None,
    push_to_db: bool = False,
    verbose: bool = False,
) -> None:
    """
    Runs episodes from a list asynchronously in batches using asyncio. It retrieves
    episodes, runs them with an LLM, and logs progress and results. The process
    continues until all episodes are processed or interrupted by the logger level
    being set to critical.

    Args:
        batch_size (int): 10 by default. It represents the maximum number of episode
            logs to be evaluated together before running them asynchronously as a
            batch.
        model (LLM_Name): Initialized with the string "gpt-4" by default. This
            suggests it refers to an artificial intelligence model, specifically
            a Large Language Model, named GPT-4.
        reeval_list (list[str]): Initialized as an empty list []. It appears to
            hold environment primary keys for which episodes are evaluated.
        tag (str | None): Used to specify a tag when running aeval on an episode,
            presumably for logging or identifying purposes, though its exact use
            is context-dependent.
        push_to_db (bool): Set to False by default. It controls whether episode
            results are pushed to a database or not. When True, episode results
            are pushed to a database.
        verbose (bool): Set to False by default. When it is True, the function
            runs with normal logging level; otherwise, the logging level is set
            to CRITICAL.

    """
    if not verbose:
        logger = logging.getLogger()
        logger.setLevel(logging.CRITICAL)
        rich_handler = logger.handlers[0]
        logger.removeHandler(rich_handler)

    episode_batch: list[EpisodeLog] = []
    while True:
        for env_pk in tqdm(reeval_list):
            episode = EpisodeLog.get(env_pk)
            episode_batch.append(episode)
            if len(episode_batch) == batch_size:
                logging.info(f"Running batch of {batch_size} episodes: {episode_batch}")
                episode_futures = [
                    aevaluate_one_episode(
                        episode=episode,
                        model=model,
                        tag=tag,
                        push_to_db=push_to_db,
                    )
                    for episode in episode_batch
                ]
                asyncio.run(
                    tqdm_asyncio.gather(*episode_futures, desc="Running one batch")
                )
                episode_batch = []
        else:
            if episode_batch:
                logging.info(f"Running batch of {batch_size} episodes: {episode_batch}")
                episode_futures = [
                    aevaluate_one_episode(
                        episode=episode,
                        model=model,
                        tag=tag,
                        push_to_db=push_to_db,
                    )
                    for episode in episode_batch
                ]
                asyncio.run(
                    tqdm_asyncio.gather(*episode_futures, desc="Running one batch")
                )
            return None


@app.command()
def run_server(
    tag: str = "reeval_llama2",
    model: str = "togethercomputer/llama-2-70b-chat",  # Why typer does not accept LLM_Name?
    batch_size: int = 10,
    push_to_db: bool = True,
    verbose: bool = False,
) -> None:
    """
    Runs an asynchronous server for evaluating LLM model performance on annotated
    episodes, utilizing a specified batch size and pushing results to a database
    if required.

    Args:
        tag (str): Initialized to "reeval_llama2" by default. It appears to be
            used as an identifier for the server run, possibly in logging or
            configuration purposes.
        model (str): Optional. It defaults to `"togethercomputer/llama-2-70b-chat"`.
            The value must be castable to an instance of the class LLM_Name.
        batch_size (int): 10 by default, used as an argument to the
            `run_async_server_in_batch_aevaluate` function where it likely determines
            the number of episodes processed together in each batch.
        push_to_db (bool): Optional, defaulting to True when not specified by the
            caller. It determines whether data should be pushed to a database after
            running the server in batch mode.
        verbose (bool): Set to False by default.
            It determines whether additional information should be printed during
            execution of the server run.

    """
    annotated_episodes_pks = [anno.episode for anno in AnnotationForEpisode.all()]
    annotated_episodes_pks = list(set(annotated_episodes_pks))
    model = typing.cast(LLM_Name, model)
    # Call the function with the specified parameters
    run_async_server_in_batch_aevaluate(
        tag=tag,
        model=model,
        batch_size=batch_size,
        push_to_db=push_to_db,
        verbose=verbose,
        reeval_list=annotated_episodes_pks,
    )


if __name__ == "__main__":
    app()
