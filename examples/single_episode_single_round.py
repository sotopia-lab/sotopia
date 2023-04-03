from rich import print

from sotopia.generation_utils import generate_episode_single_round


def create_example_episode() -> None:
    """
    Create an example episode
    """
    episode = generate_episode_single_round("gpt-3.5-turbo")
    print(episode)


create_example_episode()
