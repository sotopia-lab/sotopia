from sotopia.generation_utils import generate_episode_singleRound


def create_example_episode() -> None:
    """
    Create an example episode
    """
    episode = generate_episode_singleRound("gpt-3.5-turbo")
    print(episode)


create_example_episode()
