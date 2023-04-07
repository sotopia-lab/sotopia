from sotopia.generation_utils import generate_episode


def create_example_episode() -> None:
    """
    Create an example episode
    """
    episode = generate_episode("gpt-3.5-turbo")
    print(episode)


create_example_episode()
