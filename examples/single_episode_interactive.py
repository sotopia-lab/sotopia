from rich import print

from sotopia.generation_utils import generate_episode


def create_example_episode() -> None:
    """
    Create an example episode
    """
    participants = (
        input("Enter participants' names, separated by a comma:\t")
        or "Jack, Rose"
    )
    topic = input("Enter topic:\t") or "in a coffee shop"
    extra_info = (
        input("Enter extra info:\t")
        or "jack's goal is to figure out what rose would like for her birthday gift without letting her know"
    )
    episode = generate_episode(
        "gpt-4",
        participants=participants,
        topic=topic,
        extra_info=extra_info,
    )
    print(episode)


create_example_episode()
