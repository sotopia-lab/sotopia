from rich import print

from sotopia.generation_utils import generate_episode


def create_example_episode() -> None:
    """
    Create an example episode
    """
    participants = (
        input("Enter participants' names, separated by a comma:\t")
        or "Jack (a greedy person), Rose"
    )
    topic = input("Enter topic:\t") or "lawsuit"
    extra_info = input("Enter extra info:\t") or ""
    episode = generate_episode(
        "gpt-3.5-turbo",
        participants=participants,
        topic=topic,
        extra_info=extra_info,
    )
    print(episode)


create_example_episode()
