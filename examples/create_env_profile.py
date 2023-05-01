from sotopia.database import EnvironmentProfile

env_profile = EnvironmentProfile(
    scenario="The conversation between two friends in a cafe",
    agent_goals=[
        "trying to figure out the gift preference of the other agent, but not let them know you are buying gift for them",
        "to have a good time",
    ],
)

env_profile.save()
