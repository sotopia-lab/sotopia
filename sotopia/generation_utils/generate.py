"""
Script to generate profile, scneario, and rewards.
"""
from argparse import Namespace


class Generate:
    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.profile = ""
        self.scenario = ""
        self.rewards = ""
        self.generate_profile()
        self.generate_scenario()
        self.generate_rewards()

    def generate_profile(self) -> None:
        self.profile = "profile"

    def generate_scenario(self) -> None:
        self.scenario = "scenario"

    def generate_rewards(self) -> None:
        self.rewards = "rewards"
