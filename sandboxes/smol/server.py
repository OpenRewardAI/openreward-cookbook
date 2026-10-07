"""Serve SmolEnv as an OpenReward environment."""
from openreward.environments import Server

from smol_env import SmolEnv

if __name__ == "__main__":
    Server([SmolEnv]).run()
