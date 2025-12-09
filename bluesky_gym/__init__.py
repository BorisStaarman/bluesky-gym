from gymnasium.envs.registration import register
from ray.tune.registry import register_env

from .utils import *
 
def register_envs():
    """Import the envs module so that environments / scenarios register themselves."""
    register(
        id="DescentEnv-v0",
        entry_point="bluesky_gym.envs.descent_env:DescentEnv",
        max_episode_steps=300,
    )

    register(
        id="PlanWaypointEnv-v0",
        entry_point="bluesky_gym.envs.plan_waypoint_env:PlanWaypointEnv",
        max_episode_steps=300,
    )

    register(
        id="HorizontalCREnv-v0",
        entry_point="bluesky_gym.envs.horizontal_cr_env:HorizontalCREnv",
        max_episode_steps=300,
    )

    register(
        id="VerticalCREnv-v0",
        entry_point="bluesky_gym.envs.vertical_cr_env:VerticalCREnv",
        max_episode_steps=300,
    )

    register(
        id="SectorCREnv-v0",
        entry_point="bluesky_gym.envs.sector_cr_env:SectorCREnv",
        max_episode_steps=200,
    )
    
    register(
        id="SectorCREnv-v0_boris",
        entry_point="bluesky_gym.envs.sector_cr_env_boris:SectorCREnv_boris",
        max_episode_steps=200,
    )

    register(
        id="ma_env-v0",
        entry_point="bluesky_gym.envs.ma_env:SectorEnv",
        max_episode_steps=200,
    )
    register(
        id="ma_env-v0",
        entry_point="bluesky_gym.envs.ma_env_SAC:SectorEnv",
        max_episode_steps=200,
    )
    register(
        id="ma_env-v0",
        entry_point="bluesky_gym.envs.ma_env_SAC_new:SectorEnv",
        max_episode_steps=400,
    )
    register(
        id="StaticObstacleEnv-v0",
        entry_point="bluesky_gym.envs.static_obstacle_env:StaticObstacleEnv",
        max_episode_steps=100,
    )

    register(
        id="MergeEnv-v0",
        entry_point="bluesky_gym.envs.merge_env:MergeEnv",
        max_episode_steps=50,
    )
    
    # the environment of the MARL scenario
    # This creator function is what RLlib will use. It takes a config dictionary
    # # and passes it as arguments to your environment's __init__ method.
    
    # DIT IS NU WEG GEHAALD DUS MOET BJI ALLE MAIN.PY ERBIJ, KIJKN NAAR DE MAIN.PY VAN 
    # from bluesky_gym.envs.ma_env_SAC_AM import SectorEnv
    from bluesky_gym.envs.ma_env_ppo import SectorEnv

    # Register for RLlib (Ray Tune)
    register_env("sector_env", lambda config: SectorEnv(**config))