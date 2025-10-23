import os, uuid
from ray import air, tune, init, shutdown
from ray.tune.schedulers import ASHAScheduler
from ray.rllib.algorithms.ppo import PPOConfig
from bluesky_gym import register_envs
from ray.rllib.policy.policy import PolicySpec


RUN_ID = "20_10_hyperparam_tuning"
register_envs()

# Map all agents to the same (shared) policy. Must be a top-level def so Ray can serialize it.
def map_all_agents_to_shared_policy(agent_id, *args, **kwargs):
    return "shared_policy"

def make_cfg():
    return (
        PPOConfig()
        .environment(
            env="sector_env",
            env_config={
                "n_agents": 6,
                # unique per-trial run folder for your CSV metrics
                "run_id": tune.sample_from(lambda _: f"{RUN_ID}_{uuid.uuid4().hex[:8]}"),
            },
            disable_env_checking=True,
        )
        .framework("torch")
        # Use the classic stack like your main.py (avoids RLModule/Catalog encoder issues)
        .api_stack(enable_rl_module_and_learner=False,
                   enable_env_runner_and_connector_v2=False)
        .env_runners(
            num_env_runners=max(1, os.cpu_count() - 2),
            rollout_fragment_length="auto",   # RLlib will choose a valid value for the batch
        )
        .resources(num_gpus=0)               # set 1 to force one GPU per trial
        # Multi-agent: one shared policy mapped to all agents
        .multi_agent(
            policies={"shared_policy": PolicySpec()},  # infer spaces at runtime
            policy_mapping_fn=map_all_agents_to_shared_policy,
        )
        .training(
            train_batch_size=4000,
            model={"fcnet_hiddens": [256, 256]},
            # search space (adjust as you like; grid_search evaluates all combos)
            lr=tune.grid_search([1e-4, 3e-4, 5e-4]),
            gamma=tune.grid_search([0.95, 0.98, 0.99]),
            # entropy_coeff=tune.grid_search([0.005, 0.01, 0.02]),
            # clip_param=tune.grid_search([0.1, 0.2, 0.3]),
            # lambda_=tune.grid_search([0.90, 0.95, 0.97]),
            # vf_clip_param=tune.grid_search([50.0, 100.0]),
            # num_sgd_iter=20,
            # sgd_minibatch_size=1024,
            # grad_clip=40.0,
        )
    )

if __name__ == "__main__":
    shutdown(); init(include_dashboard=False)

    scheduler = ASHAScheduler(
        metric="env_runners/episode_return_mean", mode="max",
        grace_period=5, max_t=25, reduction_factor=3,
    )

    tuner = tune.Tuner(
        "PPO",
        param_space=make_cfg().to_dict(),
        run_config=air.RunConfig(
            stop={"training_iteration": 2}, # set to 25 if it works
            storage_path=os.path.join(os.getcwd(), "ray_results"),
            name=RUN_ID,
        ),
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            # with grid_search, num_samples=1 runs the full grid
            num_samples=1,
        ),
    )
    results = tuner.fit()
    best = results.get_best_result(metric="env_runners/episode_return_mean", mode="max")
    print("Best reward:", best.metrics.get("env_runners/episode_return_mean"))
    print("Best config:", best.config)
    if best.checkpoint:
        print("Best checkpoint:", best.checkpoint.path)