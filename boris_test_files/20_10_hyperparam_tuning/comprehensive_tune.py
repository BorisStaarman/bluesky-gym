import os, uuid
from ray import air, tune, init, shutdown
from ray.tune.schedulers import ASHAScheduler
from ray.rllib.algorithms.ppo import PPOConfig
from bluesky_gym import register_envs
from ray.rllib.policy.policy import PolicySpec


RUN_ID = "comprehensive_hyperparam_search_25iter"
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
            model={
                # Network architecture - important for learning capacity
                "fcnet_hiddens": tune.choice([
                    [128, 128],      # Smaller network - faster but less capacity
                    [256, 256],      # Your current architecture
                    [512, 256],      # Larger input processing
                    [256, 256, 128], # Deeper network
                ]),
            },
            
            # === CORE HYPERPARAMETERS ===
            # Learning rate - most critical parameter
            lr=tune.loguniform(1e-5, 1e-3),  # Log scale from 0.00001 to 0.001
            
            # Discount factor - how much future rewards matter
            gamma=tune.uniform(0.95, 0.999),  # Continuous range around your best value
            
            # === PPO-SPECIFIC PARAMETERS ===
            # Clipping parameter - controls policy update size
            clip_param=tune.uniform(0.1, 0.3),  # Standard range for PPO
            
            # Entropy coefficient - exploration vs exploitation
            entropy_coeff=tune.loguniform(1e-4, 1e-1),  # 0.0001 to 0.1
            
            # Value function coefficient
            vf_loss_coeff=tune.uniform(0.5, 2.0),
            
            # GAE lambda - bias vs variance tradeoff for advantage estimation
            lambda_=tune.uniform(0.90, 0.99),
            
            # === TRAINING DYNAMICS ===
            # Number of SGD iterations per training batch
            num_sgd_iter=tune.choice([10, 20, 30]),
            
            # Mini-batch size for SGD updates
            sgd_minibatch_size=tune.choice([256, 512, 1024]),
            
            # Gradient clipping - prevents exploding gradients
            grad_clip=tune.choice([10.0, 40.0, 100.0]),
            
            # Value function clipping
            vf_clip_param=tune.choice([10.0, 50.0, 100.0]),
            
            # KL divergence coefficient for adaptive learning
            kl_coeff=tune.uniform(0.1, 0.5),
            kl_target=tune.uniform(0.005, 0.02),
        )
    )

if __name__ == "__main__":
    shutdown(); init(include_dashboard=False)

    # ASHA scheduler - stops poorly performing trials early
    scheduler = ASHAScheduler(
        metric="env_runners/episode_reward_mean", 
        mode="max",
        grace_period=8,      # Let trials run for at least 8 iterations before stopping
        max_t=25,            # Maximum iterations per trial
        reduction_factor=3,  # Stop bottom 1/3 of trials at each evaluation
    )

    tuner = tune.Tuner(
        "PPO",
        param_space=make_cfg().to_dict(),
        run_config=air.RunConfig(
            stop={"training_iteration": 25},  # 25 iterations as requested
            storage_path=os.path.join(os.getcwd(), "..", "..", "ray_results"),
            name=RUN_ID,
            # Checkpoint configuration
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=5,  # Save every 5 iterations
                num_to_keep=2,          # Keep only 2 most recent checkpoints per trial
            ),
        ),
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            num_samples=50,  # Number of different hyperparameter combinations to try
            max_concurrent_trials=4,  # Run 4 trials in parallel (adjust based on your CPU)
        ),
    )
    
    print("Starting comprehensive hyperparameter search...")
    print(f"- {50} different hyperparameter combinations")
    print(f"- {25} iterations per successful trial")
    print(f"- {4000} timesteps per iteration")
    print(f"- Early stopping with ASHA scheduler")
    print(f"- Results will be saved to: ray_results/{RUN_ID}")
    print("\nThis will take several hours to complete...")
    
    results = tuner.fit()
    
    # Get best results
    best = results.get_best_result(metric="env_runners/episode_reward_mean", mode="max")
    print("\n" + "="*60)
    print("COMPREHENSIVE HYPERPARAMETER SEARCH COMPLETED!")
    print("="*60)
    print(f"Best episode reward mean: {best.metrics.get('env_runners/episode_reward_mean', 'N/A')}")
    print(f"Best trial completed {best.metrics.get('training_iteration', 0)} iterations")
    print("\nBest hyperparameters:")
    
    # Print the most important hyperparameters
    important_params = [
        'lr', 'gamma', 'clip_param', 'entropy_coeff', 'vf_loss_coeff', 
        'lambda_', 'num_sgd_iter', 'sgd_minibatch_size', 'grad_clip',
        'vf_clip_param', 'kl_coeff', 'kl_target'
    ]
    
    config = best.config
    for param in important_params:
        if param in config:
            print(f"  {param}: {config[param]}")
    
    print(f"\nModel architecture: {config.get('model', {}).get('fcnet_hiddens', 'N/A')}")
    
    if best.checkpoint:
        print(f"\nBest checkpoint saved at: {best.checkpoint.path}")
        print("You can use this checkpoint to continue training or for evaluation.")
    
    print(f"\nAll results saved to: ray_results/{RUN_ID}")
    print("Use the analyze_results.py script to get detailed analysis of all trials.")