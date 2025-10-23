
# Best hyperparameters found from tuning:
BEST_LR = 0.0005
BEST_GAMMA = 0.98

# You can use these in your training script like:
config = (
    PPOConfig()
    .environment(env="sector_env", env_config={"n_agents": 6})
    .framework("torch")
    .training(
        lr=BEST_LR,
        gamma=BEST_GAMMA,
        train_batch_size=4000,
        model={"fcnet_hiddens": [256, 256]},
    )
    # ... rest of your config
)
