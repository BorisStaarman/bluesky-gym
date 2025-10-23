"""
Comprehensive analysis of hyperparameter tuning results.
This script provides detailed insights into which parameters matter most.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ray.tune import Tuner
import json

def analyze_comprehensive_results():
    """Analyze results from the comprehensive hyperparameter search"""
    print("=== Comprehensive Hyperparameter Analysis ===")
    
    experiment_dir = os.path.join("..", "..", "ray_results", "comprehensive_hyperparam_search_25iter")
    
    if not os.path.exists(experiment_dir):
        print(f"Experiment directory not found: {experiment_dir}")
        print("Make sure you've run the comprehensive_tune.py script first.")
        return None
    
    # Find all trial directories
    trial_dirs = [d for d in os.listdir(experiment_dir) 
                  if os.path.isdir(os.path.join(experiment_dir, d)) and d.startswith("PPO_")]
    
    if not trial_dirs:
        print("No trial directories found.")
        return None
    
    print(f"Found {len(trial_dirs)} trials to analyze")
    
    trials_data = []
    for trial_dir in trial_dirs:
        trial_path = os.path.join(experiment_dir, trial_dir)
        result_file = os.path.join(trial_path, "result.json")
        
        if os.path.exists(result_file):
            try:
                # Read all results to get the progression
                results = []
                with open(result_file, 'r') as f:
                    for line in f:
                        results.append(json.loads(line))
                
                if not results:
                    continue
                
                # Get the best result (last result for completed trials)
                last_result = results[-1]
                
                # Extract hyperparameters from config
                config = last_result.get("config", {})
                
                trial_data = {
                    "trial_dir": trial_dir,
                    "episode_reward_mean": last_result.get("env_runners", {}).get("episode_reward_mean", float('-inf')),
                    "episode_reward_max": last_result.get("env_runners", {}).get("episode_reward_max", float('-inf')),
                    "episode_len_mean": last_result.get("env_runners", {}).get("episode_len_mean", 0),
                    "training_iterations": last_result.get("training_iteration", 0),
                    "time_total_s": last_result.get("time_total_s", 0),
                    "done": last_result.get("done", False),
                    
                    # Core hyperparameters
                    "lr": config.get("lr"),
                    "gamma": config.get("gamma"),
                    "clip_param": config.get("clip_param"),
                    "entropy_coeff": config.get("entropy_coeff"),
                    "vf_loss_coeff": config.get("vf_loss_coeff"),
                    "lambda": config.get("lambda"),
                    "num_sgd_iter": config.get("num_sgd_iter"),
                    "sgd_minibatch_size": config.get("sgd_minibatch_size"),
                    "grad_clip": config.get("grad_clip"),
                    "vf_clip_param": config.get("vf_clip_param"),
                    "kl_coeff": config.get("kl_coeff"),
                    "kl_target": config.get("kl_target"),
                    "fcnet_hiddens": str(config.get("model", {}).get("fcnet_hiddens", [])),
                    
                    # Learning progression (reward improvement)
                    "initial_reward": results[0].get("env_runners", {}).get("episode_reward_mean", float('-inf')) if len(results) > 0 else float('-inf'),
                    "final_reward": last_result.get("env_runners", {}).get("episode_reward_mean", float('-inf')),
                }
                
                # Calculate improvement
                if trial_data["initial_reward"] != float('-inf') and trial_data["final_reward"] != float('-inf'):
                    trial_data["reward_improvement"] = trial_data["final_reward"] - trial_data["initial_reward"]
                else:
                    trial_data["reward_improvement"] = 0
                
                trials_data.append(trial_data)
                
            except Exception as e:
                print(f"Error reading {result_file}: {e}")
    
    if not trials_data:
        print("No valid trial data found.")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(trials_data)
    
    print(f"\\nAnalyzed {len(df)} trials:")
    print(f"Completed trials: {df['done'].sum()}")
    print(f"Average training iterations: {df['training_iterations'].mean():.1f}")
    print(f"Total experiment time: {df['time_total_s'].sum() / 3600:.1f} hours")
    
    # === RESULTS SUMMARY ===
    print("\\n" + "="*50)
    print("TOP 10 TRIALS BY EPISODE REWARD")
    print("="*50)
    
    # Sort by episode reward and show top 10
    top_trials = df.nlargest(10, 'episode_reward_mean')
    
    for idx, (_, row) in enumerate(top_trials.iterrows()):
        print(f"\\nRank {idx+1}: Reward = {row['episode_reward_mean']:.2f}")
        print(f"  lr={row['lr']:.6f}, gamma={row['gamma']:.3f}, clip_param={row['clip_param']:.2f}")
        print(f"  entropy_coeff={row['entropy_coeff']:.6f}, iterations={row['training_iterations']}")
        print(f"  architecture={row['fcnet_hiddens']}")
    
    # === PARAMETER IMPORTANCE ANALYSIS ===
    print("\\n" + "="*50)
    print("PARAMETER CORRELATION WITH PERFORMANCE")
    print("="*50)
    
    # Calculate correlations with episode reward
    numeric_params = ['lr', 'gamma', 'clip_param', 'entropy_coeff', 'vf_loss_coeff', 
                     'lambda', 'num_sgd_iter', 'sgd_minibatch_size', 'grad_clip', 
                     'vf_clip_param', 'kl_coeff', 'kl_target']
    
    correlations = {}
    for param in numeric_params:
        if param in df.columns and df[param].notna().sum() > 1:
            corr = df[param].corr(df['episode_reward_mean'])
            if not np.isnan(corr):
                correlations[param] = corr
    
    # Sort by absolute correlation
    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    
    print("\\nParameter correlations with episode reward (|correlation| > 0.1):")
    for param, corr in sorted_corrs:
        if abs(corr) > 0.1:
            print(f"  {param:20s}: {corr:+.3f}")
    
    # === ARCHITECTURE ANALYSIS ===
    print("\\n" + "="*50)
    print("NETWORK ARCHITECTURE ANALYSIS")
    print("="*50)
    
    arch_performance = df.groupby('fcnet_hiddens')['episode_reward_mean'].agg(['mean', 'std', 'count']).round(2)
    arch_performance = arch_performance.sort_values('mean', ascending=False)
    print("\\nPerformance by network architecture:")
    print(arch_performance)
    
    # === LEARNING RATE ANALYSIS ===
    print("\\n" + "="*50)
    print("LEARNING RATE ANALYSIS")
    print("="*50)
    
    # Bin learning rates for analysis
    df['lr_bin'] = pd.cut(df['lr'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    lr_performance = df.groupby('lr_bin')['episode_reward_mean'].agg(['mean', 'std', 'count']).round(2)
    print("\\nPerformance by learning rate range:")
    print(lr_performance)
    
    # === RECOMMENDATIONS ===
    print("\\n" + "="*60)
    print("RECOMMENDATIONS BASED ON ANALYSIS")
    print("="*60)
    
    best_trial = df.loc[df['episode_reward_mean'].idxmax()]
    
    print(f"\\n🏆 BEST OVERALL CONFIGURATION:")
    print(f"   Episode Reward: {best_trial['episode_reward_mean']:.2f}")
    print(f"   Learning Rate: {best_trial['lr']:.6f}")
    print(f"   Gamma: {best_trial['gamma']:.4f}")
    print(f"   Clip Param: {best_trial['clip_param']:.3f}")
    print(f"   Entropy Coeff: {best_trial['entropy_coeff']:.6f}")
    print(f"   Network: {best_trial['fcnet_hiddens']}")
    print(f"   Training Iterations: {best_trial['training_iterations']}")
    
    # Parameter recommendations based on top performers
    top_5_trials = df.nlargest(5, 'episode_reward_mean')
    
    print(f"\\n📊 OPTIMAL PARAMETER RANGES (from top 5 trials):")
    for param in ['lr', 'gamma', 'clip_param', 'entropy_coeff']:
        if param in top_5_trials.columns:
            min_val = top_5_trials[param].min()
            max_val = top_5_trials[param].max()
            mean_val = top_5_trials[param].mean()
            print(f"   {param:15s}: {min_val:.6f} - {max_val:.6f} (mean: {mean_val:.6f})")
    
    # Save detailed results
    results_file = "comprehensive_results_analysis.csv"
    df.to_csv(results_file, index=False)
    print(f"\\n💾 Detailed results saved to: {results_file}")
    
    # Create best config file
    create_best_config_file(best_trial)
    
    return df, best_trial

def create_best_config_file(best_trial):
    """Create a configuration file with the best hyperparameters"""
    config_content = f'''"""
Best hyperparameters from comprehensive search
Episode Reward: {best_trial['episode_reward_mean']:.2f}
Training Iterations: {best_trial['training_iterations']}
"""

# === BEST HYPERPARAMETERS ===
BEST_LR = {best_trial['lr']:.8f}
BEST_GAMMA = {best_trial['gamma']:.6f}
BEST_CLIP_PARAM = {best_trial['clip_param']:.4f}
BEST_ENTROPY_COEFF = {best_trial['entropy_coeff']:.8f}
BEST_VF_LOSS_COEFF = {best_trial['vf_loss_coeff']:.4f}
BEST_LAMBDA = {best_trial['lambda']:.4f}
BEST_NUM_SGD_ITER = {best_trial['num_sgd_iter']}
BEST_SGD_MINIBATCH_SIZE = {best_trial['sgd_minibatch_size']}
BEST_GRAD_CLIP = {best_trial['grad_clip']:.1f}
BEST_VF_CLIP_PARAM = {best_trial['vf_clip_param']:.1f}
BEST_KL_COEFF = {best_trial['kl_coeff']:.4f}
BEST_KL_TARGET = {best_trial['kl_target']:.6f}
BEST_FCNET_HIDDENS = {best_trial['fcnet_hiddens']}

# Use in your training script:
from ray.rllib.algorithms.ppo import PPOConfig

config = (
    PPOConfig()
    .environment(env="sector_env", env_config={{"n_agents": 6}})
    .framework("torch")
    .training(
        lr=BEST_LR,
        gamma=BEST_GAMMA,
        clip_param=BEST_CLIP_PARAM,
        entropy_coeff=BEST_ENTROPY_COEFF,
        vf_loss_coeff=BEST_VF_LOSS_COEFF,
        lambda_=BEST_LAMBDA,
        num_sgd_iter=BEST_NUM_SGD_ITER,
        sgd_minibatch_size=BEST_SGD_MINIBATCH_SIZE,
        grad_clip=BEST_GRAD_CLIP,
        vf_clip_param=BEST_VF_CLIP_PARAM,
        kl_coeff=BEST_KL_COEFF,
        kl_target=BEST_KL_TARGET,
        train_batch_size=4000,
        model={{"fcnet_hiddens": {eval(best_trial['fcnet_hiddens'])}}},
    )
)
'''
    
    with open("best_comprehensive_config.py", "w") as f:
        f.write(config_content)
    
    print(f"✅ Best configuration saved to: best_comprehensive_config.py")

if __name__ == "__main__":
    try:
        df, best_trial = analyze_comprehensive_results()
        
        print("\\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        print("Files created:")
        print("- comprehensive_results_analysis.csv (detailed data)")
        print("- best_comprehensive_config.py (best configuration)")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        print("Make sure the comprehensive hyperparameter search has been run.")