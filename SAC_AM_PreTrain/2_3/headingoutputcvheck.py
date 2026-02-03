# standard imports
import os
import sys
import shutil
import matplotlib.pyplot as plt
import numpy as np
import time
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import io
import copy

import bluesky as bs
from ray.rllib.policy.sample_batch import SampleBatch, MultiAgentBatch
from mvp_2d import MVP_2D
import bluesky_gym.envs.common.functions as fn

# MARL ray imports
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.sac import SACConfig

from ray.rllib.models import ModelCatalog
# from attention_model_M import AttentionSACModel # Multiplicative method
from attention_model_A import AttentionSACModel # additive method

# Make sure these imports point to your custom environment registration
from bluesky_gym.envs.ma_env_SAC_AM import SectorEnv
from ray.rllib.policy.sample_batch import SampleBatch

from ray.tune.registry import register_env

import torch
from ray.rllib.algorithms.callbacks import DefaultCallbacks

from run_config import RUN_ID

# Register your custom environment directly for RLlib
register_env("sector_env", lambda config: SectorEnv(**config))
ModelCatalog.register_custom_model("attention_sac", AttentionSACModel)
class SACExpert:
    def __init__(self, safe_dist_m=100.0, lookahead_s=15.0):
        # We gebruiken de MVP logica die ook in Stage 1 PPO werkte
        self.mvp = MVP_2D(safe_distance=safe_dist_m, lookahead_time=lookahead_s)

    def get_action(self, env, agent_id):
        try:
            ac_idx = bs.traf.id2idx(agent_id)
            # 1. Huidige staat (meters t.o.v. center)
            pos = env.compute_relative_position(env.center, bs.traf.lat[ac_idx], bs.traf.lon[ac_idx])
            vel = np.array([
                np.cos(np.deg2rad(bs.traf.hdg[ac_idx])) * bs.traf.gs[ac_idx],
                np.sin(np.deg2rad(bs.traf.hdg[ac_idx])) * bs.traf.gs[ac_idx]
            ])

            # 2. Buren verzamelen
            neighbors = []
            for other in env.agents:
                if other == agent_id: continue
                o_idx = bs.traf.id2idx(other)
                neighbors.append({
                    'pos': env.compute_relative_position(env.center, bs.traf.lat[o_idx], bs.traf.lon[o_idx]),
                    'vel': np.array([np.cos(np.deg2rad(bs.traf.hdg[o_idx])) * bs.traf.gs[o_idx],
                                   np.sin(np.deg2rad(bs.traf.hdg[o_idx])) * bs.traf.gs[o_idx]])
                })

            # 3. MVP Berekening
            target_vel = self.mvp.calculate_avoidance_velocity(pos, vel, neighbors)
            
            # Indien geen conflict: direct naar waypoint (zoals in PPO Stage 1) 
            if np.array_equal(target_vel, vel):
                wpt_lat, wpt_lon = env.agent_waypoints[agent_id]
                wpt_qdr, _ = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], wpt_lat, wpt_lon)
                target_hdg, target_spd = wpt_qdr, 35.0 
            else:
                target_hdg = np.rad2deg(np.arctan2(target_vel[1], target_vel[0]))
                target_spd = np.linalg.norm(target_vel) * 1.94384 # m/s -> kts
            
            # 4. Schalen naar de [-1, 1] actie-ruimte van SectorEnv
            dh = fn.bound_angle_positive_negative_180(target_hdg - bs.traf.hdg[ac_idx])
            dv = target_spd - (bs.traf.tas[ac_idx] * 1.94384)
            
            # Schaling: D_HEADING=45, D_VELOCITY=3.33 (10/3)
            return np.array([np.clip(dh / 45.0, -1, 1), np.clip(dv / (10/3), -1, 1)], dtype=np.float32)
        except Exception:
            return np.array([0.0, 0.0], dtype=np.float32)

def diagnostic_run(n_episodes=50):
    print(f"🔍 Diagnostische run gestart: {n_episodes} episodes...")
    env = SectorEnv(n_agents=20, run_id="diagnostic")
    expert = SACExpert()
    
    all_heading_actions = []
    conflict_heading_actions = []
    threshold = 0.02

    for ep in range(n_episodes):
        obs, _ = env.reset()
        while env.agents:
            active_agents = list(obs.keys())
            agent_actions = {aid: expert.get_action(env, aid) for aid in active_agents}
            
            # --- AFSTAND BEREKENEN VIA DE OMGEVINGSDATA ---
            # We gebruiken de interne BlueSky data van de omgeving
            for aid in active_agents:
                heading_change = abs(agent_actions[aid][0])
                all_heading_actions.append(heading_change)

                # Zoek de dichtstbijzijnde buur
                min_dist = 999.0
                # In BlueSky omgevingen staan posities vaak in env.bs.lat/lon of env.ax/ay
                try:
                    # We pakken de x en y van de huidige agent uit de omgeving
                    idx_i = env.agent_ids.index(aid)
                    xi, yi = env.ax[idx_i], env.ay[idx_i]
                    
                    for other_id in active_agents:
                        if other_id == aid: continue
                        idx_j = env.agent_ids.index(other_id)
                        xj, yj = env.ax[idx_j], env.ay[idx_j]
                        
                        # Euclidische afstand in NM (als ax/ay in NM zijn)
                        dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                        if dist < min_dist:
                            min_dist = dist
                except:
                    # Fallback: als ax/ay niet werkt, schatten we het via de obs
                    # Dit gaat ervan uit dat je observatie [rel_x, rel_y, ...] is
                    pass

                # Labelen als conflict-actie
                if min_dist < 0.5:
                    conflict_heading_actions.append(heading_change)
            
            obs, _, _, _, _ = env.step(agent_actions)
            
    # --- ANALYSE ---
    all_acts = np.array(all_heading_actions)
    conf_acts = np.array(conflict_heading_actions)

    print("\n" + "="*40)
    print(f"📊 RESULTATEN VOOR DREMPEL {threshold}")
    print(f"Totaal samples: {len(all_acts)}")
    print(f"Conflict samples (<5NM): {len(conf_acts)}")
    print(f"Gem. heading-change (Totaal): {np.mean(all_acts):.4f}")
    print(f"Gem. heading-change (Conflict): {np.mean(conf_acts):.4f}")
    
    perc_above = (np.sum(all_acts > threshold) / len(all_acts)) * 100
    perc_conf_above = (np.sum(conf_acts > threshold) / len(conf_acts)) * 100
    
    print(f"Percentage van TOTAAL boven drempel: {perc_above:.2f}%")
    print(f"Percentage van CONFLICTEN boven drempel: {perc_conf_above:.2f}%")
    print("="*40)

    plt.figure(figsize=(10,6))
    plt.hist(all_acts, bins=50, alpha=0.5, label='Totaal (Log)', log=True, color='blue')
    plt.hist(conf_acts, bins=50, alpha=0.5, label='In Conflict (Log)', log=True, color='orange')
    plt.axvline(threshold, color='red', linestyle='--', label=f'Drempel {threshold}')
    plt.xlabel("Absolute Heading Change Action")
    plt.ylabel("Frequentie (Log-schaal)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    diagnostic_run()