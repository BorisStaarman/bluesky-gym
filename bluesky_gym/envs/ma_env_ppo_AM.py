from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gymnasium import spaces
import numpy as np
import pygame
import bluesky as bs
from bluesky_gym.envs.common.screen_dummy import ScreenDummy
import bluesky_gym.envs.common.functions as fn
import csv, os, shutil
from collections import defaultdict 
import time

# Add a base dir for metrics written by this env
# NOTE: When copying this folder to a new date/version, update this constant!
# Or better: pass metrics_base_dir via env_config in your training script
METRICS_BASE_DIR = "metrics"  # Fallback (not used when passed via env_config)
N_AGENTS = 20

# Configuration based on the paper
MAX_INTRUDERS = N_AGENTS -1  # Fixed buffer size to handle variable N
OWNSHIP_DIM = 3     # cos_drift, sin_drift, airspeed_norm
INTRUDER_DIM = 7    # x, y, vx, vy, cos, sin, dist

# scenario constants
POLY_AREA_RANGE = (0.75, 0.8)
CENTER = np.array([51.990426702297746, 4.376124857109851]) 

# aircraft constants
ALTITUDE = 360
AC_SPD = 9 # starting speed, in m/s !
AC_TYPE = "M600"
INTRUSION_DISTANCE = 0.054

# conversion factors
NM2KM = 1.852
MpS2Kt = 1.94384
FT2M = 0.3048

# model settings
ACTION_FREQUENCY = 1 # how many sim steps per action
NUM_AC_STATE = 3 # number of aircraft in observation vector
MAX_STEPS = 400 # max steps per episode

# penalties for reward
# DRIFT_PENALTY = -0.015  # Very small - detours should be cheap
# STEP_PENALTY = -0.005
# INTRUSION_PENALTY = -10.0  # Separation violation - penalty applied once per drone pair per episode
# WAYPOINT_REACHED_REWARD = 1.5  # Increased to encourage completion
DRIFT_PENALTY = -0.0015  # Very small - detours should be cheap
STEP_PENALTY = -0.0005
INTRUSION_PENALTY = -3.0  # Separation violation - penalty applied once per drone pair per episode
WAYPOINT_REACHED_REWARD = 0.15  # Increased to encourage completion
# waypoint rewards
WAYPOINT_RADIUS = 0.05 # in NM is about 90 meters

# Proximity shaping: apply a negative reward when within a soft band
# outside the hard intrusion distance. This gives agents early warning
# to avoid getting too close before hitting the termination boundary.
SOFT_INTRUSION_FACTOR = 2.0  # Wider soft band: 2x the intrusion distance (0.108 NM)
# At the closest non-intruding distance, penalty is 80% of intrusion penalty
# This creates strong deterrent without being as catastrophic as actual intrusion
PROXIMITY_MAX_PENALTY = 0.06 * abs(INTRUSION_PENALTY)  # -2.0 when INTRUSION_PENALTY is -2.5

# constants to control actions, 
D_HEADING = 45 # degrees
D_VELOCITY = 10/3 # knots

# normalization parameters
# For x_r, y_r: current max ~0.3, increase multiplier from 3.5 to 2.0 for better range usage (target ~0.6-0.8)
MAX_SCENARIO_DIM_M = (POLY_AREA_RANGE[1] + POLY_AREA_RANGE[0])/2 * NM2KM * 1000.0 * 2.0
# For distance: all negative means center is too high; reduce multiplier from 2.5 to 1.5
DISTANCE_CENTER_M = MAX_SCENARIO_DIM_M / 2.0
DISTANCE_SCALE_M = MAX_SCENARIO_DIM_M / 2.0 * 1.5
# For vx_r, vy_r: observed max ~1.42, increase from 50 to 75 m/s
MAX_RELATIVE_VEL_MS = 75.0
AIRSPEED_CENTER_KTS = 35.0
AIRSPEED_SCALE_KTS = 10.0 * 3.4221

# collision risk parameters
PROTECTED_ZONE_M = 100  # meters
CPA_TIME_HORIZON_S = 30 # seconds

# logging
LOG_EVERY_N = 100  # throttle repeated warnings



class SectorEnv(MultiAgentEnv):
    metadata = {"name": "ma_env", "render_modes": ["rgb_array", "human"], "render_fps": 10}

    def __init__(self, render_mode=None, n_agents=20, run_id="default",
                 debug_obs=False, debug_obs_episodes=2, debug_obs_interval=1, debug_obs_agents=None,
                 collect_obs_stats=False, print_obs_stats_per_episode=False,
                 intrusion_penalty=None, metrics_base_dir=None):
        super().__init__()
        self.render_mode = render_mode
        # Use provided intrusion penalty or default from module constant
        self.intrusion_penalty = intrusion_penalty if intrusion_penalty is not None else INTRUSION_PENALTY
        # Calculate proximity penalty based on actual intrusion penalty being used
        self.proximity_max_penalty = PROXIMITY_MAX_PENALTY
        self.num_ac = n_agents
        self.window_width = 512 
        self.window_height = 512 
        self.window_size = (self.window_width, self.window_height)
        self.poly_name = 'airspace'
        self._agent_ids = {f'kl00{i+1}'.upper() for i in range(n_agents)}
        self.agents = []
        
        # debug/inspection controls for observation vectors
        self.debug_obs = bool(debug_obs)
        self.debug_obs_episodes = int(debug_obs_episodes)
        self.debug_obs_interval = max(1, int(debug_obs_interval))
        # None or iterable of agent ids to print; if None, print all
        self.debug_obs_agents = set(debug_obs_agents) if debug_obs_agents else None
        self._episode_index = 0
        self._env_step = 0
        
        # observation statistics (for proper normalization analysis)
        self._obs_stats_enabled = bool(collect_obs_stats)
        self._print_obs_stats_per_episode = bool(print_obs_stats_per_episode)
        self._obs_stats = {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,  # Welford online mean
            "M2": None,    # Welford running variance accumulator
        }
        self.observation_space = spaces.Dict({
            # The agent's own state
            "ownship": spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(OWNSHIP_DIM,), 
                dtype=np.float32
            ),
            
            # The matrix of intruders (Fixed size N, but we fill unused spots with 0)
            "intruders": spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(MAX_INTRUDERS, INTRUDER_DIM), 
                dtype=np.float32
            ),
            
            # A mask to tell the network which rows are real data (1) and which are padding (0)
            "mask": spaces.Box(
                low=0,
                high=1,
                shape=(MAX_INTRUDERS,),
                dtype=np.float32
            )})
    
        # for multi agent csv files
        # --- per-agent CSV logging (safe for multi-worker) ---
        self.run_id = run_id or "default"
        self._flush_threshold = 25  # tune if you want more/less frequent writes
        self._agent_buffers = defaultdict(list)
        self._agent_episode_index = defaultdict(int)
        
        # Use provided metrics base dir or fall back to module constant
        metrics_base = metrics_base_dir if metrics_base_dir is not None else METRICS_BASE_DIR
        pid = os.getpid()
        self._metrics_root = os.path.join(metrics_base, f"run_{self.run_id}", f"pid_{pid}")
        
        # wipe ONLY this PID folder once per process, then append during the run
        # Use a per-instance check instead of class-level to avoid cross-contamination
        if not hasattr(self, "_metrics_dir_cleared"):
            if os.path.exists(self._metrics_root):
                shutil.rmtree(self._metrics_root)
            self._metrics_dir_cleared = True

        os.makedirs(self._metrics_root, exist_ok=True)
        print(f"[metrics] writing to: {os.path.abspath(self._metrics_root)}")
        

        # for evaluations
        self.total_intrusions = 0

        # stuff for making csv file
        self._agent_steps = {}
        self._rewards_acc = {}                 # per-agent running sums during the episode
        self._rewards_counts = {}              # per-agent step counts for safe averaging
        self._intrusions_acc = {}              # per-agent intrusion counts during the episode
        
        single_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3 + 7 * NUM_AC_STATE,), dtype=np.float32)
        single_action_space = spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
        
        self.observation_space = spaces.Dict({agent_id: single_obs_space for agent_id in self._agent_ids})
        self.action_space = spaces.Dict({agent_id: single_action_space for agent_id in self._agent_ids})
        
        bs.init(mode='sim', detached=True)
        bs.scr = ScreenDummy()
        bs.stack.stack('DT 1')
        
        self.window = None
        self.clock = None
        
        self.agent_waypoints = {}
        self.previous_distances = {}
        self.waypoint_reached_agents = set()

    def reset(self, *, seed=None, options=None):
        bs.traf.reset()
        bs.tools.areafilter.deleteArea(self.poly_name)
        self.agents = sorted(list(self._agent_ids))
        # episode counters
        self._episode_index += 1
        self._env_step = 0
        # for savin  data in csv file
        self._agent_steps = {a: 0 for a in self.agents}
        self._rewards_acc = {a: {"drift": 0.0, "progress": 0.0, "intrusion": 0.0, "proximity": 0.0} for a in self.agents}
        self._rewards_counts = {a: 0 for a in self.agents}
        self._intrusions_acc = {a: 0 for a in self.agents}  # Track intrusions per agent per episode
        # Track which drone pairs have already received intrusion penalty this episode
        self._penalized_pairs = set()  # Store tuples of (agent1, agent2) where agent1 < agent2
        self._pairs_penalized_this_step = set()  # Temporary buffer for pairs penalized in current step
        # for evaluation
        self.total_intrusions = 0
        
        self.previous_distances = {}
        self.waypoint_reached_agents = set()
        
        self._generate_polygon()
        self._generate_waypoints()
        self._generate_ac()

        for agent in self.agents:
            try:
                ac_idx = bs.traf.id2idx(agent)
                wpt_lat, wpt_lon = self.agent_waypoints[agent]
                _, dist_nm = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], wpt_lat, wpt_lon)
                self.previous_distances[agent] = dist_nm
            except Exception:
                pass

        if self.render_mode == "human":
            self._render_frame()

        observations, _, _ = self._get_observation(self.agents)
        self._update_obs_stats(observations)
        self._maybe_print_observations(observations, when="reset")
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions):
        agents_in_step = list(self.agents)
        
        # Clear the temporary buffer for this step (symmetric penalty tracking)
        self._pairs_penalized_this_step = set()
        
        self._do_action(actions)
        for _ in range(ACTION_FREQUENCY):
            bs.sim.step()

        if self.render_mode == "human": 
            self._render_frame()
        for agent in agents_in_step:
            if agent in self._agent_steps:
                self._agent_steps[agent] += 1

        observations, _, _ = self._get_observation(agents_in_step)
        self._update_obs_stats(observations)
        self._env_step += 1
        self._maybe_print_observations(observations, when="step")
        rewards, infos = self._get_reward(agents_in_step)
        
        # After computing all rewards, mark newly penalized pairs as permanently penalized
        self._penalized_pairs.update(self._pairs_penalized_this_step)
        
        terminateds = self._get_terminateds(agents_in_step)
        truncateds = self._get_truncateds(agents_in_step)
        
        agents_to_remove = {agent for agent in agents_in_step if terminateds.get(agent, False) or truncateds.get(agent, False)}
        for a in agents_to_remove:
            n = max(1, self._rewards_counts.get(a, 0))
            m_drift    = self._rewards_acc.get(a, {}).get("drift", 0.0)    / n
            m_progress = self._rewards_acc.get(a, {}).get("progress", 0.0) / n
            m_intr     = self._rewards_acc.get(a, {}).get("intrusion", 0.0)/ n
            m_prox     = self._rewards_acc.get(a, {}).get("proximity", 0.0)/ n

            # increment per-agent episode index
            self._agent_episode_index[a] += 1
            
            # Determine termination reason
            waypoint_reached = a in self.waypoint_reached_agents
            truncated = truncateds.get(a, False)

            # buffer one row for THIS agent only
            self._agent_buffers[a].append({
                "episode_index": self._agent_episode_index[a],
                "steps": self._agent_steps.get(a, 0),
                "mean_reward_drift":    m_drift,
                "mean_reward_progress": m_progress,
                "mean_reward_intrusion":m_intr,
                "mean_reward_proximity":m_prox,
                "sum_reward_drift":     self._rewards_acc.get(a, {}).get("drift", 0.0),
                "sum_reward_progress":  self._rewards_acc.get(a, {}).get("progress", 0.0),
                "sum_reward_intrusion": self._rewards_acc.get(a, {}).get("intrusion", 0.0),
                "sum_reward_proximity": self._rewards_acc.get(a, {}).get("proximity", 0.0),
                "total_intrusions":     self._intrusions_acc.get(a, 0),
                "terminated_waypoint": waypoint_reached,  # True if agent reached waypoint
                "truncated": truncated,  # True if episode truncated (out of bounds or time limit)
                "finished_at": time.time(),  
            })

            # flush this agent's buffer if threshold reached
            if len(self._agent_buffers[a]) >= self._flush_threshold:
                self._flush_agent_buffer(a)
       
        self.agents = [agent for agent in self.agents if agent not in agents_to_remove]
        
        all_done = len(self.agents) == 0
        terminateds["__all__"] = all_done
        truncateds["__all__"] = all_done
        
        if all_done:
            # optional: print a compact obs stats summary for this run so far
            if self._print_obs_stats_per_episode:
                self._print_obs_stats_summary()
            # write obs stats CSV at episode end so it's available even if close() isn't called
            try:
                self._write_obs_stats_csv()
            except Exception as e:
                print(f"[obs-stats] failed to write CSV at episode end: {e}")
            for a in agents_to_remove:
                self._flush_agent_buffer(a)
            
        return observations, rewards, terminateds, truncateds, infos

    def _maybe_print_observations(self, observations, when="step"):
        """Optionally print full observation vectors for inspection.
        Controls:
          - self.debug_obs: master switch
          - self.debug_obs_episodes: only print for first N episodes
          - self.debug_obs_interval: print every K env steps
          - self.debug_obs_agents: None for all, or subset of agent IDs
        """
        if not self.debug_obs:
            return
        if self._episode_index > self.debug_obs_episodes:
            return
        if when == "step" and (self._env_step % self.debug_obs_interval) != 0:
            return

        header = f"[obs] episode={self._episode_index} when={when} step={self._env_step}"
        print(header)
        for agent_id, vec in observations.items():
            if self.debug_obs_agents is not None and agent_id not in self.debug_obs_agents:
                continue
            try:
                arr = np.asarray(vec).ravel()
                # format compactly with 4 decimals
                formatted = ", ".join(f"{x:.4f}" for x in arr)
                print(f"  agent={agent_id} len={arr.size} -> [" + formatted + "]")
            except Exception as e:
                print(f"  agent={agent_id} <error formatting observation: {e}>")
    def _flush_agent_buffer(self, agent_id):
        """Append buffered rows for a single agent to its CSV."""
        rows = self._agent_buffers.get(agent_id, [])
        if not rows:
            return
        path = os.path.join(self._metrics_root, f"{agent_id}.csv")
        write_header = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "episode_index",    # monotonically increasing per agent
                    "steps",
                    "mean_reward_drift",
                    "mean_reward_progress",
                    "mean_reward_intrusion",
                    "mean_reward_proximity",
                    "sum_reward_drift",
                    "sum_reward_progress",
                    "sum_reward_intrusion",
                    "sum_reward_proximity",
                    "total_intrusions",
                    "terminated_waypoint",   # True if agent reached waypoint
                    "truncated",             # True if truncated (out of bounds/time limit)
                    "finished_at",
                ],
            )
            if write_header:
                w.writeheader()
            for row in rows:
                w.writerow(row)
        self._agent_buffers[agent_id].clear()

    def _flush_episode_metrics_to_csv(self):
        
        write_header = not os.path.exists(self._metrics_csv_path)
        with open(self._metrics_csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "agent","steps",
                "mean_reward_drift","mean_reward_progress","mean_reward_intrusion",
                "sum_reward_drift","sum_reward_progress","sum_reward_intrusion","finished_at",
            ])
            if write_header:
                w.writeheader()
            for row in self._episode_metrics:
                w.writerow(row)
        self._episode_metrics.clear()
    
    def _get_reward(self, active_agents):
        rewards = {}
        infos = {agent: {} for agent in active_agents}
        
        for agent in active_agents:
            try:
                ac_idx = bs.traf.id2idx(agent)
                
                drift_reward = self._check_drift(agent, ac_idx)
                intrusion_reward, agent_intrusion = self._check_intrusion(agent, ac_idx)
                proximity_reward = self._check_proximity(agent, ac_idx)
                progress_reward = self._check_progress(agent, ac_idx)
                
                rewards[agent] = drift_reward + intrusion_reward + proximity_reward + progress_reward  + STEP_PENALTY
                
                # accumulate for per-episode stats
                self._rewards_acc[agent]["drift"]     += float(drift_reward)
                self._rewards_acc[agent]["progress"]  += float(progress_reward)
                self._rewards_acc[agent]["intrusion"] += float(intrusion_reward)
                self._rewards_acc[agent]["proximity"] += float(proximity_reward)
                self._rewards_counts[agent]          += 1
                
                infos[agent]["reward_drift"] = drift_reward
                infos[agent]["reward_intrusion"] = intrusion_reward
                infos[agent]["reward_proximity"] = proximity_reward
                infos[agent]["reward_progress"] = progress_reward
                infos[agent]["intrusion"] = agent_intrusion
                
            except Exception as e:
                print(f"Error calculating reward for agent {agent}: {e}")
                rewards[agent] = 0
                infos[agent] = {}
        
        return rewards, infos
    
    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.agents = []
                    return

        max_distance = max(np.linalg.norm(p1 - p2) for p1 in self.poly_points for p2 in self.poly_points) * NM2KM
        px_per_km = self.window_width / max_distance
        canvas = pygame.Surface(self.window_size)
        canvas.fill((135, 206, 235))
        coords = [((self.window_width / 2) + p[0] * NM2KM * px_per_km, (self.window_height / 2) - p[1] * NM2KM * px_per_km) for p in self.poly_points]
        pygame.draw.polygon(canvas, (255, 0, 0), coords, width=2)

        # Get risk levels and most risky neighbor info
        _, risk_levels, most_risky = self._get_observation(self.agents)

        # Precompute positions for all agents
        agent_positions = {}
        for agent in self.agents:
            try:
                ac_idx = bs.traf.id2idx(agent)
                ac_qdr, ac_dis = bs.tools.geo.kwikqdrdist(CENTER[0], CENTER[1], bs.traf.lat[ac_idx], bs.traf.lon[ac_idx])
                x_pos = (self.window_width / 2) + (np.cos(np.deg2rad(ac_qdr)) * (ac_dis * NM2KM) * px_per_km)
                y_pos = (self.window_height / 2) - (np.sin(np.deg2rad(ac_qdr)) * (ac_dis * NM2KM) * px_per_km)
                agent_positions[agent] = (x_pos, y_pos)
            except Exception:
                continue

        # Draw lines to the top 3 most risky neighbors for agent 'KL001'
        agent1 = 'KL001'
        if agent1 in self.agents and agent1 in agent_positions:
            # Recompute candidates for agent1 (same as in _get_observation)
            try:
                ac_idx = bs.traf.id2idx(agent1)
                ac_loc = fn.latlong_to_nm(CENTER, np.array([bs.traf.lat[ac_idx], bs.traf.lon[ac_idx]])) * NM2KM * 1000
                vx = np.cos(np.deg2rad(bs.traf.hdg[ac_idx])) * bs.traf.gs[ac_idx]
                vy = np.sin(np.deg2rad(bs.traf.hdg[ac_idx])) * bs.traf.gs[ac_idx]
                candidates = []
                for i in range(self.num_ac):
                    if i == ac_idx:
                        continue
                    other_agent_id = self.agents[i] if i < len(self.agents) else None
                    int_hdg = bs.traf.hdg[i]
                    int_loc = fn.latlong_to_nm(CENTER, np.array([bs.traf.lat[i], bs.traf.lon[i]])) * NM2KM * 1000
                    dx = float(int_loc[0] - ac_loc[0])
                    dy = float(int_loc[1] - ac_loc[1])
                    vxi = np.cos(np.deg2rad(int_hdg)) * bs.traf.gs[i]
                    vyi = np.sin(np.deg2rad(int_hdg)) * bs.traf.gs[i]
                    dvx = float(vxi - vx)
                    dvy = float(vyi - vy)
                    risk = SectorEnv._cpa_risk(dx, dy, dvx, dvy)
                    candidates.append((risk, other_agent_id))
                # Sort by descending risk
                candidates.sort(key=lambda t: -t[0])
                top3 = [c for c in candidates[:3] if c[1] is not None and c[1] in agent_positions]
                line_colors = [(255,0,0), (255,140,0), (255,255,0)]  # red, orange, yellow
                for idx, (risk, neighbor_id) in enumerate(top3):
                    start = agent_positions[agent1]
                    end = agent_positions[neighbor_id]
                    color = line_colors[idx] if idx < len(line_colors) else (128,128,128)
                    pygame.draw.line(canvas, color, start, end, width=2)
            except Exception:
                pass

        # Draw aircraft, color by risk, and display risk value
        font = pygame.font.SysFont(None, 18)
        for agent in self.agents:
            try:
                ac_idx = bs.traf.id2idx(agent)
                ac_hdg = bs.traf.hdg[ac_idx]
                pos = agent_positions.get(agent, None)
                if pos is None:
                    continue
                risk_val = risk_levels.get(agent, 0.0)
                # Color: green for KL001, others red scaled by risk
                if agent == "KL001":
                    color = (0, 255, 0)
                else:
                    red_intensity = int(100 + 155 * min(1.0, risk_val))
                    color = (red_intensity, 0, 0)
                # Draw heading line
                heading_end_x = np.cos(np.deg2rad(ac_hdg)) * 10
                heading_end_y = np.sin(np.deg2rad(ac_hdg)) * 10
                pygame.draw.line(canvas, (0, 0, 0), pos, (pos[0] + heading_end_x, pos[1] - heading_end_y), width=4)
                # Draw aircraft circle
                pygame.draw.circle(canvas, color, (int(pos[0]), int(pos[1])), int(INTRUSION_DISTANCE * NM2KM * px_per_km / 2), width=2)
                # Draw risk value as text
                risk_text = f"{risk_val:.2f}"
                text_surf = font.render(risk_text, True, (0, 0, 0))
                canvas.blit(text_surf, (pos[0] + 8, pos[1] - 8))
            except Exception:
                continue

        self.window.blit(canvas, canvas.get_rect())
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
    # flush any leftovers for all agents
        if hasattr(self, "_agent_buffers"):
            for a in list(self._agent_buffers.keys()):
                self._flush_agent_buffer(a)
        # write observation stats CSV once
        try:
            self._write_obs_stats_csv()
        except Exception as e:
            print(f"[obs-stats] failed to write CSV: {e}")
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            
    
    @staticmethod
    def _cpa_risk(dx, dy, dvx, dvy, R=PROTECTED_ZONE_M, T=CPA_TIME_HORIZON_S, 
            k=0.25, w_d=0.8, w_t=0.1, w_c=0.1, diverge_penalty=0.2):
        EPS = 1e-6  # to avoid div by zero

        # Step 1: Compute relative position and velocity
        rv = dx * dvx + dy * dvy  # relative position dot relative velocity
        v2 = dvx * dvx + dvy * dvy  # relative speed squared
        r2 = dx * dx + dy * dy  # relative distance squared

        # Step 2: Handle case with no relative motion (v2 < EPS)
        if v2 < EPS:
            t_cpa = 0.0  # no CPA
            d_cpa = float(np.hypot(dx, dy))  # Euclidean distance
            approaching = False  # not moving toward each other
            speed_mag = 0.0  # no relative speed
        else:
            speed_mag = float(np.sqrt(v2))  # speed magnitude
            t_cpa = -rv / v2  # time to CPA
            t_cpa = float(np.clip(t_cpa, 0.0, T))  # clip to [0, T]
            d_cpa = float(np.hypot(dx + dvx * t_cpa, dy + dvy * t_cpa))  # distance at CPA
            approaching = (rv < 0.0)  # moving toward each other

        # Step 3: Calculate the time urgency factor (0 at horizon, 1 now)
        time_factor = 1.0 - (t_cpa / (T + EPS))

        # Step 4: Distance factor using a sigmoid function centered at R (protected zone)
        soft = max(EPS, k * R)  # avoid div-by-zero
        dist_factor = 1.0 / (1.0 + np.exp((d_cpa - R) / soft))  # smooth sigmoid decay for distance

        # Step 5: Closing speed factor (normalized), only counts if approaching
        if r2 < EPS or v2 < EPS:
            closing_norm = 0.0
        else:
            closing_speed = max(0.0, -rv / (np.sqrt(r2) + EPS))  # m/s toward each other
            closing_norm = closing_speed / (speed_mag + EPS)  # normalize to ~0..1

        # Step 6: True collision check (optional, solve the quadratic for potential collision)
        hit_bonus = 0.0
        if v2 >= EPS:
            a = v2
            b = 2.0 * rv
            c = r2 - R * R
            disc = b * b - 4 * a * c
            if disc >= 0.0:
                s = float(np.sqrt(disc))
                t1 = (-b - s) / (2 * a)
                t2 = (-b + s) / (2 * a)
                # first future intersection within horizon (if any)
                candidates = [t for t in (t1, t2) if 0.0 <= t <= T]
                if candidates:
                    t_hit = min(candidates)
                    # bonus scales with how soon the hit occurs
                    hit_bonus = 0.2 * (1.0 - t_hit / (T + EPS))  # 0..0.2

        # Step 7: Distance penalty - closer agents get a higher penalty
        current_distance = float(np.sqrt(r2))  # current separation distance
        distance_penalty = min(1.0, current_distance / (2 * R))  # 0 at contact, 1 at 4*R+

        # Step 8: Combine the factors (weighted sum)
        risk = (w_d * float(dist_factor)
                + w_t * float(time_factor)
                + w_c * float(closing_norm)
                + float(hit_bonus))

        # Step 9: Apply the distance penalty - closer threats have higher priority
        risk *= (1.0 - 0.3 * distance_penalty)  # reduce risk for distant threats

        # Step 10: Penalize if agents are diverging (moving away from each other)
        if not approaching:
            risk *= diverge_penalty  # down-weight if moving apart

        # Step 11: Clamp risk to [0, 1]
        return float(max(0.0, min(1.0, risk)))  # Return the normalized risk value between 0 and 1

            

    def _get_observation(self, active_agents):
        # CONFIG (Match these to your training setup)
        # Increase this! Attention handles variable sizes, so give it a large buffer 
        # (e.g., 30 or 50) instead of just the top few.
                
        obs = {}
        risk_levels = {}
        most_risky = {}

        for agent in active_agents:
            try:
                ac_idx = bs.traf.id2idx(agent)

                # --- 1. Ownship Features (Same as before) ---
                wpt_lat, wpt_lon = self.agent_waypoints[agent]
                wpt_qdr, _ = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], wpt_lat, wpt_lon)
                ac_hdg = bs.traf.hdg[ac_idx]
                drift = fn.bound_angle_positive_negative_180(ac_hdg - wpt_qdr)
                
                # Normalization
                airspeed_kts = bs.traf.tas[ac_idx] * MpS2Kt
                airspeed_norm = (airspeed_kts - AIRSPEED_CENTER_KTS) / AIRSPEED_SCALE_KTS
                cos_drift = np.cos(np.deg2rad(drift))
                sin_drift = np.sin(np.deg2rad(drift))

                # Create Ownship Vector (Shape: [3])
                ownship_vec = np.array([cos_drift, sin_drift, airspeed_norm], dtype=np.float32)

                # --- 2. Candidate Computation (Mostly same as before) ---
                vx = np.cos(np.deg2rad(ac_hdg)) * bs.traf.gs[ac_idx]
                vy = np.sin(np.deg2rad(ac_hdg)) * bs.traf.gs[ac_idx]
                ac_loc = fn.latlong_to_nm(CENTER, np.array([bs.traf.lat[ac_idx], bs.traf.lon[ac_idx]])) * NM2KM * 1000

                candidates = []
                for i in range(self.num_ac):
                    if i == ac_idx: continue
                    
                    int_hdg = bs.traf.hdg[i]
                    int_loc = fn.latlong_to_nm(CENTER, np.array([bs.traf.lat[i], bs.traf.lon[i]])) * NM2KM * 1000
                    dx = float(int_loc[0] - ac_loc[0])
                    dy = float(int_loc[1] - ac_loc[1])
                    vxi = np.cos(np.deg2rad(int_hdg)) * bs.traf.gs[i]
                    vyi = np.sin(np.deg2rad(int_hdg)) * bs.traf.gs[i]
                    dvx = float(vxi - vx)
                    dvy = float(vyi - vy)
                    trk = np.arctan2(dvy, dvx)
                    d_now = float(np.hypot(dx, dy))
                    # ... (Your existing candidate logic for dx, dy, dvx, dvy, risk, etc.) ...
                    # Copy-paste your existing logic here for calculating dx, dy, risk, etc.
                    # For brevity, I assume you calculate: risk, x_rn, y_rn, vx_rn, vy_rn, cos_trk, sin_trk, dist_n
                    
                    # Append the tuple of normalized features
                    # Note: We append the FEATURES directly, not just raw values, to save time later
                    candidates.append((dx, dy, dvx, dvy, np.cos(trk), np.sin(trk), d_now))


                # Optional: You can still sort if you want, but Attention handles unsorted data well.
                # Sorting helps convergence speed slightly by keeping "important" agents at the top.
                candidates.sort(key=lambda x: -x[0]) # Sort by risk descending

                # --- 3. Constructing the Structured Output ---
                
                # Prepare empty arrays (Padding with Zeros)
                intruder_matrix = np.zeros((MAX_INTRUDERS, INTRUDER_DIM), dtype=np.float32)
                mask_vec = np.zeros(MAX_INTRUDERS, dtype=np.float32)

                # Fill the arrays with actual data
                # We take up to MAX_INTRUDERS (e.g., 30). If we have fewer, the rest stay 0.
                num_visible = min(len(candidates), MAX_INTRUDERS)
                
                for i in range(num_visible):
                    intruder_matrix[i] = np.array(candidates[i], dtype=np.float32)
                    mask_vec[i] = 1.0  # Mark this row as "Real Data"

                # --- 4. Final Dictionary Output ---
                obs[agent] = {
                    "ownship": ownship_vec,      # Shape (3,)
                    "intruders": intruder_matrix, # Shape (30, 8)
                    "mask": mask_vec             # Shape (30,)
                }

                # Handle your risk tracking logic (unchanged)
                if candidates:
                    risk_levels[agent] = candidates[0][0] # First item in first candidate
                    # (You might need to adjust indices depending on how you stored IDs in candidates)
                else:
                    risk_levels[agent] = 0.0

            except Exception as e:
                # Fallback for errors
                obs[agent] = {
                    "ownship": np.zeros(OWNSHIP_DIM, dtype=np.float32),
                    "intruders": np.zeros((MAX_INTRUDERS, INTRUDER_DIM), dtype=np.float32),
                    "mask": np.zeros(MAX_INTRUDERS, dtype=np.float32)
                }

        return obs, risk_levels, most_risky

    # ---- Observation statistics helpers ----
    def _get_obs_feature_names(self):
        dim = 3 + 8 * NUM_AC_STATE
        names = []
        names.extend(["cos_drift", "sin_drift", "airspeed_norm"])
        # Then 8 blocks each of size NUM_AC_STATE in the order we concatenate
        blocks = [
            ("risk", NUM_AC_STATE),
            ("x_r", NUM_AC_STATE),
            ("y_r", NUM_AC_STATE),
            ("vx_r", NUM_AC_STATE),
            ("vy_r", NUM_AC_STATE),
            ("cos_trk", NUM_AC_STATE),
            ("sin_trk", NUM_AC_STATE),
            ("dist_n", NUM_AC_STATE),
        ]
        for label, count in blocks:
            for i in range(count):
                names.append(f"{label}_{i}")
        assert len(names) == dim, f"feature naming mismatch: {len(names)} != {dim}"
        return names

    def _update_obs_stats(self, observations):
        if not self._obs_stats_enabled:
            return
        if not observations:
            return
        for vec in observations.values():
            x = np.asarray(vec, dtype=float).ravel()
            s = self._obs_stats
            if s["min"] is None:
                # initialize arrays
                s["min"] = x.copy()
                s["max"] = x.copy()
                s["mean"] = np.zeros_like(x)
                s["M2"] = np.zeros_like(x)
                s["count"] = 0
            # elementwise min/max
            s["min"] = np.minimum(s["min"], x)
            s["max"] = np.maximum(s["max"], x)
            # Welford online mean/variance (vectorized)
            s["count"] += 1
            n = s["count"]
            delta = x - s["mean"]
            s["mean"] = s["mean"] + delta / n
            delta2 = x - s["mean"]
            s["M2"] = s["M2"] + delta * delta2

    def _print_obs_stats_summary(self):
        if not self._obs_stats_enabled or self._obs_stats.get("count", 0) == 0:
            return
        s = self._obs_stats
        n = s["count"]
        mean = s["mean"]
        std = np.sqrt(np.maximum(0.0, s["M2"] / max(1, n - 1)))
        names = self._get_obs_feature_names()
        print(f"[obs-stats] samples={n}")
        # Print a compact grouped view
        def pr_range(label, idxs):
            mn = s["min"][idxs]
            mx = s["max"][idxs]
            mu = mean[idxs]
            sd = std[idxs]
            # summarize with min(min), max(max), mean(mean), mean(std)
            print(f"  {label:10s}  min=[{mn.min(): .3f}]  max=[{mx.max(): .3f}]  mean=[{mu.mean(): .3f}]  std~[{sd.mean(): .3f}]")
        # indices per block
        i0 = 0
        pr_range("cos_drift", slice(i0, i0+1)); i0 += 1
        pr_range("sin_drift", slice(i0, i0+1)); i0 += 1
        pr_range("airspeed", slice(i0, i0+1)); i0 += 1
        # 8 blocks of NUM_AC_STATE
        labels = ["risk", "x_r", "y_r", "vx_r", "vy_r", "cos_trk", "sin_trk", "dist_n"]
        for lab in labels:
            pr_range(lab, slice(i0, i0+NUM_AC_STATE))
            i0 += NUM_AC_STATE

    def _write_obs_stats_csv(self):
        if not self._obs_stats_enabled or self._obs_stats.get("count", 0) == 0:
            return
        # write one CSV per run/pid
        path = os.path.join(self._metrics_root, "obs_stats.csv")
        s = self._obs_stats
        n = s["count"]
        mean = s["mean"]
        std = np.sqrt(np.maximum(0.0, s["M2"] / max(1, n - 1)))
        names = self._get_obs_feature_names()
        write_header = not os.path.exists(path)
        
        # Silent write - only print on first write for this env
        if write_header:
            print(f"[obs-stats] Creating CSV: {os.path.abspath(path)}")
        
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["feature", "index", "min", "max", "mean", "std", "samples", "episode_index"])
            if write_header:
                w.writeheader()
            for i, name in enumerate(names):
                w.writerow({
                    "feature": name,
                    "index": i,
                    "min": float(s["min"][i]),
                    "max": float(s["max"][i]),
                    "mean": float(mean[i]),
                    "std": float(std[i]),
                    "samples": int(n),
                    "episode_index": int(self._episode_index),
                })
    
    def _get_terminateds(self, active_agents):
        # Only terminate agents that reached their waypoint
        # Intrusions no longer cause episode termination
        return {agent: agent in self.waypoint_reached_agents for agent in active_agents}
    
    def _get_truncateds(self, active_agents):
        truncateds = {}
        for agent in active_agents:
            try:
                ac_idx = bs.traf.id2idx(agent)
                outside = not bs.tools.areafilter.checkInside(
                    self.poly_name,
                    np.array([bs.traf.lat[ac_idx]]),
                    np.array([bs.traf.lon[ac_idx]]),
                    np.array([ALTITUDE * FT2M]),
                )
            except Exception:
                outside = True

            hit_time_limit = self._agent_steps.get(agent, 0) >= MAX_STEPS
            truncateds[agent] = outside or hit_time_limit
        return truncateds
    
    def _check_progress(self, agent_id, ac_idx):
        wpt_lat, wpt_lon = self.agent_waypoints[agent_id]
        _, current_dist = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], wpt_lat, wpt_lon)
        
        # reward for reaching waypoint
        if current_dist < WAYPOINT_RADIUS:
            if agent_id not in self.waypoint_reached_agents:
                self.waypoint_reached_agents.add(agent_id)
                return WAYPOINT_REACHED_REWARD
            else:
                return 0.0
        
        # Dense reward: reward for getting closer to waypoint each step
        # Use previous_distances (already tracked for other purposes)
        prev_dist = self.previous_distances.get(agent_id, current_dist)
        distance_improvement = prev_dist - current_dist  # Positive if getting closer
        self.previous_distances[agent_id] = current_dist
        
        # Small reward proportional to progress; scale slightly higher to better offset sparse penalties
        # Typical distance improvement per step is ~0.001-0.01 NM; scale by 2.0 => ~0.002-0.02 per step
        progress_reward = distance_improvement 
        
        return progress_reward

    def _do_action(self, actions):
        for agent, action in actions.items():
            try: 
                ac_idx = bs.traf.id2idx(agent)
                dh = action[0] * D_HEADING
                dv = action[1] * D_VELOCITY
                heading_new = fn.bound_angle_positive_negative_180(bs.traf.hdg[ac_idx] + dh)
                speed_new = (bs.traf.tas[ac_idx] * MpS2Kt) + dv
                bs.stack.stack(f"HDG {agent} {heading_new}")
                bs.stack.stack(f"SPD {agent} {speed_new}")
            except Exception: continue
            
    def _generate_polygon(self):
        R = np.sqrt(POLY_AREA_RANGE[1] / np.pi)
        p = [fn.random_point_on_circle(R) for _ in range(3)]
        p = fn.sort_points_clockwise(p)
        while fn.polygon_area(p) < POLY_AREA_RANGE[0]: 
            p.append(fn.random_point_on_circle(R))
            p = fn.sort_points_clockwise(p)
        self.poly_points = np.array(p)
        p_latlong = [fn.nm_to_latlong(CENTER, point) for point in self.poly_points]
        points = [coord for point in p_latlong for coord in point]
        bs.tools.areafilter.defineArea(self.poly_name, 'POLY', points)

    def _generate_waypoints(self):
        edges, perim_tot = [], 0
        for i in range(len(self.poly_points)): 
            p1, p2 = self.poly_points[i], self.poly_points[(i + 1) % len(self.poly_points)]
            len_edge = fn.euclidean_distance(p1, p2)
            edges.append((p1, p2, len_edge))
            perim_tot += len_edge
        d_list = sorted([np.random.uniform(0, perim_tot) for _ in range(self.num_ac)])
        self.wpts_nm = []
        edge_idx, current_d = 0, 0
        for d in d_list:
            while edge_idx < len(edges) - 1 and d > current_d + edges[edge_idx][2]: 
                current_d += edges[edge_idx][2]
                edge_idx += 1
            edge, frac = edges[edge_idx], (d - current_d) / edges[edge_idx][2]
            self.wpts_nm.append(edge[0] + frac * (edge[1] - edge[0]))
    
    def _generate_ac(self):
        padding = 0.1
        min_x, max_x = self.poly_points[:, 0].min() + padding, self.poly_points[:, 0].max() - padding
        min_y, max_y = self.poly_points[:, 1].min() + padding, self.poly_points[:, 1].max() - padding

        init_p_latlong = []
        while len(init_p_latlong) < self.num_ac:
            p_nm = np.array([np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y)])
            p_latlong = fn.nm_to_latlong(CENTER, p_nm)
            if bs.tools.areafilter.checkInside(self.poly_name, np.array([p_latlong[0]]), np.array([p_latlong[1]]), np.array([ALTITUDE * FT2M])):
                init_p_latlong.append(p_latlong)

        self.agent_waypoints = {}
        wpts_latlon = [fn.nm_to_latlong(CENTER, p) for p in self.wpts_nm]
        
        sorted_agent_ids = sorted(list(self._agent_ids))

        for idx, agent_id in enumerate(sorted_agent_ids):
            self.agent_waypoints[agent_id] = wpts_latlon[idx]
            
            init_pos_agent = init_p_latlong[idx]
            hdg_agent = fn.get_hdg(init_pos_agent, self.agent_waypoints[agent_id])
            
            bs.traf.cre(
                agent_id,
                actype=AC_TYPE,
                aclat=init_pos_agent[0],
                aclon=init_pos_agent[1],
                achdg=hdg_agent,
                acspd=AC_SPD,
                acalt=ALTITUDE,
            )
    
    def _check_drift(self, agent_id, ac_idx):
        ac_hdg = bs.traf.hdg[ac_idx]
        wpt_lat, wpt_lon = self.agent_waypoints[agent_id]
        wpt_qdr, _ = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], wpt_lat, wpt_lon)
        drift = abs(np.deg2rad(fn.bound_angle_positive_negative_180(ac_hdg - wpt_qdr)))
        #self.average_drift = np.append(self.average_drift, drift)
        return drift * DRIFT_PENALTY
    
    def _check_proximity(self, agent_id, ac_idx):
        """Gentle shaping penalty when within a soft band above intrusion distance.

        - If distance >= soft_thresh: 0 penalty
        - If INTRUSION_DISTANCE <= distance < soft_thresh: linear penalty from
          0 down to -self.proximity_max_penalty as aircraft get closer to the hard boundary
        - If distance < INTRUSION_DISTANCE: do not add proximity shaping here
          (the hard intrusion penalty covers this case to avoid double-penalizing)
        """
        soft_thresh = SOFT_INTRUSION_FACTOR * INTRUSION_DISTANCE
        # Compute min distance to any other aircraft
        min_dist = np.inf
        for i in range(self.num_ac):
            if i == ac_idx:
                continue
            _, d = bs.tools.geo.kwikqdrdist(
                bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], bs.traf.lat[i], bs.traf.lon[i]
            )
            if d < min_dist:
                min_dist = d

        # Outside soft band: no shaping
        if min_dist >= soft_thresh:
            return 0.0

        # Inside hard intrusion: handled by _check_intrusion, so skip shaping here
        if min_dist < INTRUSION_DISTANCE:
            return 0.0

        # Linearly scale penalty within [INTRUSION_DISTANCE, soft_thresh)
        # ratio in [0..1): 0 near soft boundary, 1 near hard boundary
        band = soft_thresh - INTRUSION_DISTANCE
        ratio = (soft_thresh - min_dist) / band if band > 0 else 0.0
        ratio = np.clip(ratio, 0.0, 1.0)
        return -self.proximity_max_penalty * float(ratio)
    
    def _check_intrusion(self, agent_id, ac_idx):
        """Return intrusion penalty for this agent on this step, and intrusion flag.

        Each drone pair can only receive the intrusion penalty ONCE per episode.
        Both agents in the pair receive the penalty on the step when they first intrude.
        After that, subsequent intrusions between that same pair are ignored.
        
        Uses a two-level tracking system:
        - _pairs_penalized_this_step: ensures BOTH agents get penalty in the same step
        - _penalized_pairs: prevents the same pair from being penalized in future steps
        
        Returns:
            tuple: (step_penalty, intrusion_occurred)
                - step_penalty: float, the penalty for intrusions (only applied once per pair)
                - intrusion_occurred: bool, True if any intrusion was detected (new or old)
        """
        had_intrusion = False  # True if ANY intrusion detected (for info tracking)
        step_penalty = 0.0
        
        for i in range(self.num_ac):
            if i == ac_idx:
                continue
            
            _, int_dis = bs.tools.geo.kwikqdrdist(
                bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], bs.traf.lat[i], bs.traf.lon[i]
            )
            
            if int_dis < INTRUSION_DISTANCE:
                had_intrusion = True  # Intrusion detected (may or may not be penalized)
                
                # Get the other agent's ID
                other_agent_id = self.agents[i] if i < len(self.agents) else None
                if other_agent_id is None:
                    continue
                
                # Create a consistent pair identifier (sorted so order doesn't matter)
                pair = tuple(sorted([agent_id, other_agent_id]))
                
                # Check if this pair has already been penalized in a PREVIOUS step
                if pair in self._penalized_pairs:
                    # Already penalized in a previous step - no penalty
                    pass
                else:
                    # First time this episode for this pair
                    # Ensure BOTH agents get the penalty in THIS step:
                    if pair in self._pairs_penalized_this_step:
                        # Second agent of the pair - also gets penalty
                        step_penalty += self.intrusion_penalty
                        # Track per-agent intrusion count (only when penalty is applied)
                        if agent_id in self._intrusions_acc:
                            self._intrusions_acc[agent_id] += 1
                    else:
                        # First agent of the pair - gets penalty and marks the pair
                        self._pairs_penalized_this_step.add(pair)
                        step_penalty += self.intrusion_penalty
                        # Count unique intrusion pairs once (by the first agent to detect it)
                        self.total_intrusions += 1
                        # Track per-agent intrusion count (only when penalty is applied)
                        if agent_id in self._intrusions_acc:
                            self._intrusions_acc[agent_id] += 1

        return step_penalty, had_intrusion
