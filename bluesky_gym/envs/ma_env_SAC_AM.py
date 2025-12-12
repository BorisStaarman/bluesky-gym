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

N_AGENTS = 20

# scenario constants
POLY_AREA_RANGE = (0.75, 0.8)
# Default center (can be overridden in __init__)
CENTER = np.array([52.362566, 4.881444]) # new center from training scenario

# aircraft constants
ALTITUDE = 360
AC_SPD = 9 # starting speed, in m/s !
AC_TYPE = "M600"
INTRUSION_DISTANCE = 1 / 1852 * 100  # was 0.054
MIN_SPAWN_SEPARATION_NM = 1 / 1852 * 125  # dit is iets van 125 meter
COLLISION_DISTANCE = 1 / 1852 * 100  # 100 meters - collision threshold (terminates both aircraft)

# conversion factors
NM2KM = 1.852
MpS2Kt = 1.94384
FT2M = 0.3048

# model settings
ACTION_FREQUENCY = 1 # how many sim steps per action
NUM_AC_STATE = N_AGENTS-1 # number of aircraft in observation vector
MAX_STEPS = 400 # max steps per episode

# =========================== REWARD PENALTIES PARAMETERS ===========================
DRIFT_PENALTY = -0.003  # Small penalty for heading deviation
INTRUSION_PENALTY = -15.0  # Separation violation - penalty applied every timestep during intrusion
WAYPOINT_REACHED_REWARD = 2.0  # Reward for reaching waypoint
PROGRESS_REWARD_SCALE = 2.0  # Scale factor for distance-to-waypoint progress
PATH_EFFICIENCY_SCALE = 0.0  # Disabled (set to 0) - can re-enable later for experiments
BOUNDARY_VIOLATION_PENALTY = -2.0  # Penalty for leaving polygon boundary (not at waypoint)
STEP_PENALTY = -0.012  # Small penalty applied every step to encourage efficiency

# Proximity penalty parameters
SOFT_INTRUSION_FACTOR = 1.5  # Soft zone starts at 1.5x the intrusion distance (150m when intrusion is 100m)
PROXIMITY_MAX_PENALTY = -1.0  # Maximum penalty when at hard boundary (100m)

# constants to control actions, 
D_HEADING = 45 # degrees
D_VELOCITY = 10/3 # knots
# waypoint rewards
WAYPOINT_RADIUS = 0.05 # in NM is about 90 meters

# normalization parameters
# MAX_SCENARIO_DIM_M = (POLY_AREA_RANGE[1] + POLY_AREA_RANGE[0])/2 * NM2KM * 1000.0 * 2.0 # Old value
# new value from scenario calculatec 

#NORMALIZER LAT LON
MAX_LAT_LON = 0.013749

MAX_SCENARIO_DIM_M = 1870.0

# For distance: all negative means center is too high; reduce multiplier from 2.5 to 1.5
DISTANCE_CENTER_M = MAX_SCENARIO_DIM_M / 2.0
DISTANCE_SCALE_M = MAX_SCENARIO_DIM_M / 2.0 * 1.5
# For vx_r, vy_r: observed max ~1.42, increase from 50 to 75 m/s
MAX_RELATIVE_VEL_MS = 70.0 # TODO dit is knts ipv m/s
AIRSPEED_CENTER_KTS = 35.0
AIRSPEED_SCALE_KTS = 10.0 * 3.4221

# collision risk parameters
PROTECTED_ZONE_M = 105  # meters
CPA_TIME_HORIZON_S = 15 # seconds

# logging
LOG_EVERY_N = 100  # throttle repeated warnings

# Add a base dir for metrics written by this env
# NOTE: This is just a fallback - metrics_base_dir is passed via env_config from main.py
METRICS_BASE_DIR = "metrics"  # Fallback (not used when passed via env_config)

class SectorEnv(MultiAgentEnv):
    metadata = {"name": "ma_env", "render_modes": ["rgb_array", "human"], "render_fps": 10}

    def __init__(self, render_mode=None, n_agents=20, run_id="default",
                 debug_obs=False, debug_obs_episodes=2, debug_obs_interval=1, debug_obs_agents=None,
                 collect_obs_stats=False, print_obs_stats_per_episode=False,
                 intrusion_penalty=None, proximity_max_penalty=None, metrics_base_dir=None, center=None):
        super().__init__()
        
        self.render_mode = render_mode
        # Use provided intrusion penalty or default from module constant
        self.intrusion_penalty = intrusion_penalty if intrusion_penalty is not None else INTRUSION_PENALTY
        # Use provided proximity penalty or default from module constant
        self.proximity_max_penalty = proximity_max_penalty if proximity_max_penalty is not None else PROXIMITY_MAX_PENALTY
        self.num_ac = n_agents
        # Use provided center or default - this is the reference point for relative coordinates
        self.center = np.array(center) if center is not None else CENTER
        self.window_width = 512 
        self.window_height = 512 
        self.window_size = (self.window_width, self.window_height)
        self.poly_name = 'airspace'
        # Use zero-padding to ensure proper alphabetical sorting (KL0001, KL0002, ..., KL0020)
        # Width 4 supports up to 9999 agents
        self._agent_ids = {f'KL{str(i+1).zfill(4)}' for i in range(n_agents)}
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
        
        # for multi agent csv files
        # --- per-agent CSV logging (safe for multi-worker) ---
        self.run_id = run_id or "default"
        self._flush_threshold = 5  # tune if you want more/less frequent writes
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
        
        # Store attention weights for visualization
        self.attention_weights = {}
        # Store neighbor mapping for attention visualization (agent -> list of neighbor IDs sorted by distance)
        self.neighbor_mapping = {}

        # stuff for making csv file
        self._agent_steps = {}
        self._rewards_acc = {}                 # per-agent running sums during the episode
        self._rewards_counts = {}              # per-agent step counts for safe averaging
        self._intrusions_acc = {}              # per-agent intrusion counts during the episode
                
        # Define observation and action spaces for RLlib
        # Observation: 7 ownship features + 5 features per intruder
        # Ownship: cos_drift, sin_drift, airspeed, x, y, vx, vy
        # Intruder: distance, dx_rel, dy_rel, vx_rel, vy_rel
        single_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7 + 5 * NUM_AC_STATE,), dtype=np.float32)
        single_action_space = spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
        
        self.areas_km2 = []
        
        self.observation_space = single_obs_space
        self.action_space = single_action_space
        
        bs.init(mode='sim', detached=True)
        bs.scr = ScreenDummy()
        bs.stack.stack('DT 1')
        
        self.window = None
        self.clock = None
        
        self.agent_waypoints = {}
        self.previous_distances = {}
        self.waypoint_reached_agents = set()

    @staticmethod
    def compute_relative_position(center, lat, lon):
        """
        Helper method to convert absolute lat/lon to relative position in meters.
        This is useful when using a trained model for inference.
        
        Args:
            center: np.array([lat, lon]) - reference point for coordinate system
            lat: float - aircraft latitude
            lon: float - aircraft longitude
            
        Returns:
            np.array([x, y]) - relative position in meters from center
        """
        return fn.latlong_to_nm(center, np.array([lat, lon])) * NM2KM * 1000

    def reset(self, *, seed=None, options=None):
        bs.traf.reset()
        bs.tools.areafilter.deleteArea(self.poly_name)
        self.agents = sorted(list(self._agent_ids))
        # episode counters
        self._episode_index += 1
        self._env_step = 0
        # for savin  data in csv file
        self._agent_steps = {a: 0 for a in self.agents}
        self._rewards_acc = {a: {"drift": 0.0, "progress": 0.0, "intrusion": 0.0, "path_efficiency": 0.0, "boundary": 0.0, "step": 0.0, "proximity": 0.0} for a in self.agents}
        self._rewards_counts = {a: 0 for a in self.agents}
        self._intrusions_acc = {a: 0 for a in self.agents}  # Track intrusions per agent per episode
        # Track which drone pairs have already received intrusion penalty this episode
        self._penalized_pairs = set()  # Store tuples of (agent1, agent2) where agent1 < agent2
        self._pairs_penalized_this_step = set()  # Temporary buffer for pairs penalized in current step
        # Track collisions
        self.collided_agents = set()  # Agents that have collided and should be terminated
        # for evaluation
        self.total_intrusions = 0
        
        self.previous_distances = {}
        self.waypoint_reached_agents = set()
        
        # Reset start positions for path efficiency tracking
        self._agent_start_positions = {}
        
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

        observations = self._get_observation(self.agents)
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

        observations = self._get_observation(agents_in_step)
        self._update_obs_stats(observations)
        self._env_step += 1
        self._maybe_print_observations(observations, when="step")
        rewards, infos = self._get_reward(agents_in_step)
        
        # Note: _pairs_penalized_this_step is cleared at the start of each step
        # This allows the same pair to be penalized again in the next timestep
        
        # Check for collisions
        self._check_collisions(agents_in_step)
        
        terminateds = self._get_terminateds(agents_in_step)
        truncateds = self._get_truncateds(agents_in_step)
        
        agents_to_remove = {agent for agent in agents_in_step if terminateds.get(agent, False) or truncateds.get(agent, False)}
        for a in agents_to_remove:
            n = max(1, self._rewards_counts.get(a, 0))
            m_drift    = self._rewards_acc.get(a, {}).get("drift", 0.0)    / n
            m_progress = self._rewards_acc.get(a, {}).get("progress", 0.0) / n
            m_intr     = self._rewards_acc.get(a, {}).get("intrusion", 0.0)/ n
            m_path_eff = self._rewards_acc.get(a, {}).get("path_efficiency", 0.0)/ n
            m_boundary = self._rewards_acc.get(a, {}).get("boundary", 0.0)/ n
            m_step     = self._rewards_acc.get(a, {}).get("step", 0.0)/ n
            m_proximity = self._rewards_acc.get(a, {}).get("proximity", 0.0)/ n

            # increment per-agent episode index
            self._agent_episode_index[a] += 1
            
            # Determine termination reason
            waypoint_reached = a in self.waypoint_reached_agents
            collided = a in self.collided_agents
            truncated = truncateds.get(a, False)

            # buffer one row for THIS agent only
            self._agent_buffers[a].append({
                "episode_index": self._agent_episode_index[a],
                "steps": self._agent_steps.get(a, 0),
                "mean_reward_drift":    m_drift,
                "mean_reward_progress": m_progress,
                "mean_reward_intrusion":m_intr,
                "mean_reward_path_efficiency": m_path_eff,
                "mean_reward_boundary": m_boundary,
                "mean_reward_step": m_step,
                "mean_reward_proximity": m_proximity,
                "sum_reward_drift":     self._rewards_acc.get(a, {}).get("drift", 0.0),
                "sum_reward_progress":  self._rewards_acc.get(a, {}).get("progress", 0.0),
                "sum_reward_intrusion": self._rewards_acc.get(a, {}).get("intrusion", 0.0),
                "sum_reward_path_efficiency": self._rewards_acc.get(a, {}).get("path_efficiency", 0.0),
                "sum_reward_boundary": self._rewards_acc.get(a, {}).get("boundary", 0.0),
                "sum_reward_step": self._rewards_acc.get(a, {}).get("step", 0.0),
                "sum_reward_proximity": self._rewards_acc.get(a, {}).get("proximity", 0.0),
                "total_intrusions":     self._intrusions_acc.get(a, 0),
                "terminated_waypoint": waypoint_reached,  # True if agent reached waypoint
                "terminated_collision": collided,  # True if agent collided with another aircraft
                "truncated": truncated,  # True if episode truncated (out of bounds or time limit)
                "finished_at": time.time(),  
            })

            # flush this agent's buffer if threshold reached
            if len(self._agent_buffers[a]) >= self._flush_threshold:
                self._flush_agent_buffer(a)
       
        self.agents = [agent for agent in self.agents if agent not in agents_to_remove]
        
        # Set __all__ to False (episode continues until no agents remain)
        terminateds["__all__"] = len(self.agents) == 0
        truncateds["__all__"] = len(self.agents) == 0
        
        # Flush buffers and write stats when all agents are done
        if len(self.agents) == 0:
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
                    "mean_reward_path_efficiency",
                    "mean_reward_boundary",
                    "mean_reward_step",
                    "mean_reward_proximity",
                    "sum_reward_drift",
                    "sum_reward_progress",
                    "sum_reward_intrusion",
                    "sum_reward_path_efficiency",
                    "sum_reward_boundary",
                    "sum_reward_step",
                    "sum_reward_proximity",
                    "total_intrusions",
                    "terminated_waypoint",   # True if agent reached waypoint
                    "terminated_collision",  # True if agent collided with another aircraft
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
                progress_reward = self._check_progress(agent, ac_idx)
                path_efficiency_reward = self._check_path_efficiency(agent, ac_idx)
                boundary_penalty = self._check_boundary_violation(agent, ac_idx)
                proximity_penalty = self._check_proximity(agent, ac_idx)
                step_penalty = STEP_PENALTY  # Applied every step
                
                rewards[agent] = (drift_reward + intrusion_reward + 
                                progress_reward + path_efficiency_reward + 
                                boundary_penalty + proximity_penalty + step_penalty) / 100.0
                
                # accumulate for per-episode stats
                self._rewards_acc[agent]["drift"]     += float(drift_reward)
                self._rewards_acc[agent]["progress"]  += float(progress_reward)
                self._rewards_acc[agent]["intrusion"] += float(intrusion_reward)
                self._rewards_acc[agent]["path_efficiency"] += float(path_efficiency_reward)
                self._rewards_acc[agent]["boundary"] += float(boundary_penalty)
                self._rewards_acc[agent]["proximity"] += float(proximity_penalty)
                self._rewards_acc[agent]["step"] += float(step_penalty)
                self._rewards_counts[agent]          += 1
                
                infos[agent]["reward_drift"] = drift_reward
                infos[agent]["reward_intrusion"] = intrusion_reward
                infos[agent]["reward_progress"] = progress_reward
                infos[agent]["reward_path_efficiency"] = path_efficiency_reward
                infos[agent]["reward_boundary"] = boundary_penalty
                infos[agent]["reward_proximity"] = proximity_penalty
                infos[agent]["reward_step"] = step_penalty
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

        # Precompute positions for all agents
        agent_positions = {}
        for agent in self.agents:
            try:
                ac_idx = bs.traf.id2idx(agent)
                ac_qdr, ac_dis = bs.tools.geo.kwikqdrdist(self.center[0], self.center[1], bs.traf.lat[ac_idx], bs.traf.lon[ac_idx])
                x_pos = (self.window_width / 2) + (np.cos(np.deg2rad(ac_qdr)) * (ac_dis * NM2KM) * px_per_km)
                y_pos = (self.window_height / 2) - (np.sin(np.deg2rad(ac_qdr)) * (ac_dis * NM2KM) * px_per_km)
                agent_positions[agent] = (x_pos, y_pos)
            except Exception:
                continue

        # Draw aircraft, color by risk, and display attention weight
        font = pygame.font.SysFont(None, 18)
        for agent in self.agents:
            try:
                ac_idx = bs.traf.id2idx(agent)
                ac_hdg = bs.traf.hdg[ac_idx]
                pos = agent_positions.get(agent, None)
                if pos is None:
                    continue
                # Color: green for first agent in list, red for others
                if agent == self.agents[0]:
                    color = (0, 255, 0)
                else:
                    color = (200, 0, 0)
                # Draw heading line
                heading_end_x = np.cos(np.deg2rad(ac_hdg)) * 10
                heading_end_y = np.sin(np.deg2rad(ac_hdg)) * 10
                pygame.draw.line(canvas, (0, 0, 0), pos, (pos[0] + heading_end_x, pos[1] - heading_end_y), width=4)
                # Draw aircraft circle
                pygame.draw.circle(canvas, color, (int(pos[0]), int(pos[1])), int(INTRUSION_DISTANCE * NM2KM * px_per_km / 2), width=2)
                # Draw agent ID and attention weight as text
                if agent in self.attention_weights:
                    alpha_val = self.attention_weights[agent]
                    alpha_text = f"{agent} : {alpha_val:.3f}"
                else:
                    # Show agent ID even if no attention weight available
                    alpha_text = f"{agent}"
                text_surf = font.render(alpha_text, True, (0, 0, 0))
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
    def _calculate_conflict_metrics(dx, dy, dvx, dvy, R=PROTECTED_ZONE_M, T=CPA_TIME_HORIZON_S):
        """
        Calculate conflict detection metrics: tcpa, dcpa, and tLOS.
        Based on BlueSky's conflict detection logic (detection.py).
        
        Args:
            dx, dy: Relative position (m) - position of intruder relative to ownship
            dvx, dvy: Relative velocity (m/s) - velocity of ownship minus intruder
            R: Protected zone radius (m)
            T: Time horizon for conflict detection (s)
            
        Returns:
            tcpa: Time to closest point of approach (s)
            dcpa: Distance at closest point of approach (m)
            tLOS: Time to loss of separation (s), or np.inf if no conflict
        """
        EPS = 1e-6
        
        # Current distance and relative velocity squared
        dist = np.sqrt(dx * dx + dy * dy)
        dv2 = dvx * dvx + dvy * dvy
        dv2 = max(dv2, EPS)  # Limit lower absolute value (BlueSky uses 1e-6)
        vrel = np.sqrt(dv2)
        
        # Calculate time to CPA (BlueSky method)
        # tcpa = -(du * dx + dv * dy) / dv2
        tcpa = -(dvx * dx + dvy * dy) / dv2
        
        # Calculate distance at CPA (BlueSky method)
        # dcpa2 = abs(dist^2 - tcpa^2 * dv2)
        dcpa2 = abs(dist * dist - tcpa * tcpa * dv2)
        dcpa = float(np.sqrt(dcpa2))
        
        # Check for horizontal conflict
        R2 = R * R
        swhorconf = dcpa2 < R2
        
        if not swhorconf:
            # No conflict predicted
            tLOS = np.inf
        else:
            # Calculate times of entering conflict zone (BlueSky method)
            # dxinhor = sqrt(R^2 - dcpa^2) is half the distance traveled inside zone
            dxinhor = np.sqrt(max(0.0, R2 - dcpa2))
            dtinhor = dxinhor / vrel
            
            # Entry time into conflict zone
            tinhor = tcpa - dtinhor
            
            # Check if entry time is positive and within lookahead
            if tinhor > 0.0 and tinhor < T:
                tLOS = tinhor
            elif dist < R:
                # Already in conflict
                tLOS = 0.0
            else:
                # Conflict outside time horizon or in past
                tLOS = np.inf
        
        return tcpa, dcpa, tLOS
    
    @staticmethod
    def _calculate_risk_score(dcpa, tLOS, T=CPA_TIME_HORIZON_S, R=PROTECTED_ZONE_M):
        """
        Calculate a risk score based on conflict metrics.
        Higher score = higher risk/priority.
        
        Only aircraft with actual conflicts (tLOS < inf) receive a risk score.
        Non-conflict aircraft are filtered out before this function is called.
        
        Risk scoring logic:
        - Time urgency: How soon the conflict occurs (smaller tLOS = higher risk)
        - Severity: How close the CPA is (smaller dcpa = higher risk)
        
        Note: Risk is only used for sorting, not as NN input, so no clipping needed.
        
        Args:
            dcpa: Distance at closest point of approach (m)
            tLOS: Time to loss of separation (s)
            T: Time horizon (s)
            R: Protected zone radius (m)
            
        Returns:
            risk: Risk score (higher = more urgent conflict)
        """
        EPS = 1e-9
        
        # Time urgency: sooner conflicts are more urgent
        # tLOS=0 gives 1.0, tLOS=T gives ~0
        time_urgency = 1.0 - (tLOS / (T + EPS))
        
        # Severity: how close will they get
        # dcpa=0 gives 1.0, dcpa=R gives 0.5, dcpa>R decreases
        severity = 1.0 / (1.0 + (dcpa / (R * 0.5 + EPS)))
        
        # Combine: 60% time urgency, 40% severity
        # This means imminent conflicts get highest priority
        risk = 0.6 * time_urgency + 0.4 * severity
        
        return float(risk)

            

    def _get_observation(self, active_agents):
        # code that builds the observation vector. 
        
        # origin reference for absolute positions
        y_origin = self.center[0]
        x_origin = self.center[1]
        
        # observation vector
        # 7 ownship features + 5 features per intruder (distance, dx_rel, dy_rel, vx_rel, vy_rel)
        dim = 7 + 5 * NUM_AC_STATE
        obs = {}

        if not hasattr(self, "_obs_errors"):
            self._obs_errors = {}

        for agent in active_agents:
            try:
                ac_idx = bs.traf.id2idx(agent)

                # --- 1. Ownship Features ---
                wpt_lat, wpt_lon = self.agent_waypoints[agent]
                wpt_qdr, _ = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], wpt_lat, wpt_lon)
                ac_hdg = bs.traf.hdg[ac_idx]
                drift = fn.bound_angle_positive_negative_180(ac_hdg - wpt_qdr)
                
                cos_drift, sin_drift = np.cos(np.deg2rad(drift)), np.sin(np.deg2rad(drift))
                airspeed = bs.traf.tas[ac_idx] / 18 # normalize on to a max of 18 m/s (~35 kt)
                
                # location
                dx = (bs.traf.lon[ac_idx] - x_origin) / MAX_LAT_LON # normalized  difference in longitude TODO dit misschien nog * 100 doen ofzo om wat zwaarder mee te laten tellen?
                dy = (bs.traf.lat[ac_idx] - y_origin) / MAX_LAT_LON # difference in latitude
                
                # velocity
                vx = (np.cos(np.deg2rad(ac_hdg)) * bs.traf.gs[ac_idx]  )  / 18 # normalize on to a max of 18 m/s (~35 kt)
                vy = (np.sin(np.deg2rad(ac_hdg)) * bs.traf.gs[ac_idx]  ) / 18 # normalize on to a max of 18 m/s (~35 kt)
                # ac_loc = fn.latlong_to_nm(self.center, np.array([bs.traf.lat[ac_idx], bs.traf.lon[ac_idx]])) * NM2KM * 1000
                
                # maybe for determining the relative position of other ac
                # own_lat = bs.traf.lat[ac_idx]
                # own_lon = bs.traf.lon[ac_idx]
                # own_gs = bs.traf.gs[ac_idx]
                # own_hdg = bs.traf.hdg[ac_idx]

                # --- 2. Build Candidate List (in index order, no sorting) ---
                candidates = []
                # Get current agent ID to index mapping
                agent_id_to_idx = {agent_id: idx for idx, agent_id in enumerate(self.agents)}
                
                for other_agent_id in self.agents:
                    if other_agent_id == agent:
                        continue
                    
                    # Skip if we can't find the waypoint
                    if other_agent_id not in self.agent_waypoints:
                        continue
                    
                    # Get the BlueSky index for this agent
                    i = agent_id_to_idx[other_agent_id]
                    
                    # Calculate absolute positions
                    dxi_abs = (bs.traf.lon[i] - x_origin) / MAX_LAT_LON
                    dyi_abs = (bs.traf.lat[i] - y_origin) / MAX_LAT_LON
                    
                    # Calculate absolute velocities
                    int_hdg = bs.traf.hdg[i]
                    vxi_abs = (np.cos(np.deg2rad(int_hdg)) * bs.traf.gs[i]) / 18
                    vyi_abs = (np.sin(np.deg2rad(int_hdg)) * bs.traf.gs[i]) / 18
                    
                    # Make positions and velocities RELATIVE to ownship
                    dx_rel = dxi_abs - dx  # intruder x - ownship x
                    dy_rel = dyi_abs - dy  # intruder y - ownship y
                    vx_rel = vxi_abs - vx  # intruder vx - ownship vx
                    vy_rel = vyi_abs - vy  # intruder vy - ownship vy
                    
                    # Calculate distance between ownship and intruder (in nautical miles)
                    _, distance_nm = bs.tools.geo.kwikqdrdist(
                        bs.traf.lat[ac_idx], bs.traf.lon[ac_idx],
                        bs.traf.lat[i], bs.traf.lon[i]
                    )
                    # Normalize distance (typical max ~1 NM in this scenario)
                    distance_normalized = float(distance_nm)
                    
                    # Store (distance, dx_rel, dy_rel, vx_rel, vy_rel, agent_id)
                    candidates.append((distance_normalized, dx_rel, dy_rel, vx_rel, vy_rel, other_agent_id))

                # Sort by distance (closest first) and take top NUM_AC_STATE neighbors
                candidates.sort(key=lambda x: x[0])
                top = candidates[:NUM_AC_STATE]
                    
                # Store neighbor mapping for attention visualization (neighbor IDs sorted by distance)
                self.neighbor_mapping[agent] = [c[5] for c in top if c[5] is not None]

                # --- 3. Construct Vector ---
                # We iterate through neighbors and append ALL features for that neighbor sequentially.
                # This creates [Ownship, Agent1(5 features), Agent2(5 features), ...]
                # Agents are sorted by distance (closest first)
                
                intruder_features = []
                for i in range(NUM_AC_STATE):
                    if i < len(top):
                        # Extract features from tuple
                        t = top[i]
                        # Indices: 0=distance, 1=dx_rel, 2=dy_rel, 3=vx_rel, 4=vy_rel, 5=id
                        distance, dx_rel, dy_rel, vx_rel, vy_rel = t[0], t[1], t[2], t[3], t[4]
                        
                        # Append 5 features for this intruder
                        intruder_features.extend([distance, dx_rel, dy_rel, vx_rel, vy_rel])
                    else:
                        # Padding (5 zeros per missing agent)
                        intruder_features.extend([0.0] * 5)

                # Concatenate Ownship + Intruders
                ownship_feats = np.array([cos_drift, sin_drift, airspeed, dx, dy, vx, vy], dtype=np.float32)
                intruder_feats = np.array(intruder_features, dtype=np.float32)
                
                vec = np.concatenate([ownship_feats, intruder_feats])

                # Sanitize and shape-check
                # if not np.isfinite(vec).all():
                #     vec = np.nan_to_num(vec, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)

                # if vec.shape[0] != dim:
                #     fixed = np.zeros(dim, dtype=np.float32)
                #     n = min(dim, vec.shape[0])
                #     fixed[:n] = vec[:n]
                #     vec = fixed

                obs[agent] = vec

            except (IndexError, KeyError, ValueError, RuntimeError) as e:
                obs[agent] = np.zeros(dim, dtype=np.float32)
            except Exception as e:
                obs[agent] = np.zeros(dim, dtype=np.float32)

        return obs

    # ---- Observation statistics helpers ----
    def _get_obs_feature_names(self):
        dim = 7 + 5 * NUM_AC_STATE
        names = []
        # Ownship features (7 total)
        names.extend(["cos_drift", "sin_drift", "airspeed", "x", "y", "vx", "vy"])
        
        # Intruder features (5 per intruder)
        feature_labels = ["distance", "dx_rel", "dy_rel", "vx_rel", "vy_rel"]
        
        for i in range(NUM_AC_STATE):
            for label in feature_labels:
                names.append(f"intruder_{i}_{label}")
                
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
        # indices per block - ownship has 7 features
        i0 = 0
        pr_range("cos_drift", slice(i0, i0+1)); i0 += 1
        pr_range("sin_drift", slice(i0, i0+1)); i0 += 1
        pr_range("airspeed", slice(i0, i0+1)); i0 += 1
        pr_range("x", slice(i0, i0+1)); i0 += 1
        pr_range("y", slice(i0, i0+1)); i0 += 1
        pr_range("vx", slice(i0, i0+1)); i0 += 1
        pr_range("vy", slice(i0, i0+1)); i0 += 1
        # 5 features per intruder (NUM_AC_STATE intruders)
        labels = ["distance", "dx_rel", "dy_rel", "vx_rel", "vy_rel"]
        for lab in labels:
            pr_range(f"intruder_{lab}", slice(i0, i0+NUM_AC_STATE))
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
        # Terminate agents that reached their waypoint OR collided with another aircraft
        return {agent: (agent in self.waypoint_reached_agents or agent in self.collided_agents) 
                for agent in active_agents}
    
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
        
        # Scale the progress reward to make it more significant
        # Typical distance improvement per step: ~0.001-0.01 NM
        # With scale factor 5.0: reward ~0.005-0.05 per step
        # This helps offset STEP_PENALTY (-0.005) when making progress
        
        # Option 1: Give reward AND penalty (positive when closer, negative when farther)
        progress_reward = distance_improvement * PROGRESS_REWARD_SCALE
        
        # Option 2: Only give positive rewards (no penalty for moving away) - CURRENTLY ACTIVE
        # progress_reward = max(0, distance_improvement * PROGRESS_REWARD_SCALE)
        
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
        
        # code to check polygon area
        area_checker = fn.polygon_area(self.poly_points) #  in NM^2
        area_checker_km2 = area_checker * NM2KM * NM2KM
        # print(f"density generated area of {area_checker_km2:.2f} km^2 = {6/area_checker_km2:.2f} ac/km^2")
        self.areas_km2.append(area_checker_km2)
        
        p_latlong = [fn.nm_to_latlong(self.center, point) for point in self.poly_points]
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

        init_p_nm = []  # Store positions in NM for distance checking
        init_p_latlong = []
        max_attempts = 1000  # Prevent infinite loop if area is too crowded
        
        while len(init_p_latlong) < self.num_ac:
            # Try to find a valid spawn point
            for attempt in range(max_attempts):
                p_nm = np.array([np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y)])
                p_latlong = fn.nm_to_latlong(self.center, p_nm)
                
                # Check if inside polygon
                if not bs.tools.areafilter.checkInside(self.poly_name, np.array([p_latlong[0]]), 
                                                       np.array([p_latlong[1]]), np.array([ALTITUDE * FT2M])):
                    continue
                
                # Check if point is far enough from all existing spawn points
                too_close = False
                for existing_p_nm in init_p_nm:
                    distance = fn.euclidean_distance(p_nm, existing_p_nm)
                    if distance < MIN_SPAWN_SEPARATION_NM:
                        too_close = True
                        break
                
                if not too_close:
                    # Valid point found
                    init_p_nm.append(p_nm)
                    init_p_latlong.append(p_latlong)
                    break
            else:
                # If max_attempts reached, accept any valid point (fallback)
                print(f"[WARNING] Could not find well-separated spawn point for aircraft {len(init_p_latlong)+1}, using less separated point")
                p_nm = np.array([np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y)])
                p_latlong = fn.nm_to_latlong(self.center, p_nm)
                if bs.tools.areafilter.checkInside(self.poly_name, np.array([p_latlong[0]]), 
                                                   np.array([p_latlong[1]]), np.array([ALTITUDE * FT2M])):
                    init_p_nm.append(p_nm)
                    init_p_latlong.append(p_latlong)

        self.agent_waypoints = {}
        wpts_latlon = [fn.nm_to_latlong(self.center, p) for p in self.wpts_nm]
        
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
    
    def _check_path_efficiency(self, agent_id, ac_idx):
        """Reward for staying close to the straight-line path to waypoint.
        
        This discourages large unnecessary detours while still allowing 
        deviations needed for conflict avoidance.
        
        Measures the perpendicular distance from the aircraft to the 
        straight line between start position and waypoint.
        """
        if not hasattr(self, '_agent_start_positions'):
            self._agent_start_positions = {}
        
        # Store initial position on first call for this agent
        if agent_id not in self._agent_start_positions:
            ac_loc = fn.latlong_to_nm(self.center, 
                np.array([bs.traf.lat[ac_idx], bs.traf.lon[ac_idx]]))
            self._agent_start_positions[agent_id] = ac_loc
            return 0.0  # No penalty/reward on first step
        
        # Get current position
        ac_loc = fn.latlong_to_nm(self.center, 
            np.array([bs.traf.lat[ac_idx], bs.traf.lon[ac_idx]]))
        
        # Get waypoint position
        wpt_lat, wpt_lon = self.agent_waypoints[agent_id]
        wpt_loc = fn.latlong_to_nm(self.center, np.array([wpt_lat, wpt_lon]))
        
        # Get start position
        start_loc = self._agent_start_positions[agent_id]
        
        # Calculate perpendicular distance to straight-line path
        # Using point-to-line distance formula
        start_to_wpt = wpt_loc - start_loc
        start_to_ac = ac_loc - start_loc
        
        path_length = np.linalg.norm(start_to_wpt)
        
        if path_length < 1e-6:  # Aircraft at waypoint
            return 0.0
        
        # Project aircraft position onto the path vector
        projection = np.dot(start_to_ac, start_to_wpt) / path_length
        
        # Calculate perpendicular distance (cross-track error)
        cross_track_error = np.abs(np.cross(start_to_wpt, start_to_ac)) / path_length
        
        # Reward inversely proportional to cross-track error
        # Small detours (< 0.1 NM) get small penalty
        # Large detours (> 0.3 NM) get larger penalty
        # Scale: -PATH_EFFICIENCY_SCALE at ~0.3 NM cross-track error
        penalty = -cross_track_error * PATH_EFFICIENCY_SCALE
        
        return float(penalty)
    
    def _check_boundary_violation(self, agent_id, ac_idx):
        """Check if agent has left the polygon boundary (when not at waypoint).
        
        Returns:
            float: Penalty if outside boundary (not at waypoint), 0.0 otherwise
        """
        try:
            # Check if agent has reached waypoint - no penalty if at waypoint
            if agent_id in self.waypoint_reached_agents:
                return 0.0
            
            # Check if agent is outside the polygon
            outside = not bs.tools.areafilter.checkInside(
                self.poly_name,
                np.array([bs.traf.lat[ac_idx]]),
                np.array([bs.traf.lon[ac_idx]]),
                np.array([ALTITUDE * FT2M]),
            )
            
            if outside:
                return BOUNDARY_VIOLATION_PENALTY
            else:
                return 0.0
                
        except Exception as e:
            # If we can't check boundary, assume it's okay
            return 0.0
    
    def _check_collisions(self, active_agents):
        """Check for collisions between aircraft and mark collided agents for termination.
        
        A collision occurs when two aircraft are closer than COLLISION_DISTANCE.
        Both aircraft in a collision are marked for termination.
        
        Args:
            active_agents: List of currently active agent IDs
        """
        agents_to_check = [agent for agent in active_agents if agent not in self.collided_agents]
        
        for i, agent_i in enumerate(agents_to_check):
            if agent_i in self.collided_agents:
                continue
                
            try:
                ac_idx_i = bs.traf.id2idx(agent_i)
            except (IndexError, KeyError):
                continue
            
            for agent_j in agents_to_check[i+1:]:
                if agent_j in self.collided_agents:
                    continue
                    
                try:
                    ac_idx_j = bs.traf.id2idx(agent_j)
                except (IndexError, KeyError):
                    continue
                
                # Calculate distance between aircraft
                _, distance = bs.tools.geo.kwikqdrdist(
                    bs.traf.lat[ac_idx_i], bs.traf.lon[ac_idx_i],
                    bs.traf.lat[ac_idx_j], bs.traf.lon[ac_idx_j]
                )
                
                # Check for collision
                if distance < COLLISION_DISTANCE:
                    self.collided_agents.add(agent_i)
                    self.collided_agents.add(agent_j)
                    print(f"[COLLISION] Agents {agent_i} and {agent_j} collided at distance {distance*1852:.1f}m")
    
    def _check_proximity(self, agent_id, ac_idx):
        """Calculate proximity penalty for getting close to other aircraft.
        
        This creates a "force field" effect that gently pushes agents away from each other
        before they reach the hard intrusion boundary.
        
        Penalty applies in a "soft band" between INTRUSION_DISTANCE and SOFT_INTRUSION_FACTOR * INTRUSION_DISTANCE.
        The penalty increases quadratically (squared) as agents get closer, creating smooth gradients.
        
        Args:
            agent_id: The agent identifier
            ac_idx: The aircraft index in BlueSky's traffic arrays
            
        Returns:
            float: Proximity penalty (0.0 or negative value)
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
            min_dist = min(min_dist, d)  # Slightly faster built-in min

        # Outside soft band: no shaping
        if min_dist >= soft_thresh:
            return 0.0

        # Calculate Ratio (0.0 = Far, 1.0 = Touching Hard Boundary)
        band = soft_thresh - INTRUSION_DISTANCE
        ratio = (soft_thresh - min_dist) / band if band > 0 else 0.0
        
        # TWEAK 1: Clamp at 1.0, but DON'T return 0 if inside intrusion.
        # We want the "pain" of the soft penalty to persist on top of the hard penalty.
        ratio = np.clip(ratio, 0.0, 1.0)

        # TWEAK 2: Use Square (Exponential) for the "Force Field" effect.
        # 0.5 distance -> 0.25 pain. 0.9 distance -> 0.81 pain.
        return -PROXIMITY_MAX_PENALTY * float(ratio ** 2)
    
    def _check_intrusion(self, agent_id, ac_idx):
        """Return intrusion penalty for this agent on this step, and intrusion flag.

        Drone pairs receive the intrusion penalty EVERY TIMESTEP they are intruding.
        Both agents in the pair receive the penalty in the same step when they intrude.
        
        Uses _pairs_penalized_this_step to ensure BOTH agents get penalty in the same step,
        preventing double-counting within a single timestep.
        
        Returns:
            tuple: (step_penalty, intrusion_occurred)
                - step_penalty: float, the penalty for intrusions (applied every timestep)
                - intrusion_occurred: bool, True if any intrusion was detected
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
                had_intrusion = True  # Intrusion detected
                
                # Get the other agent's ID
                other_agent_id = self.agents[i] if i < len(self.agents) else None
                if other_agent_id is None:
                    continue
                
                # Create a consistent pair identifier (sorted so order doesn't matter)
                pair = tuple(sorted([agent_id, other_agent_id]))
                
                # Ensure BOTH agents get the penalty in THIS step:
                if pair in self._pairs_penalized_this_step:
                    # Second agent of the pair - also gets penalty
                    step_penalty += self.intrusion_penalty
                    # Track per-agent intrusion count (only when penalty is applied)
                    if agent_id in self._intrusions_acc:
                        self._intrusions_acc[agent_id] += 1
                else:
                    # First agent of the pair - gets penalty and marks the pair for this step
                    self._pairs_penalized_this_step.add(pair)
                    step_penalty += self.intrusion_penalty
                    # Count intrusion occurrences (every timestep it happens)
                    self.total_intrusions += 1
                    # Track per-agent intrusion count (only when penalty is applied)
                    if agent_id in self._intrusions_acc:
                        self._intrusions_acc[agent_id] += 1

        return step_penalty, had_intrusion
