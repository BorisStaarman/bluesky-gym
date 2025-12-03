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


# scenario constants
POLY_AREA_RANGE = (0.75, 0.8)
# Default center (can be overridden in __init__)
CENTER = np.array([52.362566, 4.881444]) # new center from training scenario

# aircraft constants
ALTITUDE = 360
AC_SPD = 9 # starting speed, in m/s !
AC_TYPE = "M600"
INTRUSION_DISTANCE = 1 / 1852 * 50  # was 0.054

# conversion factors
NM2KM = 1.852
MpS2Kt = 1.94384
FT2M = 0.3048

# model settings
ACTION_FREQUENCY = 1 # how many sim steps per action
NUM_AC_STATE = 3 # number of aircraft in observation vector
MAX_STEPS = 300 # max steps per episode

# penalties for reward
DRIFT_PENALTY = -0.006  # Very small - detours should be cheap
STEP_PENALTY = -0.00075 
INTRUSION_PENALTY = -12.0  # Separation violation - penalty applied once per drone pair per episode
WAYPOINT_REACHED_REWARD = 8.0  # Increased to encourage completion
# waypoint rewards
WAYPOINT_RADIUS = 0.05 # in NM is about 90 meters
PROGRESS_REWARD_SCALE = 5.0  # Scale factor for distance-to-waypoint progress (higher = more emphasis on reaching goal)
SOFT_INTRUSION_FACTOR = 0.5  # Wider soft band: 2x the intrusion distance (0.108 NM)
PROXIMITY_MAX_PENALTY = 0.1 * abs(INTRUSION_PENALTY)  # was 0.06

# constants to control actions, 
D_HEADING = 45 # degrees
D_VELOCITY = 10/3 # knots

# normalization parameters
# MAX_SCENARIO_DIM_M = (POLY_AREA_RANGE[1] + POLY_AREA_RANGE[0])/2 * NM2KM * 1000.0 * 2.0

# THIS SHOULD be 1.87 km 
MAX_SCENARIO_DIM_M = 1870.0

# For distance: all negative means center is too high; reduce multiplier from 2.5 to 1.5
DISTANCE_CENTER_M = MAX_SCENARIO_DIM_M / 2.0
DISTANCE_SCALE_M = MAX_SCENARIO_DIM_M / 2.0 * 1.5
# For vx_r, vy_r: observed max ~1.42, increase from 50 to 75 m/s
MAX_RELATIVE_VEL_MS = 70.0 # TODO dit is knts ipv m/s
AIRSPEED_CENTER_KTS = 35.0
AIRSPEED_SCALE_KTS = 10.0 * 3.4221

# collision risk parameters
PROTECTED_ZONE_M = 50  # meters
CPA_TIME_HORIZON_S = 15 # seconds

# logging
LOG_EVERY_N = 100  # throttle repeated warnings

# Add a base dir for metrics written by this env
# NOTE: This is just a fallback - metrics_base_dir is passed via env_config from main.py
METRICS_BASE_DIR = "metrics"  # Fallback (not used when passed via env_config)

class SectorEnv(MultiAgentEnv):
    metadata = {"name": "ma_env", "render_modes": ["rgb_array", "human"], "render_fps": 10}

    def __init__(self, render_mode=None, n_agents=30, run_id="default",
                 debug_obs=False, debug_obs_episodes=2, debug_obs_interval=1, debug_obs_agents=None,
                 collect_obs_stats=False, print_obs_stats_per_episode=False,
                 intrusion_penalty=None, metrics_base_dir=None, center=None):
        super().__init__()
        
        self.render_mode = render_mode
        # Use provided intrusion penalty or default from module constant
        self.intrusion_penalty = intrusion_penalty if intrusion_penalty is not None else INTRUSION_PENALTY
        # Calculate proximity penalty based on actual intrusion penalty being used
        self.proximity_max_penalty = PROXIMITY_MAX_PENALTY
        self.num_ac = n_agents
        # Use provided center or default - this is the reference point for relative coordinates
        self.center = np.array(center) if center is not None else CENTER
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

        # stuff for making csv file
        self._agent_steps = {}
        self._rewards_acc = {}                 # per-agent running sums during the episode
        self._rewards_counts = {}              # per-agent step counts for safe averaging
        self._intrusions_acc = {}              # per-agent intrusion counts during the episode
                
        single_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3 + 8 * NUM_AC_STATE,), dtype=np.float32)
        single_action_space = spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
        
        self.areas_km2 = []
        
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
        
        # Note: _pairs_penalized_this_step is cleared at the start of each step
        # This allows the same pair to be penalized again in the next timestep
        
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
        
        # End episode when 1 or fewer agents remain
        all_done = len(self.agents) <= 1
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
                ac_qdr, ac_dis = bs.tools.geo.kwikqdrdist(self.center[0], self.center[1], bs.traf.lat[ac_idx], bs.traf.lon[ac_idx])
                x_pos = (self.window_width / 2) + (np.cos(np.deg2rad(ac_qdr)) * (ac_dis * NM2KM) * px_per_km)
                y_pos = (self.window_height / 2) - (np.sin(np.deg2rad(ac_qdr)) * (ac_dis * NM2KM) * px_per_km)
                agent_positions[agent] = (x_pos, y_pos)
            except Exception:
                continue

        # Draw lines to the top 3 most risky neighbors for agent 'KL001'
        # Use pre-computed most_risky data instead of recalculating
        agent1 = 'KL001'
        if agent1 in most_risky and agent1 in agent_positions:
            # Get the top 3 risky neighbors from the sorted risk_levels
            try:
                # Sort all other agents by risk level
                other_agents = [(agent, risk_levels.get(agent, 0.0)) 
                               for agent in self.agents if agent != agent1]
                other_agents.sort(key=lambda x: -x[1])
                top3 = [(agent, risk) for agent, risk in other_agents[:3] 
                        if agent in agent_positions]
                
                line_colors = [(255,0,0), (255,140,0), (255,255,0)]  # red, orange, yellow
                for idx, (neighbor_id, _) in enumerate(top3):
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
        dim = 3 + 8 * NUM_AC_STATE
        obs = {}
        risk_levels = {}
        most_risky = {}  # agent_id: (most_risky_other_agent_id, risk_value)

        if not hasattr(self, "_obs_errors"):
            self._obs_errors = {}

        def note(agent, msg):
            c = self._obs_errors.get(agent, 0) + 1
            self._obs_errors[agent] = c
            if c % LOG_EVERY_N == 1:
                print(f"[OBS-WARN] {agent}: {msg} (count={c})")

        for agent in active_agents:
            try:
                ac_idx = bs.traf.id2idx(agent)

                # feature computations 
                wpt_lat, wpt_lon = self.agent_waypoints[agent]
                wpt_qdr, _ = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], wpt_lat, wpt_lon)
                ac_hdg = bs.traf.hdg[ac_idx]
                drift = fn.bound_angle_positive_negative_180(ac_hdg - wpt_qdr)
                # if agent == "KL001":
                #     print(f"agent={agent}, drift = {drift}") 
                cos_drift, sin_drift = np.cos(np.deg2rad(drift)), np.sin(np.deg2rad(drift))
                airspeed_kts = bs.traf.tas[ac_idx] * MpS2Kt
                
                vx = np.cos(np.deg2rad(ac_hdg)) * bs.traf.gs[ac_idx]
                vy = np.sin(np.deg2rad(ac_hdg)) * bs.traf.gs[ac_idx]

                ac_loc = fn.latlong_to_nm(self.center, np.array([bs.traf.lat[ac_idx], bs.traf.lon[ac_idx]])) * NM2KM * 1000

                # Build candidate list with CPA risk, and keep track of agent ids
                candidates = []
                for i in range(self.num_ac):
                    if i == ac_idx:
                        continue
                    other_agent_id = self.agents[i] if i < len(self.agents) else None
                    int_hdg = bs.traf.hdg[i]
                    int_loc = fn.latlong_to_nm(self.center, np.array([bs.traf.lat[i], bs.traf.lon[i]])) * NM2KM * 1000
                    dx = float(int_loc[0] - ac_loc[0])
                    dy = float(int_loc[1] - ac_loc[1])
                    
                    vxi = np.cos(np.deg2rad(int_hdg)) * bs.traf.gs[i]
                    vyi = np.sin(np.deg2rad(int_hdg)) * bs.traf.gs[i]
                    dvx = float(vxi - vx)
                    dvy = float(vyi - vy)
                    trk = np.arctan2(dvy, dvx)
                    d_now = float(np.hypot(dx, dy))
                    risk = SectorEnv._cpa_risk(dx, dy, dvx, dvy)
                    candidates.append((risk, dx, dy, dvx, dvy, np.cos(trk), np.sin(trk), d_now, other_agent_id))

                # Sort by descending risk; break ties by current distance (closer first)
                candidates.sort(key=lambda t: (-t[0], t[7]))
                top = candidates[:NUM_AC_STATE]

                # Find the most risky other agent
                if candidates:
                    most_risk, *_, most_risky_id = candidates[0]
                    most_risky[agent] = (most_risky_id, most_risk)
                    risk_levels[agent] = most_risk
                else:
                    most_risky[agent] = (None, 0.0)
                    risk_levels[agent] = 0.0

                # Unpack top-N into arrays
                risk       = [t[0] for t in top]
                x_r        = [t[1] for t in top]
                y_r        = [t[2] for t in top]
                vx_r       = [t[3] for t in top]
                vy_r       = [t[4] for t in top]
                cos_track  = [t[5] for t in top]
                sin_track  = [t[6] for t in top]
                distances  = [t[7] for t in top]

                # Normalization WITHOUT clipping; we will track min/max/mean to tune scales later
                airspeed_norm = (airspeed_kts - AIRSPEED_CENTER_KTS) / AIRSPEED_SCALE_KTS

                risk_arr = np.array(risk, dtype=float)
                x_rn = np.array(x_r, dtype=float) / MAX_SCENARIO_DIM_M
                y_rn = np.array(y_r, dtype=float) / MAX_SCENARIO_DIM_M
                vx_rn = np.array(vx_r, dtype=float) / MAX_RELATIVE_VEL_MS
                vy_rn = np.array(vy_r, dtype=float) / MAX_RELATIVE_VEL_MS
                cos_trk = np.array(cos_track, dtype=float)
                sin_trk = np.array(sin_track, dtype=float)
                dist_n = (np.array(distances, dtype=float) - DISTANCE_CENTER_M) / DISTANCE_SCALE_M

                parts = [
                    np.array([cos_drift], dtype=float), 
                    np.array([sin_drift], dtype=float),
                    np.array([airspeed_norm], dtype=float),
                    risk_arr,
                    # risk_arr removed - only used for sorting, not in observation
                    x_rn,
                    y_rn,
                    vx_rn, 
                    vy_rn,
                    cos_trk, # already in [-1,1]
                    sin_trk, # already in [-1,1]
                    dist_n,
                ]
                vec = np.concatenate([v.ravel() for v in parts]).astype(np.float32)

                # sanitize and shape-check
                if not np.isfinite(vec).all():
                    note(agent, "non-finite values in observation; sanitizing")
                    vec = np.nan_to_num(vec, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)

                if vec.shape[0] != dim:
                    note(agent, f"obs length {vec.shape[0]} != expected {dim}; padding/truncating")
                    fixed = np.zeros(dim, dtype=np.float32)
                    n = min(dim, vec.shape[0])
                    fixed[:n] = vec[:n]
                    vec = fixed

                obs[agent] = vec

            except (IndexError, KeyError, ValueError, RuntimeError) as e:
                note(agent, f"{type(e).__name__}: {e}")
                obs[agent] = np.zeros(dim, dtype=np.float32)

            except Exception as e:
                # unexpected error → still keep training robustly, but make it visible
                note(agent, f"Unexpected {type(e).__name__}: {e}")
                obs[agent] = np.zeros(dim, dtype=np.float32)
                # during development you could: raise

        return obs, risk_levels, most_risky

    # ---- Observation statistics helpers ----
    def _get_obs_feature_names(self):
        dim = 3 +8 * NUM_AC_STATE
        names = []
        names.extend(["cos_drift", "sin_drift", "airspeed_norm"])
        # Then 7 blocks each of size NUM_AC_STATE in the order we concatenate
        blocks = [
            ("risk", NUM_AC_STATE),  # Removed - only used for sorting
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
        # 7 blocks of NUM_AC_STATE (risk removed)
        labels = ["x_r", "y_r", "vx_r", "vy_r", "cos_trk", "sin_trk", "dist_n"]
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
        print(f"density generated area of {area_checker_km2:.2f} km^2 = {self.num_ac/area_checker_km2:.2f} ac/km^2")
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

        init_p_latlong = []
        while len(init_p_latlong) < self.num_ac:
            p_nm = np.array([np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y)])
            p_latlong = fn.nm_to_latlong(self.center, p_nm)
            if bs.tools.areafilter.checkInside(self.poly_name, np.array([p_latlong[0]]), np.array([p_latlong[1]]), np.array([ALTITUDE * FT2M])):
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
        # Penalty increases as aircraft get CLOSER to intrusion boundary
        # - At soft_thresh (0.108 NM): penalty = 0 (no penalty, safe distance)
        # - At INTRUSION_DISTANCE (0.054 NM): penalty = -PROXIMITY_MAX_PENALTY (maximum warning)
        band = soft_thresh - INTRUSION_DISTANCE  # Width of the warning zone (0.054 NM)
        ratio = (soft_thresh - min_dist) / band if band > 0 else 0.0  # 0 when far, 1 when close
        ratio = np.clip(ratio, 0.0, 1.0)
        return -self.proximity_max_penalty * float(ratio)
    
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
