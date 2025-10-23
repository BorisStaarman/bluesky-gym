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
POLY_AREA_RANGE = (1.0, 1.2)
CENTER = np.array([51.990426702297746, 4.376124857109851]) 

# aircraft constants
ALTITUDE = 360
AC_SPD = 35 # starting speed ?
AC_TYPE = "m600"
INTRUSION_DISTANCE = 0.054

# conversion factors
NM2KM = 1.852
MpS2Kt = 1.94384
FT2M = 0.3048

# model settings
ACTION_FREQUENCY = 5 # how many sim steps per action
NUM_AC_STATE = 3 # number of aircraft in observation vector
MAX_STEPS = 400 # max steps per episode

# penalties for reward
DRIFT_PENALTY = -0.02  # Very small - detours should be cheap
STEP_PENALTY = -0.01
INTRUSION_PENALTY = -3.0  # Moderate but significant penalty
WAYPOINT_REACHED_REWARD = 1  # Increased to encourage completion
# waypoint rewards
WAYPOINT_RADIUS = 0.05 # in NM is about 90 meters

# Proximity shaping: apply a gentle negative reward when within a soft band
# outside the hard intrusion distance. Scale smoothly from 0 at the soft
# boundary to a modest penalty near the hard boundary.
SOFT_INTRUSION_FACTOR = 1.5  # narrower soft band begins at 1.5x the hard separation
# At the closest non-intruding distance, max shaping magnitude is 15% of the hard penalty
PROXIMITY_MAX_PENALTY = 0.3 * abs(INTRUSION_PENALTY)

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

# Add a base dir for metrics written by this env
# NOTE: When copying this folder to a new date/version, update this constant!
# Or better: pass metrics_base_dir via env_config in your training script
METRICS_BASE_DIR = "metrics_23_10"

class SectorEnv(MultiAgentEnv):
    metadata = {"name": "ma_env", "render_modes": ["rgb_array", "human"], "render_fps": 10}

    def __init__(self, render_mode=None, n_agents=10, run_id="default",
                 debug_obs=False, debug_obs_episodes=2, debug_obs_interval=1, debug_obs_agents=None,
                 collect_obs_stats=False, print_obs_stats_per_episode=False,
                 intrusion_penalty=None, metrics_base_dir=None):
        super().__init__()
        self.render_mode = render_mode
        # Use provided intrusion penalty or default from module constant
        self.intrusion_penalty = intrusion_penalty if intrusion_penalty is not None else INTRUSION_PENALTY
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
        
        single_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3 + 8 * NUM_AC_STATE,), dtype=np.float32)
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

        observations = self._get_observation(self.agents)
        self._update_obs_stats(observations)
        self._maybe_print_observations(observations, when="reset")
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions):
        agents_in_step = list(self.agents)
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
                "total_intrusions":     self._intrusions_acc.get(a, 0),  # NEW: intrusions per episode
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
                    "total_intrusions",  # NEW: Count of intrusions per episode
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
                intrusion_reward = self._check_intrusion(agent, ac_idx)
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
        px_per_km = self.window_width/max_distance
        canvas = pygame.Surface(self.window_size)
        canvas.fill((135,206,235))
        coords = [((self.window_width/2) + p[0]*NM2KM*px_per_km, (self.window_height/2) - p[1]*NM2KM*px_per_km) for p in self.poly_points]
        pygame.draw.polygon(canvas, (255, 0, 0), coords, width=2)
        
        for agent in self.agents: 
            try:
                ac_idx = bs.traf.id2idx(agent)
            except Exception: continue
            ac_hdg = bs.traf.hdg[ac_idx]
            ac_qdr, ac_dis = bs.tools.geo.kwikqdrdist(CENTER[0], CENTER[1], bs.traf.lat[ac_idx], bs.traf.lon[ac_idx])
            separation = bs.tools.geo.kwikdist(np.concatenate((bs.traf.lat[:ac_idx], bs.traf.lat[ac_idx+1:])), np.concatenate((bs.traf.lon[:ac_idx], bs.traf.lon[ac_idx+1:])), bs.traf.lat[ac_idx], bs.traf.lon[ac_idx])
            color = (220,20,60) if np.any(separation < INTRUSION_DISTANCE) else (80,80,80)
            x_pos = (self.window_width/2) + (np.cos(np.deg2rad(ac_qdr)) * (ac_dis * NM2KM) * px_per_km)
            y_pos = (self.window_height/2) - (np.sin(np.deg2rad(ac_qdr)) * (ac_dis * NM2KM) * px_per_km)
            heading_end_x = np.cos(np.deg2rad(ac_hdg)) * 10 
            heading_end_y = np.sin(np.deg2rad(ac_hdg)) * 10
            pygame.draw.line(canvas, (0,0,0), (x_pos,y_pos), (x_pos+heading_end_x, y_pos-heading_end_y), width=4)
            pygame.draw.circle(canvas, color, (x_pos,y_pos), radius=INTRUSION_DISTANCE*NM2KM*px_per_km/2, width=2)
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
            
    # function for determining which observation is important
    # static method means it does not need self parameter
    @staticmethod
    def _cpa_risk(dx, dy, dvx, dvy,
              R=PROTECTED_ZONE_M, T=CPA_TIME_HORIZON_S,
              k=0.25, w_d=0.8, w_t=0.1, w_c=0.1, diverge_penalty=0.2):
        """Return a scalar risk in [0,1]; higher means more collision risk."""
        EPS = 1e-6 # to avoid div by zero
        rv = dx*dvx + dy*dvy          # relative position dot relative velocity
        v2 = dvx*dvx + dvy*dvy        # relative speed squared  
        r2 = dx*dx + dy*dy            # relative distance squared
        if v2 < EPS: # no relative motion
            t_cpa = 0.0 # no CPA
            d_cpa = float(np.hypot(dx, dy)) # eucledian distance
            approaching = False # not moving towards eaach other
            speed_mag = 0.0 # no speed relative
        else:
            speed_mag = float(np.sqrt(v2)) # speed magnitude
            t_cpa = - rv / v2 # time to CPA
            t_cpa = float(np.clip(t_cpa, 0.0, T)) # set max and min
            d_cpa = float(np.hypot(dx + dvx*t_cpa, dy + dvy*t_cpa))
            approaching = (rv < 0.0)  # negative dot = moving toward each other
        
        # 1) Time urgency (0 at horizon, 1 now)
        time_factor = 1.0 - (t_cpa / (T + EPS))

        # 2) Distance factor: smooth "inside R ≈ 1, outside decays"
        #    Sigmoid centered at d=R, softness = k*R
        soft = max(EPS, k*R)  # avoid div-by-zero
        dist_factor = 1.0 / (1.0 + np.exp((d_cpa - R) / soft))  # in (0,1)

        # 3) Closing speed (normalized), only counts if approaching
        if r2 < EPS or v2 < EPS:
            closing_norm = 0.0
        else:
            closing_speed = max(0.0, - rv / (np.sqrt(r2) + EPS))   # m/s toward each other
            closing_norm  = closing_speed / (speed_mag + EPS)      # ~0..1

        # --- Optional true collision check: solve ||r + v t|| = R ---
        hit_bonus = 0.0
        if v2 >= EPS:
            a = v2
            b = 2.0 * rv
            c = r2 - R*R
            disc = b*b - 4*a*c
            if disc >= 0.0:
                s = float(np.sqrt(disc))
                t1 = (-b - s) / (2*a)
                t2 = (-b + s) / (2*a)
                # first future intersection within horizon (if any)
                candidates = [t for t in (t1, t2) if 0.0 <= t <= T]
                if candidates:
                    t_hit = min(candidates)
                    # bonus scales with how soon the hit occurs
                    hit_bonus = 0.2 * (1.0 - t_hit / (T + EPS))  # 0..0.2

        # --- Add current distance penalty for immediate threat prioritization ---
        current_distance = float(np.sqrt(r2))  # current separation distance
        # Penalty scales exponentially with distance (closer = less penalty)
        distance_penalty = min(1.0, current_distance / (2 * R))  # 0 at contact, 0.5 at 2*R, 1.0 at 4*R+
        
        # --- Combine (weighted sum), then penalize diverging pairs ---
        risk = (w_d * float(dist_factor)
            + w_t * float(time_factor)
            + w_c * float(closing_norm)
            + float(hit_bonus))
        
        # Apply current distance penalty - closer threats get higher priority
        risk *= (1.0 - 0.3 * distance_penalty)  # reduce risk by up to 30% for distant threats

        if not approaching:
            risk *= diverge_penalty  # down-weight if moving apart

        # clamp to [0,1]
        return float(max(0.0, min(1.0, risk)))
            

    def _get_observation(self, active_agents):
        dim = 3 + 8 * NUM_AC_STATE
        obs = {}
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
                cos_drift, sin_drift = np.cos(np.deg2rad(drift)), np.sin(np.deg2rad(drift))
                airspeed_kts = bs.traf.tas[ac_idx] * MpS2Kt
                vx = np.cos(np.deg2rad(ac_hdg)) * bs.traf.gs[ac_idx]
                vy = np.sin(np.deg2rad(ac_hdg)) * bs.traf.gs[ac_idx]

                ac_loc = fn.latlong_to_nm(CENTER, np.array([bs.traf.lat[ac_idx], bs.traf.lon[ac_idx]])) * NM2KM * 1000
                                
                # Build candidate list with CPA risk
                candidates = []
                for i in range(self.num_ac):
                    if i == ac_idx:
                        continue
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
                    risk = SectorEnv._cpa_risk(dx, dy, dvx, dvy)
                    candidates.append((risk, dx, dy, dvx, dvy, np.cos(trk), np.sin(trk), d_now))

                # Sort by descending risk; break ties by current distance (closer first)
                candidates.sort(key=lambda t: (-t[0], t[7]))
                top = candidates[:NUM_AC_STATE]

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
                    risk_arr,  # Risk values (normalized to [0,1])
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

        return obs

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
        progress_reward = distance_improvement * 2.0
        
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
          0 down to -PROXIMITY_MAX_PENALTY as aircraft get closer to the hard boundary
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
        return -PROXIMITY_MAX_PENALTY * float(ratio)
    
    def _check_intrusion(self, agent_id, ac_idx):
        """Return capped intrusion penalty for this agent on this step.

        Rationale: With many agents, an aircraft can be in conflict with
        multiple neighbors at the same time-step. Penalizing once per
        neighbor causes very large, highly variable negative spikes.
        Instead, cap the per-step penalty to a single application if any
        conflict exists (variance reduction), while still tracking all
        conflicts for metrics.
        """
        had_conflict = False
        step_penalty = 0.0
        for i in range(self.num_ac):
            if i == ac_idx:
                continue
            _, int_dis = bs.tools.geo.kwikqdrdist(
                bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], bs.traf.lat[i], bs.traf.lon[i]
            )
            if int_dis < INTRUSION_DISTANCE:
                had_conflict = True
                self.total_intrusions += 1  # keep full conflict count for logging
                if agent_id in self._intrusions_acc:
                    self._intrusions_acc[agent_id] += 1

        # Apply at most one intrusion penalty per agent per step
        if had_conflict:
            step_penalty += self.intrusion_penalty

        return step_penalty

    def update_intrusion_penalty(self, new_penalty: float):
        """Update the intrusion penalty value dynamically during training."""
        self.intrusion_penalty = new_penalty



