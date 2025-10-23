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
POLY_AREA_RANGE = (2, 3)
CENTER = np.array([51.990426702297746, 4.376124857109851])
# aircraft constants
ALTITUDE = 360
AC_SPD = 35
AC_TYPE = "m600"
INTRUSION_DISTANCE = 0.054
# conversion factors
NM2KM = 1.852
MpS2Kt = 1.94384
FT2M = 0.3048
# model settings
ACTION_FREQUENCY = 5
NUM_AC_STATE = 3
MAX_STEPS = 400

# penalties for reward
DRIFT_PENALTY = -0.1
STEP_PENALTY = -0.02
INTRUSION_PENALTY = -1 # was -1
# waypoint rewards
WAYPOINT_RADIUS = 0.01
WAYPOINT_REACHED_REWARD = 5 # was 50
PROGRESS_REWARD_SCALE = 0.5 # was 0.5

# constants to control actions
D_HEADING = 45
D_VELOCITY = 10/3

# normalization parameters
MAX_SCENARIO_DIM_M = 4000.0
DISTANCE_CENTER_M = 2000.0
DISTANCE_SCALE_M = 2000.0
MAX_RELATIVE_VEL_MS = 50.0
AIRSPEED_CENTER_KTS = 35.0
AIRSPEED_SCALE_KTS = 10.0

LOG_EVERY_N = 100  # throttle repeated warnings

class SectorEnv(MultiAgentEnv):
    metadata = {"name": "ma_env", "render_modes": ["rgb_array", "human"], "render_fps": 10}

    def __init__(self, render_mode=None, n_agents=10, run_id="default"):
        super().__init__()
        self.render_mode = render_mode
        self.num_ac = n_agents
        self.window_width = 512; self.window_height = 512; self.window_size = (self.window_width, self.window_height)
        self.poly_name = 'airspace'
        self._agent_ids = {f'kl00{i+1}'.upper() for i in range(n_agents)}
        self.agents = []
        
        # for multi agent csv files
        # --- per-agent CSV logging (safe for multi-worker) ---
        self.run_id = run_id or "default"
        self._flush_threshold = 25  # tune if you want more/less frequent writes
        self._agent_buffers = defaultdict(list)
        self._agent_episode_index = defaultdict(int)
        
        pid = os.getpid()
        self._metrics_root = os.path.join("metrics",  f"run_{self.run_id}", f"pid_{pid}")
        
        # wipe ONLY this PID folder once per process, then append during the run
        if not hasattr(SectorEnv, "_cleared_metric_dirs"):
            SectorEnv._cleared_metric_dirs = set()
        if self._metrics_root not in SectorEnv._cleared_metric_dirs and os.path.exists(self._metrics_root):
            shutil.rmtree(self._metrics_root)
            SectorEnv._cleared_metric_dirs.add(self._metrics_root)

        os.makedirs(self._metrics_root, exist_ok=True)
        print(f"[metrics] writing to: {os.path.abspath(self._metrics_root)}")
        

        # for evaluations
        self.total_intrusions = 0

        # stuff for making csv file
        self._agent_steps = {}
        self._rewards_acc = {}                 # per-agent running sums during the episode
        self._rewards_counts = {}              # per-agent step counts for safe averaging
        
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
        # for savin  data in csv file
        self._agent_steps = {a: 0 for a in self.agents}
        self._rewards_acc = {a: {"drift": 0.0, "progress": 0.0, "intrusion": 0.0} for a in self.agents}
        self._rewards_counts = {a: 0 for a in self.agents}
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
        rewards, infos = self._get_reward(agents_in_step)
        terminateds = self._get_terminateds(agents_in_step)
        truncateds = self._get_truncateds(agents_in_step)
        
        agents_to_remove = {agent for agent in agents_in_step if terminateds.get(agent, False) or truncateds.get(agent, False)}
        for a in agents_to_remove:
            n = max(1, self._rewards_counts.get(a, 0))
            m_drift    = self._rewards_acc.get(a, {}).get("drift", 0.0)    / n
            m_progress = self._rewards_acc.get(a, {}).get("progress", 0.0) / n
            m_intr     = self._rewards_acc.get(a, {}).get("intrusion", 0.0)/ n

            # increment per-agent episode index
            self._agent_episode_index[a] += 1

            # buffer one row for THIS agent only
            self._agent_buffers[a].append({
                "episode_index": self._agent_episode_index[a],
                "steps": self._agent_steps.get(a, 0),
                "mean_reward_drift":    m_drift,
                "mean_reward_progress": m_progress,
                "mean_reward_intrusion":m_intr,
                "sum_reward_drift":     self._rewards_acc.get(a, {}).get("drift", 0.0),
                "sum_reward_progress":  self._rewards_acc.get(a, {}).get("progress", 0.0),
                "sum_reward_intrusion": self._rewards_acc.get(a, {}).get("intrusion", 0.0),
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
            for a in agents_to_remove:
                self._flush_agent_buffer(a)
            
        return observations, rewards, terminateds, truncateds, infos
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
                    "sum_reward_drift",
                    "sum_reward_progress",
                    "sum_reward_intrusion",
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
                intrusion_reward = self._check_intrusion(ac_idx)
                progress_reward = self._check_progress(agent, ac_idx)
                
                rewards[agent] = drift_reward + intrusion_reward + progress_reward  + STEP_PENALTY
                
                # accumulate for per-episode stats
                self._rewards_acc[agent]["drift"]    += float(drift_reward)
                self._rewards_acc[agent]["progress"] += float(progress_reward)
                self._rewards_acc[agent]["intrusion"]+= float(intrusion_reward)
                self._rewards_counts[agent]          += 1
                
                infos[agent]["reward_drift"] = drift_reward
                infos[agent]["reward_intrusion"] = intrusion_reward
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
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            
    # def _get_observation(self, active_agents):
    #     obs = {}
    #     for agent in active_agents:
    #         try:
    #             ac_idx = bs.traf.id2idx(agent)
    #             wpts = self.agent_waypoints[agent]
    #             wpt_qdr, _ = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], wpts[0], wpts[1])
    #             ac_hdg = bs.traf.hdg[ac_idx]; drift = fn.bound_angle_positive_negative_180(ac_hdg - wpt_qdr)
    #             cos_drift, sin_drift = np.cos(np.deg2rad(drift)), np.sin(np.deg2rad(drift))
    #             airspeed_kts = bs.traf.tas[ac_idx] * MpS2Kt
    #             vx = np.cos(np.deg2rad(ac_hdg)) * bs.traf.gs[ac_idx]
    #             vy = np.sin(np.deg2rad(ac_hdg)) * bs.traf.gs[ac_idx]
    #             ac_loc = fn.latlong_to_nm(CENTER, np.array([bs.traf.lat[ac_idx], bs.traf.lon[ac_idx]])) * NM2KM * 1000
    #             dist = [fn.euclidean_distance(ac_loc, fn.latlong_to_nm(CENTER, np.array([bs.traf.lat[i], bs.traf.lon[i]])) * NM2KM * 1000) for i in range(self.num_ac)]
    #             ac_idx_by_dist = np.argsort(dist)
    #             x_r, y_r, vx_r, vy_r, cos_track, sin_track, distances = [], [], [], [], [], [], []
    #             count = 0
    #             for i in ac_idx_by_dist:
    #                 if i == ac_idx: continue
    #                 if count >= NUM_AC_STATE: break
    #                 int_hdg = bs.traf.hdg[i]
    #                 int_loc = fn.latlong_to_nm(CENTER, np.array([bs.traf.lat[i], bs.traf.lon[i]])) * NM2KM * 1000; x_r.append(int_loc[0] - ac_loc[0])
    #                 y_r.append(int_loc[1] - ac_loc[1])
    #                 vx_int = np.cos(np.deg2rad(int_hdg)) * bs.traf.gs[i]
    #                 vy_int = np.sin(np.deg2rad(int_hdg)) * bs.traf.gs[i]
    #                 vx_r.append(vx_int - vx)
    #                 vy_r.append(vy_int - vy)
    #                 track = np.arctan2(vy_int - vy, vx_int - vx)
    #                 cos_track.append(np.cos(track))
    #                 sin_track.append(np.sin(track))
    #                 distances.append(dist[i])
    #                 count += 1
    #             obs[agent] = np.concatenate([v.flatten() for v in {
    #                 "cos(drift)": np.array([cos_drift]), "sin(drift)": np.array([sin_drift]),
    #                 "airspeed": np.array([(airspeed_kts - AIRSPEED_CENTER_KTS) / AIRSPEED_SCALE_KTS]),
    #                 "x_r": np.array(x_r) / MAX_SCENARIO_DIM_M, "y_r": np.array(y_r) / MAX_SCENARIO_DIM_M,
    #                 "vx_r": np.array(vx_r) / MAX_RELATIVE_VEL_MS, "vy_r": np.array(vy_r) / MAX_RELATIVE_VEL_MS,
    #                 "cos(track)": np.array(cos_track), "sin(track)": np.array(sin_track),
    #                 "distances": (np.array(distances) - DISTANCE_CENTER_M) / DISTANCE_SCALE_M
    #             }.values()]).astype(np.float32)
    #         except Exception: obs[agent] = np.zeros(3 + 7 * NUM_AC_STATE, dtype=np.float32)
    #     return obs

    def _get_observation(self, active_agents):
        dim = 3 + 7 * NUM_AC_STATE
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

                # --- your existing feature computations (unchanged) ---
                wpt_lat, wpt_lon = self.agent_waypoints[agent]
                wpt_qdr, _ = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], wpt_lat, wpt_lon)
                ac_hdg = bs.traf.hdg[ac_idx]
                drift = fn.bound_angle_positive_negative_180(ac_hdg - wpt_qdr)
                cos_drift, sin_drift = np.cos(np.deg2rad(drift)), np.sin(np.deg2rad(drift))
                airspeed_kts = bs.traf.tas[ac_idx] * MpS2Kt
                vx = np.cos(np.deg2rad(ac_hdg)) * bs.traf.gs[ac_idx]
                vy = np.sin(np.deg2rad(ac_hdg)) * bs.traf.gs[ac_idx]

                ac_loc = fn.latlong_to_nm(CENTER, np.array([bs.traf.lat[ac_idx], bs.traf.lon[ac_idx]])) * NM2KM * 1000
                dist = [fn.euclidean_distance(ac_loc, fn.latlong_to_nm(CENTER, np.array([bs.traf.lat[i], bs.traf.lon[i]])) * NM2KM * 1000)
                        for i in range(self.num_ac)]
                ac_idx_by_dist = np.argsort(dist)

                x_r, y_r, vx_r, vy_r, cos_track, sin_track, distances = [], [], [], [], [], [], []
                count = 0
                for i in ac_idx_by_dist:
                    if i == ac_idx:
                        continue
                    if count >= NUM_AC_STATE:
                        break
                    int_hdg = bs.traf.hdg[i]
                    int_loc = fn.latlong_to_nm(CENTER, np.array([bs.traf.lat[i], bs.traf.lon[i]])) * NM2KM * 1000
                    x_r.append(int_loc[0] - ac_loc[0]); y_r.append(int_loc[1] - ac_loc[1])
                    vxi = np.cos(np.deg2rad(int_hdg)) * bs.traf.gs[i]
                    vyi = np.sin(np.deg2rad(int_hdg)) * bs.traf.gs[i]
                    vx_r.append(vxi - vx); vy_r.append(vyi - vy)
                    trk = np.arctan2(vyi - vy, vxi - vx)
                    cos_track.append(np.cos(trk)); sin_track.append(np.sin(trk))
                    distances.append(dist[i]); count += 1

                parts = [
                    np.array([cos_drift]), np.array([sin_drift]),
                    np.array([(airspeed_kts - AIRSPEED_CENTER_KTS) / AIRSPEED_SCALE_KTS]),
                    np.array(x_r) / MAX_SCENARIO_DIM_M, np.array(y_r) / MAX_SCENARIO_DIM_M,
                    np.array(vx_r) / MAX_RELATIVE_VEL_MS, np.array(vy_r) / MAX_RELATIVE_VEL_MS,
                    np.array(cos_track), np.array(sin_track),
                    (np.array(distances) - DISTANCE_CENTER_M) / DISTANCE_SCALE_M,
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
        if current_dist < WAYPOINT_RADIUS:
            if agent_id not in self.waypoint_reached_agents:
                self.waypoint_reached_agents.add(agent_id)
                return WAYPOINT_REACHED_REWARD
            else:
                return 0.0
        prev_dist = self.previous_distances.get(agent_id, current_dist)
        distance_delta = prev_dist - current_dist
        progress_reward = distance_delta * PROGRESS_REWARD_SCALE
        self.previous_distances[agent_id] = current_dist
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
    
    def _check_intrusion(self, ac_idx):
        reward = 0
        for i in range(self.num_ac):
            if i == ac_idx: continue
            _, int_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], bs.traf.lat[i], bs.traf.lon[i])
            if int_dis < INTRUSION_DISTANCE: 
                self.total_intrusions += 1
                reward += INTRUSION_PENALTY
        return reward


