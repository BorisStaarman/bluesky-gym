# mvp_2d.py
import numpy as np
SAFE_DISTANCE_DEFAULT = 1 / 1852 * 100  # 100 meters in nautical miles
class MVP_2D:
    def __init__(self, safe_distance=SAFE_DISTANCE_DEFAULT, lookahead_time=15.0):
        """
        Args:
            safe_distance (float): Minimum separation distance (meters) - originally 'rpz'
            lookahead_time (float): Time horizon to check for collisions (seconds)
        """
        self.rpz = safe_distance  
        self.t_lookahead = lookahead_time 

    def calculate_avoidance_velocity(self, agent_pos, agent_vel, neighbors):
        """
        Calculates the MVP resolution velocity.
        
        Args:
            agent_pos: np.array([x, y])
            agent_vel: np.array([vx, vy])
            neighbors: list of dicts or objects, each having 'pos' and 'vel' attributes
                       (or simply a list of [pos, vel] arrays)
        
        Returns:
            np.array([new_vx, new_vy]) - The suggested safe velocity
        """
        # Start with a resolution vector of zero (no change needed yet)
        dv_total = np.zeros(2)
        
        # We need to sum up resolutions for all dangerous neighbors
        conflict_found = False

        for neighbor in neighbors:
            # 1. Unpack Neighbor State
            # Adjust these keys based on how your environment provides neighbor info
            neigh_pos = neighbor['pos'] # e.g., np.array([x, y])
            neigh_vel = neighbor['vel'] # e.g., np.array([vx, vy])
            
            # 2. Relative State
            d_rel = neigh_pos - agent_pos
            v_rel = neigh_vel - agent_vel
            
            # 3. Check for Time to Closest Point of Approach (tCPA)
            # Formula: t_cpa = -(d_rel . v_rel) / (|v_rel|^2)
            v_rel_sq = np.dot(v_rel, v_rel)
            
            # If relative velocity is near zero, they are moving parallel (no collision soon)
            if v_rel_sq < 1e-6:
                continue
                
            t_cpa = -np.dot(d_rel, v_rel) / v_rel_sq
            
            # 4. Filter: Is the collision in the future and within our lookahead?
            if t_cpa < 0 or t_cpa > self.t_lookahead:
                continue

            # 5. Calculate Distance at CPA (dCPA)
            # P_cpa = P_current + V * t
            # d_cpa vector = d_rel + v_rel * t_cpa
            d_cpa_vec = d_rel + v_rel * t_cpa
            d_cpa = np.linalg.norm(d_cpa_vec)
            
            # 6. Filter: Are they actually going to crash? (Distance < Safe Zone)
            if d_cpa >= self.rpz:
                continue
                
            # --- IF WE REACH HERE, A CONFLICT IS DETECTED ---
            conflict_found = True
            
            # 7. Apply MVP Resolution Math (Derived from your provided file)
            # Calculate Intrusion (how much they overlap)
            intrusion = self.rpz - d_cpa
            
            # Prevent division by zero if d_cpa is extremely small (head-on exact hit)
            if d_cpa < 1e-6:
                d_cpa = 1e-6
                # Force a side-step vector if perfectly head-on
                d_cpa_vec = np.array([-d_rel[1], d_rel[0]]) 
            
            # The MVP Formula:
            # Push the velocity perpendicular to the collision path
            # dv = (Intrusion * dCPA_Vector) / (abs(tCPA) * |dCPA_Vector|)
            
            factor = intrusion / (abs(t_cpa) * d_cpa)
            
            dv = d_cpa_vec * factor
            
            # Add to total resolution (in case of multiple neighbors)
            dv_total -= dv # Subtract because we want to move *away* from the intrusion logic

        # 8. Return the final Target Velocity
        # Original Velocity + Change Required
        if conflict_found:
            return agent_vel + dv_total
        else:
            # If no conflicts, just return current velocity (or preferred velocity)
            return agent_vel
