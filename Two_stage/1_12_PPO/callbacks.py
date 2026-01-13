from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.policy.sample_batch import SampleBatch
import numpy as np

class MVPDataBridgeCallback(DefaultCallbacks):
    """
    Callback to extract teacher actions from environment info and inject them
    into the training batch for Stage 1 imitation learning.
    """
    def on_postprocess_trajectory(
        self, worker, episode, agent_id, policy_id, 
        policies, postprocessed_batch, original_batches, **kwargs
    ):
        # Check if we have data for this agent
        if agent_id in original_batches:
            # Access the original 'infos' list from your environment
            original_infos = original_batches[agent_id][SampleBatch.INFOS]
            
            # Extract the teacher_action you saved in the step function
            # Use a default [0,0] if it's missing to prevent crashes
            teacher_actions = []
            for info in original_infos:
                if "teacher_action" in info:
                    teacher_action = info["teacher_action"]
                    # Ensure it's a numpy array with correct dtype
                    if not isinstance(teacher_action, np.ndarray):
                        teacher_action = np.array(teacher_action, dtype=np.float32)
                    teacher_actions.append(teacher_action)
                else:
                    # Default action if missing
                    teacher_actions.append(np.zeros(2, dtype=np.float32))
            
            # Convert to numpy array for batch processing
            teacher_actions_array = np.array(teacher_actions, dtype=np.float32)
            
            # Write it into the batch so the Loss Function can see it
            postprocessed_batch["teacher_targets"] = teacher_actions_array