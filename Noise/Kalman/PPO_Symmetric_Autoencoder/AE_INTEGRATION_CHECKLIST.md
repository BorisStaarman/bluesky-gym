# Autoencoder Integration Verification Prompt

I have integrated a pretrained Autoencoder into a multi-agent reinforcement learning environment for UAV conflict resolution.
The AE detects sensor noise by computing a reconstruction error signal that is appended to each agent's observation.
Please verify the following checklist end-to-end to confirm the implementation is consistent and correct.
The relevant files are attached.

---

## 1. AE Architecture — `bluesky_gym/autoencoder.py`

The `FlightAutoencoder` class must have topology: **20 → 8 → 3 → 8 → 20**

Check:
- `input_dim = 20` (= `AE_WINDOW_SIZE × AE_FEATURES = 5 × 4`)
- Encoder: `Linear(20, 8) → ReLU → Linear(8, 3) → ReLU`
- Decoder: `Linear(3, 8) → ReLU → Linear(8, 20)` — NO final activation
- `forward()` returns a reconstruction of the same shape as the input

---

## 2. Constants — must be identical in BOTH files

Check that these values match exactly between `bluesky_gym/autoencoder.py`
and `bluesky_gym/envs/ma_env_two_stage_AM_PPO_NOISE_autoencoder.py`:

| Constant        | Required value | Location         |
|-----------------|----------------|------------------|
| AE_WINDOW_SIZE  | 5              | both files       |
| AE_FEATURES     | 4              | both files       |
| AE_INPUT_DIM    | 20 (= 5×4)     | both files       |
| AE_DELTA_NORM   | 15.0  (m)      | both files       |
| AE_VEL_NORM     | 15.0  (m/s)    | both files       |
| AE_MSE_SCALE    | 0.08           | env file only    |

---

## 3. Sliding Window — `_update_obs_window` in the environment

Check:
- Called every timestep for each active agent, BEFORE `_compute_ae_noise_signal`
- Reads raw physical values from `self._noisy_states[ac_idx]`:
  `[x_m, y_m, vx_ms, vy_ms]` — metres and m/s, NOT arena-normalised
- Appends a length-4 `np.float32` array to `self._obs_windows[agent_id]`
  which is a `deque(maxlen=5)`
- The deque is initialised/reset inside `reset()` (not only in `__init__`)
  so each new episode starts with an empty window

---

## 4. Noise Signal Computation — `_compute_ae_noise_signal` in the environment

Check:
- Returns `0.0` immediately if `self._ae_model is None` or window has < 5 entries
- Builds `ae_input` of shape `(5, 4)` from the raw window using this scheme:
  - Timestep 0: `Δx = 0`, `Δy = 0`, `vx / AE_VEL_NORM`, `vy / AE_VEL_NORM`
  - Timesteps 1–4: `(x[t]-x[t-1]) / AE_DELTA_NORM`, `(y[t]-y[t-1]) / AE_DELTA_NORM`, `vx / AE_VEL_NORM`, `vy / AE_VEL_NORM`
- Flattens to shape `(1, 20)` as a `torch.float32` tensor
- Runs forward pass inside `torch.no_grad()`
- Computes `MSE = mean((input - reconstruction)²)`
- Returns `float(clip(MSE / AE_MSE_SCALE, 0.0, 1.0))`

---

## 5. Observation Vector Layout — `_get_observation` in the environment

The AE signal must be the **8th (last) element** of the ownship feature block:

```python
ownship_feats = np.array(
    [cos_drift, sin_drift, airspeed, dx, dy, vx / 36.0, vy / 36.0, ae_signal],
    dtype=np.float32
)
```

The full observation vector is then:
```
[8 ownship features] + [5 features × NUM_AC_STATE intruders]
```

The observation space declaration in `__init__` must reflect this:
```python
single_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8 + 5 * NUM_AC_STATE,), dtype=np.float32)
```
It must NOT use 7.

---

## 6. Policy Model — `attention_model_A.py`

Check:
- `self.ownship_dim = 8`  — NOT 7
- The query projection layer must accept 8 inputs:
  `nn.Linear(self.ownship_dim, self.head_dim, bias=True)` → i.e. `Linear(8, 5)`
- `self.intruder_dim = 5` (unchanged)

---

## 7. AE Model Loading — `load_autoencoder` in the environment

Check:
- `torch.load(model_path, map_location='cpu', weights_only=False)`
- `.eval()` called on the loaded model
- All parameters frozen: `for p in self._ae_model.parameters(): p.requires_grad = False`
- Prints a confirmation line containing `input_dim=20`

---

## 8. Training Script Wiring — `main.py`

Check:
1. `AE_MODEL_PATH` points to the `.pt` file in the same folder as the script:
   ```python
   AE_MODEL_PATH = os.path.join(script_dir, "autoencoder_pretrained.pt")
   ```
2. `env_config` dict contains `"autoencoder_path": AE_MODEL_PATH`
3. The evaluation `SectorEnv(...)` constructor call contains `autoencoder_path=AE_MODEL_PATH`
4. The `SectorEnv` class is imported from `ma_env_two_stage_AM_PPO_NOISE_autoencoder`
   (NOT the non-AE variant of the environment)

---

## 9. Pretrained Model File

Check:
- `autoencoder_pretrained.pt` physically exists in the same folder as `main.py`
- The model was trained on **clean (noise-free)** trajectory data
- It was trained using the same delta-normalised feature scheme described in point 4
  (if trained on raw absolute positions or different normalisations, the signal will be meaningless)
- Expected reconstruction MSE on clean data: ~0.001–0.003 (delta-normalised space)
  which maps to an AE signal of ~0.01–0.04 for clean flight
  versus ~0.4–1.0 for noisy flight (given AE_MSE_SCALE = 0.08)

---

## Files to Check

Please verify all of the above using these four files:

1. `bluesky_gym/autoencoder.py` — AE architecture and constants
2. `bluesky_gym/envs/ma_env_two_stage_AM_PPO_NOISE_autoencoder.py` — environment: obs construction, window logic, AE signal
3. `attention_model_A.py` — policy model: must have ownship_dim=8
4. `main.py` — training script: AE path, env_config wiring, correct env import

If none of the above points reveal an inconsistency, also check whether
`autoencoder_pretrained.pt` was actually saved with `torch.save(model, path)`
(whole-model save) rather than `torch.save(model.state_dict(), path)`
(weights-only save), because `load_autoencoder` uses `torch.load` which
expects the full model object, not a state dict.
