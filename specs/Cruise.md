# Atlas-Cruise: ultra-simple adaptive cruise + lane-centering system

## 0. Purpose

Implement a compact, practical GTA V driving-assist system named:

```text
Atlas-Cruise
```

`Atlas-Cruise` is **not** a full autonomous-driving foundation model. It is a simple monocular lane-centering and adaptive-cruise system.

Primary goals:

```text
1. Keep the vehicle centered in the current visible road/lane corridor.
2. Follow clearly visible lane markings, curbs, road edges, barriers, or pavement boundaries.
3. Maintain speed when the road ahead is clear.
4. Slow or brake when the demonstrated / learned behavior indicates traffic or obstacle risk ahead.
5. Track a predicted future trajectory using a separate classical controller.
6. Use minimal GTA privileged ego odometry only for training labels, not as a required inference input.
```

Explicit non-goals:

```text
1. No lane changes.
2. No route following.
3. No traffic-light or stop-sign handling.
4. No intersection planning.
5. No pedestrian-heavy urban driving.
6. No full world model.
7. No LiDAR/depth/semantic labels.
8. No lane graph extraction.
9. No actor-state extraction.
10. No dense occupancy.
11. No general planner/evaluator.
```

The core model should be:

```text
video + action history
  -> compact pretrained visual encoder
  -> temporal model
  -> ego-speed/yaw estimate
  -> future ego trajectory
  -> trajectory legalizer
  -> classical controller
  -> steer/throttle/brake
```

Training labels come from:

```text
video + controls + minimal ego odometry
```

The minimal ego odometry is used only to derive trajectory and ego-motion targets.

---

# 1. Main design decision

Do **not** train pure direct video-to-control as the main model.

Use:

```text
video/actions -> future ego trajectory + ego kinematics
trajectory -> controller -> GTA controls
```

This keeps the model interpretable and allows a deterministic controller/legalizer to enforce speed/curve constraints.

Direct video-to-control should only be implemented as a debugging baseline.

Why trajectory is preferred:

```text
1. Trajectory targets are smoother than raw human controls.
2. The controller can smooth and legalize outputs.
3. Speed can be clamped for curves and poor visibility.
4. Steering oscillation is easier to debug.
5. The model can be evaluated geometrically before closed-loop driving.
6. Minimal ego odometry is much easier than full privileged labeling.
```

---

# 2. Minimal privileged data policy

Atlas-Cruise should use only the following privileged game-state signals during data collection/training:

```text
ego position
ego rotation/yaw
ego velocity or speed
timestamp
optional camera pose/FOV for diagnostics
```

Do **not** require:

```text
LiDAR
depth
semantics
actor states
traffic lights
traffic signs
lane graphs
map elements
occupancy
road topology
```

At inference, the deployable model should be able to run from:

```text
front RGB
actions
dt/timestamps
UI/freshness metadata
```

The controller should use the model-predicted ego speed/yaw estimate by default. A debug mode may optionally use privileged ego speed if the GTA odometry plugin is running, but that must be clearly marked as non-deployable / privileged.

---

# 3. ODD

## 3.1 Supported

Atlas-Cruise should work on:

```text
1. Highways.
2. Freeways.
3. Divided roads.
4. Well-marked roads.
5. Roads with clear lane markings.
6. Roads with curbs or pavement edges.
7. Roads with dividers/barriers.
8. Daytime.
9. Nighttime.
10. Rain.
11. Moderate traffic directly ahead.
12. First-person hood camera.
13. Fixed 16:9 GTA window.
```

## 3.2 Unsupported

Atlas-Cruise should not be expected to handle:

```text
1. Traffic lights.
2. Stop signs.
3. Intersections.
4. Dense urban turns.
5. Parking lots.
6. Off-road dirt trails.
7. Route navigation.
8. Lane changes.
9. Overtaking.
10. Pedestrian-heavy scenes.
11. Complex obstacle negotiation.
12. Unusual camera modes.
13. Manual camera looking around.
```

## 3.3 Behavior outside main ODD

If the road is not a highway but is still clearly road-like and lane-followable, Atlas-Cruise should continue in caution mode.

Examples:

```text
well-marked mountain road -> continue slowly
clear single-lane road -> continue slowly
ambiguous but road-like scene -> slow down
dirt trail / forest / no visible road -> fallback / takeover
```

---

# 4. Coordinate convention

Use longitudinal/lateral coordinate naming.

```text
long_m = longitudinal position in meters, positive forward from ego
lat_m  = lateral position in meters, positive left of ego, negative right of ego
yaw_rad = yaw relative to current ego heading, radians
speed_mps = speed, meters per second
```

Coordinate frame:

```text
long_m > 0  means ahead of ego
lat_m  > 0  means left of ego
lat_m  < 0  means right of ego
yaw_rad > 0 means leftward / counterclockwise rotation
```

Trajectory point layout:

```text
[long_m, lat_m, yaw_rad, speed_mps]
```

---

# 5. Global tensor notation

```text
B = batch size
T = number of sampled visual frames in a training window
C = image channels, usually 3
H = model input height
W = model input width
D = hidden dimension
F = visual tokens per frame
S = cached historical frame-token count at inference
M = sampled action-history length
A = action vector dimension
R = number of action-summary tokens
N = number of future trajectory samples
Hc = optional auxiliary control horizon length
```

Recommended defaults:

```text
B: training-dependent
T: 8 to 16
C: 3
H,W: 540 × 960 default, 720 × 1280 stretch
D: 128
F: 32 or 64
S: 100 for 5 s at 20 Hz, optional 200 for 10 s
M: 100 for 5 s at 20 Hz
A: 6
R: 4
N: 30
Hc: 5
```

---

# 6. System architecture overview

Runtime pipeline:

```text
GTA display capture
  -> UI/freshness preprocessing
  -> resize/crop to model resolution
  -> Atlas-Cruise model
  -> trajectory sanitizer
  -> trajectory legalizer / speed profiler
  -> classical lateral + longitudinal controller
  -> GTA action output
```

Training pipeline:

```text
RGB video + controls + minimal ego odometry
  -> build ego trajectory targets
  -> train video/action model to predict future trajectory and ego kinematics
  -> validate offline
  -> deploy closed-loop
  -> collect interventions/recovery
  -> fine-tune
```

---

# 7. Source capture and model input

## 7.1 Source capture

Collect:

```text
RGB video: 1920×1080 @ 60 Hz target
actions: 60 Hz target
ego odometry: 60 Hz or higher if possible
timestamps: high-resolution host timestamps
```

## 7.2 Model input resolution

Default:

```text
960 × 540
```

Stretch:

```text
1280 × 720
```

Do not start this simplified model at 1536×864 or full 1080p. Those resolutions are more appropriate for the larger highway-assist model with explicit lane/lead heads.

For Atlas-Cruise, `960×540` is a better starting point because:

```text
1. The task is narrow.
2. The model should be easy to train.
3. The dataset is small.
4. The visual input is mainly lane geometry and direct lead behavior.
5. Lower resolution permits more temporal context and faster closed-loop iteration.
```

---

# 8. GTA ego odometry extraction

## 8.1 Recommendation

Implement a custom GTA single-player plugin that logs minimal ego odometry.

Recommended implementation:

```text
C++ ScriptHookV ASI plugin
```

Alternative for faster prototyping:

```text
ScriptHookVDotNet C# script
```

DeepGTAV is not necessary for Atlas-Cruise. DeepGTAV is more useful when extracting richer ground truth such as depth, LiDAR, object annotations, or semantic labels. Atlas-Cruise only needs ego pose/speed, so a custom lightweight plugin is simpler.

Script Hook V is specifically a library for accessing GTA V script native functions in custom `.asi` plugins, and it does not work in GTA Online. That is appropriate here because Atlas-Cruise should be developed and tested in single-player GTA V. ([dev-c.com](https://www.dev-c.com/gtav/))

## 8.2 Why not use full DeepGTAV?

DeepGTAV is useful but overkill for this simplified system. It is described as a plugin that transforms GTA V into a vision-based self-driving-car research environment, and related forks can extract richer ground-truth data. But for Atlas-Cruise, the required signal is only ego pose/speed/yaw, so integrating the entire DeepGTAV stack is unnecessary unless the custom plugin path fails. ([github.com](https://github.com/aitorzip/deepgtav))

Use DeepGTAV only as:

```text
1. Reference code.
2. Backup extractor.
3. Future richer-label system.
```

## 8.3 ScriptHookV C++ plugin signals

Create:

```text
gtapilot/native/ego_telemetry/
```

The plugin should run every game tick or at a fixed high-rate timer.

For each sample, log:

```python
{
    "host_qpc_ns": int,
    "game_time_ms": int,
    "frame_counter": int,

    "vehicle_valid": bool,
    "vehicle_model_hash": int,

    "ego_pos_world": [float, float, float],
    "ego_rot_world": [float, float, float],   # roll, pitch, yaw or game-native order
    "ego_heading_yaw": float,

    "ego_velocity_world": [float, float, float],
    "ego_speed_mps": float,
    "ego_speed_forward_mps": float,
    "ego_speed_right_mps": float,

    "camera_pos_world": [float, float, float],     # optional
    "camera_rot_world": [float, float, float],     # optional
    "camera_fov_deg": float,                       # optional
}
```

Useful GTA native functions are available through the native API surface. For example, `GET_ENTITY_SPEED` returns speed in meters per second, while entity coordinate/rotation/velocity functions are exposed through the native database. ([docs.fivem.net](https://docs.fivem.net/natives/?_0xD5037BA82E12416F=), [docs.fivem.net](https://docs.fivem.net/natives/?_0x3FEF770D40960D5A=), [docs.fivem.net](https://docs.fivem.net/natives/?_0xAFBD61CC738D9EB9=))

Recommended native-level data:

```text
PlayerPedId()
GetVehiclePedIsIn(player_ped, false)
GET_ENTITY_COORDS(vehicle, true)
GET_ENTITY_ROTATION(vehicle, rotationOrder=2)
GET_ENTITY_HEADING(vehicle)
GET_ENTITY_VELOCITY(vehicle)
GET_ENTITY_SPEED(vehicle)
GET_ENTITY_SPEED_VECTOR(vehicle, relative=true)
GET_ENTITY_FORWARD_VECTOR(vehicle)
GET_GAME_TIMER()
```

The FiveM native docs also expose `GET_ENTITY_SPEED_VECTOR`, which is useful because relative speed can indicate forward/reverse components in the entity’s local frame. ([docs.fivem.net](https://docs.fivem.net/natives/?_0x9A8D700A51CB7B0D=))

## 8.4 ScriptHookVDotNet alternative

A C# ScriptHookVDotNet script can be faster to prototype. ScriptHookVDotNet is an ASI plugin that allows scripts written in .NET languages to run in-game. ([github.com](https://github.com/scripthookvdotnet/scripthookvdotnet))

The SHVDN `Vehicle` class exposes vehicle-related values including speed/wheel speed/forward speed style properties in its public docs. ([nitanmarcel.github.io](https://nitanmarcel.github.io/shvdn-docs.github.io/class_g_t_a_1_1_vehicle.html))

Recommended SHVDN prototype fields:

```text
Game.GameTime
Game.Player.Character.CurrentVehicle.Position
Vehicle.Rotation
Vehicle.Heading
Vehicle.Velocity
Vehicle.Speed / ForwardSpeed / WheelSpeed if available
Vehicle.Model.Hash
```

Use SHVDN for fast proof-of-concept. Use C++ ASI if timing, overhead, or binary IPC precision becomes important.

## 8.5 Plugin-to-GTA-Pilot transport

The ego telemetry plugin should send odometry to the Python system using one of:

```text
1. ZeroMQ PUB socket
2. UDP localhost packets
3. shared memory ring buffer
4. newline JSONL file for debugging only
```

Recommended production approach:

```text
ZeroMQ PUB or shared memory ring buffer
```

Channel name:

```text
ego.telemetry
```

Message rate:

```text
60 Hz minimum
```

Message format:

```python
@dataclass
class EgoTelemetrySample:
    host_qpc_ns: int
    game_time_ms: int
    valid: bool

    pos_x: float
    pos_y: float
    pos_z: float

    rot_roll: float
    rot_pitch: float
    rot_yaw: float

    heading_yaw: float

    vel_x: float
    vel_y: float
    vel_z: float

    speed_mps: float
    forward_speed_mps: float
    lateral_speed_mps: float

    vehicle_model_hash: int
```

All timestamps should be based on the same host high-resolution clock used by the display capture system where possible. If the plugin cannot use the same exact clock, include enough fields to align post-hoc:

```text
host_qpc_ns
game_time_ms
receive_time_ns
```

---

# 9. Dataset format

## 9.1 Required logs

Each recording should contain:

```text
video file
frame metadata
action stream
ego telemetry stream
session metadata
```

Suggested files:

```text
recording_id_video.mkv
recording_id_frames.jsonl
recording_id_actions.jsonl
recording_id_ego.jsonl
recording_id_session.json
```

## 9.2 Frame metadata

Per displayed frame:

```python
{
    "frame_id": int,
    "capture_timestamp_ns": int,
    "source_frame_id": int,
    "is_fresh": bool,
    "frame_age_s": float,
    "video_frame_index": int,
    "width": int,
    "height": int
}
```

## 9.3 Action metadata

Per action sample:

```python
{
    "timestamp_ns": int,
    "steer": float,
    "throttle": float,
    "brake": float,
    "handbrake": float,
    "reverse": float,
    "pilot_active": float
}
```

## 9.4 Ego metadata

Per ego sample:

```python
{
    "host_qpc_ns": int,
    "game_time_ms": int,
    "valid": bool,

    "ego_pos_world": [float, float, float],
    "ego_rot_world": [float, float, float],
    "ego_heading_yaw": float,

    "ego_velocity_world": [float, float, float],
    "ego_speed_mps": float,
    "ego_speed_forward_mps": float,
    "ego_speed_lateral_mps": float,

    "vehicle_model_hash": int
}
```

## 9.5 Session metadata

```python
{
    "recording_id": str,
    "map": "GTA_V",
    "camera_mode": "first_person_hood",
    "source_resolution": [1920, 1080],
    "nominal_capture_hz": 60,
    "vehicle_model_hash": int,
    "weather": str,
    "time_of_day": str,
    "notes": str
}
```

---

# 10. Target trajectory generation

Implement:

```text
gtapilot/atlas-cruise/targets.py
```

## 10.1 Target trajectory shape

```python
target_traj : [B, N, 4]
```

Default:

```text
N = 30
horizon_s = 4.5
traj_dt_s = 4.5 / 30 = 0.15 s
```

Each target point:

```text
[long_m, lat_m, yaw_rad, speed_mps]
```

## 10.2 Base timestamp

For a frame at time `t_frame`, the target should start from:

```text
t_base = t_frame + target_latency_s
```

Default:

```python
target_latency_s = 0.10
```

This approximates the delay between visual observation and control application.

## 10.3 Pose interpolation

The target builder should interpolate ego pose at:

```text
t_base
t_base + 1 * traj_dt_s
t_base + 2 * traj_dt_s
...
t_base + (N-1) * traj_dt_s
```

Use:

```text
linear interpolation for position
slerp or angle-unwrapped interpolation for yaw/orientation
linear interpolation for speed
```

## 10.4 Transform into ego frame

Let:

```text
P0 = ego pose at t_base
Pi = ego pose at future sample i
```

Compute relative pose:

```text
relative_position = inverse(P0) * Pi.position
relative_yaw = wrap_angle(Pi.yaw - P0.yaw)
```

Then:

```text
long_m = forward component of relative_position
lat_m  = left component of relative_position
yaw_rad = relative_yaw
speed_mps = interpolated future ego speed
```

## 10.5 Validity masks

Generate:

```python
target_traj_valid : [B, N]
```

Invalid if:

```text
future sample missing
ego telemetry invalid
vehicle not valid
recording ended
teleport/reset detected
crash detected if possible
large pose discontinuity
```

## 10.6 Smoothing

Before target generation or after target generation, smooth ego pose and speed slightly:

```text
small Savitzky-Golay or low-pass smoothing
unwrap yaw before smoothing
do not oversmooth curves
```

Target smoothing is important because the model should learn stable paths rather than frame-to-frame jitter.

---

# 11. Input preprocessing

Implement:

```text
gtapilot/atlas-cruise/ui.py
gtapilot/atlas-cruise/preprocess.py
```

## 11.1 UI sanitizer

Inputs:

```text
raw 1920×1080 RGB
```

Outputs:

```python
scene_rgb : [3, H, W]
ui_mask   : [1, H, W]
ui_state  : UIState
```

Default:

```text
H,W = 540×960
```

Handle:

```text
minimap
top-left popup
phone bottom-right
center reticle
```

For Atlas-Cruise:

```text
mask/neutralize minimap
mask/neutralize phone while visible
mask/neutralize top-left popup if detected
erase tiny center reticle if detected/configured
```

No minimap route branch is needed.

## 11.2 Resize

```text
raw 1920×1080 -> scene 960×540 default
```

Use bilinear resizing.

Normalize with ImageNet statistics if using pretrained ImageNet RegNet:

```text
mean/std from torchvision weights metadata
```

Codex should use the preprocessing transform associated with the selected TorchVision weights where practical.

---

# 12. Model architecture

Create:

```text
gtapilot/atlas-cruise/
```

Files:

```text
__init__.py
config.py
model.py
backbone.py
temporal.py
action_encoder.py
heads.py
state.py
preprocess.py
ui.py
targets.py
losses.py
dataset.py
train.py
live_cruise.py
controller.py
trajectory_legalizer.py
visualization.py
```

## 12.1 Config

```python
@dataclass
class AtlasCruiseConfig:
    # Source/input.
    source_h: int = 1080
    source_w: int = 1920
    input_h: int = 540
    input_w: int = 960
    input_channels: int = 3

    # Runtime.
    model_hz: float = 20.0
    controller_hz: float = 60.0

    # Visual context for training.
    visual_context_s: float = 3.5
    num_visual_frames: int = 16

    # Action context.
    action_context_s: float = 5.0
    action_sample_hz: float = 20.0
    action_dim: int = 6

    # Backbone.
    backbone_name: str = "regnet_y_800mf"
    pretrained_backbone: bool = True
    hidden_dim: int = 128
    tokens_per_frame: int = 32

    # Temporal model.
    action_tokens: int = 4
    temporal_layers: int = 2
    temporal_heads: int = 4
    temporal_mlp_ratio: float = 2.0

    # Trajectory output.
    traj_points: int = 30
    traj_horizon_s: float = 4.5
    target_latency_s: float = 0.10

    # Optional auxiliary control output.
    predict_control_aux: bool = True
    control_horizon_steps: int = 5
    control_dt_s: float = 0.10

    # UI/freshness.
    use_ui_mask: bool = True
    use_frame_freshness: bool = True

    # Training regularization.
    action_dropout_prob: float = 0.15
    rgb_aug_prob: float = 0.5
```

## 12.2 Model forward signature

```python
class AtlasCruise(nn.Module):
    def forward(
        self,
        rgb_recent: torch.Tensor,          # [B, T, 3, H, W]
        actions_hist: torch.Tensor,        # [B, M, 6]
        dt_hist: torch.Tensor,             # [B, M, 1]
        ui_mask_recent: torch.Tensor | None = None,   # [B, T, 1, H, W]
        frame_freshness: torch.Tensor | None = None,  # [B, T, 2]
        state: AtlasCruiseState | None = None,
    ) -> dict[str, torch.Tensor]:
        ...
```

Return:

```python
{
    "traj": torch.Tensor,                 # [B, N, 4]
    "ego_kinematics": torch.Tensor,       # [B, 4]
    "traj_conf_logit": torch.Tensor,      # [B, 1]
    "slow_or_brake_logit": torch.Tensor,  # [B, 1]
    "fallback_logit": torch.Tensor,       # [B, 1]

    # optional auxiliary behavior-cloning head
    "control_aux": torch.Tensor,          # [B, Hc, 3]
}
```

## 12.3 Output definitions

`traj`:

```text
[B, N, 4]
```

Each point:

```text
[long_m, lat_m, yaw_rad, speed_mps]
```

`ego_kinematics`:

```text
[B, 4] = [speed_mps, a_long_mps2, yaw_rate_radps, curvature_inv_m]
```

`traj_conf_logit`:

```text
confidence that predicted trajectory is reliable
```

`slow_or_brake_logit`:

```text
probability that the vehicle should slow/brake due to front context
```

`fallback_logit`:

```text
probability that model/controller should enter fallback or request intervention
```

`control_aux`:

```text
optional auxiliary imitation output [steer, throttle, brake]
```

Runtime should primarily use `traj`, not `control_aux`.

---

# 13. Backbone

## 13.1 Recommended default

Use:

```text
RegNetY-800MF pretrained
```

Alternative faster option:

```text
RegNetY-400MF if available
```

Stretch:

```text
RegNetY-1.6GF
```

RegNetY-800MF is available in TorchVision with pretrained weights and metadata, making it a reasonable low-data visual backbone. ([docs.pytorch.org](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.regnet_y_800mf.html))

## 13.2 Feature extraction

Remove classifier.

Use intermediate features if easy:

```text
c3/c4/c5
```

or final feature only for simplest MVP.

Recommended tokenizer:

```text
pool feature maps into fixed grids
project to hidden_dim
learned-query pool to tokens_per_frame
```

Default:

```text
hidden_dim = 128
tokens_per_frame = 32
```

Per-frame visual output:

```python
frame_tokens  : [B, T, 32, 128]
frame_summary : [B, T, 128]
```

---

# 14. Temporal model

## 14.1 Visual temporal input

Recommended frame offsets over 3.5 seconds:

```python
visual_offsets_s = [
    0.00,
   -0.10,
   -0.20,
   -0.35,
   -0.50,
   -0.70,
   -0.90,
   -1.10,
   -1.35,
   -1.60,
   -1.90,
   -2.20,
   -2.50,
   -2.75,
   -3.00,
   -3.50,
]
```

Training samples frames by timestamp.

Live inference can either:

```text
1. re-encode selected frames for a simple first version, or
2. encode current frame and cache frame summaries/tokens.
```

Recommended live mode:

```text
encode current frame only
cache frame_summary and frame_tokens
sample cached features by timestamp offsets
```

## 14.2 Action temporal input

Sample action history:

```text
5 seconds at 20 Hz = 100 action samples
```

Input:

```python
actions_hist : [B, 100, 6]
dt_hist      : [B, 100, 1]
```

Encode to:

```python
action_tokens : [B, 4, 128]
```

## 14.3 Temporal transformer

Simplest temporal sequence:

```python
temporal_input = concat(frame_summary_sequence, action_tokens)
```

Shape:

```python
frame_summary_sequence : [B, 16, 128]
action_tokens          : [B, 4, 128]
temporal_input         : [B, 20, 128]
```

Use:

```text
2 transformer encoder layers
4 heads
MLP ratio 2
pre-norm
dropout 0.1
```

Output:

```python
policy_context : [B, 128]
```

Optionally include a learned `[CLS]` token.

---

# 15. Heads

## 15.1 Trajectory head

MLP:

```text
policy_context [B,128] -> traj [B,N,4]
```

Default:

```text
N = 30
output dim = 30 × 4 = 120
```

## 15.2 Ego head

```text
policy_context [B,128] -> ego_kinematics [B,4]
```

## 15.3 Confidence heads

```text
policy_context [B,128] -> traj_conf_logit [B,1]
policy_context [B,128] -> slow_or_brake_logit [B,1]
policy_context [B,128] -> fallback_logit [B,1]
```

## 15.4 Optional auxiliary control head

```text
policy_context [B,128] -> control_aux [B,Hc,3]
```

Default:

```text
Hc = 5
output dim = 5 × 3 = 15
```

`control_aux` layout:

```text
[steer, throttle, brake]
```

This is an auxiliary training signal, not the primary runtime output.

---

# 16. Trajectory legalizer and controller

Implement:

```text
gtapilot/atlas-cruise/trajectory_legalizer.py
gtapilot/atlas-cruise/controller.py
```

## 16.1 Trajectory sanitizer

Before controller:

```text
1. clamp speed >= 0
2. smooth trajectory
3. reject NaNs
4. enforce reasonable forward progress
5. recompute yaw from long/lat if necessary
```

## 16.2 Curve speed cap

Compute curvature:

```text
kappa = path curvature, 1/m
```

Lateral acceleration:

```text
a_lat = speed^2 * abs(kappa)
```

Speed cap:

```text
v_cap_curve = sqrt(a_lat_limit / (abs(kappa) + eps))
```

Use conservative default:

```text
a_lat_limit = 3.0 m/s² comfort
a_lat_hard = 5.0 m/s² hard
rain multiplier = 0.75
```

## 16.3 Authorized speed

```text
authorized_speed = min(
    predicted_speed,
    curve_speed_cap,
    user_max_speed,
    caution_speed_cap
)
```

If `slow_or_brake_logit` is high:

```text
reduce authorized speed
allow brake
```

If `fallback_logit` is high or frame stale:

```text
minimum-risk slowdown
```

## 16.4 Lateral controller

Use pure pursuit or Stanley.

Input:

```python
traj : [N,4]
ego_kinematics
```

Output:

```text
steer
```

## 16.5 Longitudinal controller

Use PID/ACC-like speed tracking.

Input:

```text
authorized_speed profile
predicted ego speed
slow_or_brake signal
fallback signal
```

Output:

```text
throttle
brake
```

## 16.6 Control postprocessing

Always apply:

```text
steering smoothing
steering rate limit
throttle/brake conflict suppression
throttle/brake smoothing
frame-staleness slowdown
fallback slowdown
```

---

# 17. Training targets

## 17.1 Required targets

```python
target_traj          : [B, N, 4]
target_traj_valid    : [B, N]
target_ego           : [B, 4]
target_control_aux   : [B, Hc, 3] optional
target_slow_or_brake : [B, 1]
target_fallback      : [B, 1]
```

## 17.2 Target trajectory

Derived from minimal ego odometry.

For each image timestamp `t`:

```text
t_base = t + target_latency_s
```

For each future step `i`:

```text
t_i = t_base + i * traj_dt_s
```

Interpolate ego pose at `t_base` and `t_i`.

Transform future pose into ego frame at `t_base`.

Output:

```text
[long_m, lat_m, yaw_rad, speed_mps]
```

## 17.3 Ego target

From odometry:

```text
speed_mps
a_long_mps2
yaw_rate_radps
curvature_inv_m
```

Compute:

```text
a_long from speed derivative
yaw_rate from yaw derivative
curvature = yaw_rate / max(speed, eps)
```

Smooth these targets.

## 17.4 Slow/brake target

Use controls and future speed drop.

Set `target_slow_or_brake = 1` if:

```text
brake control above threshold
or future speed drops significantly
or target trajectory speed profile decreases rapidly
```

Example threshold:

```text
brake > 0.15
or speed[t+1s] < speed[t] - 3 m/s
```

## 17.5 Fallback target

Set fallback target only for invalid/hard cases:

```text
offroad
crash
menu/pause
no valid road
very stale frame burst
manual intervention after model failure in DAgger data
```

Most ordinary curves or non-highway marked roads should have:

```text
fallback = 0
```

---

# 18. Losses

Implement:

```text
gtapilot/atlas-cruise/losses.py
```

## 18.1 Trajectory loss

```text
L_traj_pos = Huber(pred_long_lat, target_long_lat)
L_traj_yaw = Huber(pred_yaw, target_yaw)
L_traj_speed = Huber(pred_speed, target_speed)
```

Use `target_traj_valid`.

Suggested weights:

```text
long/lat: 2.0
yaw: 0.5
speed: 1.0
```

## 18.2 Ego loss

```text
L_ego = Huber(pred_ego, target_ego)
```

## 18.3 Slow/brake loss

```text
L_slow = BCEWithLogits(slow_or_brake_logit, target_slow_or_brake)
```

## 18.4 Fallback loss

```text
L_fallback = BCEWithLogits(fallback_logit, target_fallback)
```

Weight fallback carefully because positives are rare.

## 18.5 Auxiliary control loss

If enabled:

```text
L_control = Huber(control_aux, target_control_aux)
```

This should be auxiliary, not dominant.

## 18.6 Smoothness loss

```text
L_smooth = Huber(diff(pred_traj))
```

Penalize:

```text
jagged lat_m
jagged speed
jagged yaw
```

## 18.7 Dynamics/legal speed penalty

Compute predicted curvature and lateral acceleration:

```text
a_lat = speed^2 * abs(kappa)
```

Penalty:

```text
L_latacc = mean(relu(a_lat - a_lat_soft_limit)^2)
```

## 18.8 Total loss

Initial weights:

```text
L =
  3.0 * L_traj_pos
+ 0.5 * L_traj_yaw
+ 1.0 * L_traj_speed
+ 0.5 * L_ego
+ 0.5 * L_slow
+ 0.2 * L_fallback
+ 0.3 * L_control
+ 0.2 * L_smooth
+ 0.2 * L_latacc
```

Tune empirically.

---

# 19. Data collection

## 19.1 Minimum first dataset

Target:

```text
3–5 effective hours
```

Purpose:

```text
validate telemetry, labels, training, inference, controller.
```

Suggested mix:

```text
1.5 h clear lane keeping
0.75 h curves / winding marked roads
0.75 h following traffic
0.5 h braking / slow lead situations
0.5 h mild recovery offsets
0.25 h rain
0.25 h night
0.25 h UI/phone/popup cases
```

## 19.2 Useful dataset

Target:

```text
8–15 effective hours
```

Suggested mix:

```text
3 h lane keeping across many segments
2 h curves / hills / winding roads
2 h following and braking behind traffic
2 h recovery offsets
1 h rain
1 h night
1 h UI/stale-frame robustness
2 h mixed validation-style driving
```

## 19.3 Recovery data is required

Collect examples where the car is:

```text
slightly left of center and recovers right
slightly right of center and recovers left
too fast entering a curve and slows
approaching a slow lead vehicle
following a vehicle smoothly
recovering from small steering mistakes
```

Do not train only on perfect centered cruising.

---

# 20. Training curriculum

## 20.1 Phase 0: telemetry and target validation

Before training:

```text
verify ego telemetry timestamps
verify video/action/ego alignment
visualize generated trajectories over video
validate speed/yaw/curvature traces
```

Do not train until target trajectories look correct.

## 20.2 Phase 1: overfit tiny subset

Train on 5–10 minutes of clean data.

Goal:

```text
model should overfit trajectory and ego targets.
```

If it cannot, fix data/model before scaling.

## 20.3 Phase 2: first full training

Train on 3–5 hours.

Use:

```text
pretrained RegNet
AMP
gradient clipping
moderate augmentation
early stopping
validation split by route/episode
```

## 20.4 Phase 3: live test and intervention mining

Run live.

Log:

```text
lane drift
oscillation
late braking
speed too high for curve
UI failure
stale-frame failure
fallback events
manual interventions
```

Add those clips to training.

## 20.5 Phase 4: iterative fine-tuning

Repeat:

```text
collect failure -> relabel from ego odometry -> train -> live test
```

This is DAgger-lite, but with trajectory labels from ego odometry.

---

# 21. Dataset splits

Do not random-split frames.

Split by:

```text
episode
road segment
weather/time
traffic scenario
```

Maintain:

```text
train
validation-known-road
validation-heldout-road
closed-loop-test
```

This reduces leakage on GTA’s fixed map.

---

# 22. Augmentation

Use realistic augmentations:

```text
brightness
contrast
gamma
rain/night variation
slight blur
compression artifacts
small crop/scale jitter
small perspective jitter
UI overlay augmentation
phone mask bursts
frame staleness/repeat augmentation
```

Avoid aggressive augmentation that changes the lane geometry without correcting targets.

Action dropout:

```text
randomly zero/drop action history with probability 0.15
```

This prevents overreliance on previous controls.

---

# 23. Live runtime

Implement:

```text
gtapilot/atlas-cruise/live_cruise.py
```

Runtime loop:

```text
1. Receive latest frame from display capture.
2. Receive latest action history.
3. UI-sanitize and resize frame.
4. Maintain recent frame/history buffers.
5. Run Atlas-Cruise at 20 Hz.
6. Sanitize/legalize predicted trajectory.
7. Run lateral controller.
8. Run longitudinal controller.
9. Smooth control outputs.
10. Send GTA actions.
11. Log outputs and debug info.
```

Controller tick:

```text
60 Hz
```

Model tick:

```text
20 Hz
```

Between model ticks:

```text
reuse last predicted/legalized trajectory
advance controller along it
smooth commands
```

---

# 24. Visualization

Implement:

```text
gtapilot/atlas-cruise/visualization.py
gtapilot/atlas-cruise/tools/replay_cruise.py
```

Visualize:

```text
raw RGB
sanitized RGB
UI mask
predicted trajectory projected into image if possible
predicted trajectory in top-down ego frame
target trajectory in offline replay
ego speed target/prediction
slow_or_brake probability
fallback probability
authorized speed
steer/throttle/brake output
frame freshness
latency
```

This is mandatory for debugging.

---

# 25. Evaluation metrics

## 25.1 Offline metrics

```text
trajectory long/lat ADE
trajectory final displacement error
yaw error
speed error
ego speed/yaw-rate error
slow/brake classification accuracy
fallback false positive rate
trajectory smoothness
curvature/speed legality violation rate
```

## 25.2 Live metrics

```text
time/distance between interventions
lane drift events
lane departure events
barrier/curb hits
collisions
phantom braking events
late braking events
curve overspeed events
steering oscillation
fallback events
manual interventions
```

---

# 26. Codex implementation checklist

Codex should implement:

```text
1. gtapilot/native/ego_telemetry C++ ASI plugin or SHVDN prototype.
2. ego.telemetry IPC channel.
3. Python receiver/logger for ego telemetry.
4. Dataset merger aligning video, actions, and ego telemetry by timestamps.
5. Target trajectory builder from ego odometry.
6. AtlasCruiseConfig.
7. AtlasCruise model.
8. RegNet backbone wrapper.
9. Temporal transformer/action encoder.
10. Trajectory/ego/confidence heads.
11. Losses.
12. Training script.
13. Trajectory sanitizer/legalizer.
14. Controller.
15. Live inference process.
16. Replay visualization.
17. Dataset validation tools.
18. Model profiling/stat tools.
```

---

# 27. File structure

Create:

```text
gtapilot/atlas-cruise/
    __init__.py
    config.py
    model.py
    backbone.py
    temporal.py
    action_encoder.py
    heads.py
    state.py
    preprocess.py
    ui.py
    targets.py
    losses.py
    dataset.py
    train.py
    live_cruise.py
    controller.py
    trajectory_legalizer.py
    visualization.py

    tools/
        replay_cruise.py
        profile_cruise.py
        print_model_stats.py
        validate_dataset.py
        build_targets.py

gtapilot/native/ego_telemetry/
    CMakeLists.txt
    EgoTelemetry.cpp
    EgoTelemetry.h
    README.md
```

---

# 28. Milestones

## Milestone 1: telemetry proof

```text
Run GTA.
Plugin logs ego position/speed/yaw.
Python receives ego.telemetry.
Plots speed/yaw/position over time.
```

## Milestone 2: target proof

```text
Build target trajectories.
Replay video with top-down target trajectory overlay.
Verify curves and speed profiles look correct.
```

## Milestone 3: model overfit

```text
Train on 5–10 minutes.
Model overfits trajectory targets.
```

## Milestone 4: first live lane centering

```text
Train on 3–5 hours.
Run live with trajectory controller.
Maintain lane on familiar road/highway.
```

## Milestone 5: adaptive cruise behavior

```text
Add following/braking data.
Model slows behind traffic.
Controller tracks speed profile.
```

## Milestone 6: recovery loop

```text
Collect interventions.
Fine-tune.
Improve drift recovery and braking.
```

---

# 29. Final recommendation

Atlas-Cruise should use the following core design:

```text
RGB video + action history
  -> compact pretrained RegNet temporal model
  -> predicted future ego trajectory
  -> predicted ego speed/yaw
  -> slow/brake/fallback confidence
  -> trajectory legalizer
  -> classical controller
```

Training labels should come from:

```text
minimal GTA ego odometry
```

not from:

```text
SLAM
LiDAR
depth
semantics
lane graphs
actor states
direct controls alone
```

The minimal ego odometry plugin is the highest-value piece of new infrastructure. It turns the problem from noisy direct behavior cloning into clean trajectory supervision while keeping data collection far simpler than the full privileged Atlas stack.
