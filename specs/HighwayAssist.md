# Atlas-HA: full standalone implementation plan and technical specification

## 0. Purpose

Implement a new compact **Highway Assist** version of the Atlas model, named:

```text
Atlas-HA
```

This is a **restricted-ODD monocular camera-only driving assistant** for GTA V. It is not intended to be a general autonomous-driving foundation model. It should be optimized for:

```text
highways, freeways, divided arterial roads, and well-marked roads
```

The first functional target is:

```text
adaptive cruise control + lane centering + conservative system-initiated lane changes
```

The system should be trainable on a **single RTX 5090** and deployable on a **dedicated RTX 2080 Ti 11 GB**, while the game runs separately.

The central design is:

```text
1080p GTA capture
  -> UI-aware scene preprocessing
  -> compact pretrained RegNet/BiFPN visual encoder
  -> 30-second cached full-token temporal context
  -> action-history encoder
  -> explicit highway affordance heads
  -> semantic trajectory candidates
  -> trajectory sanitizer/legalizer/speed profiler
  -> lane-change FSM + classical controller
  -> GTA actions
```

The model should not attempt to solve dense urban autonomy, intersection handling, traffic-light reasoning, full lane-graph extraction, or general hidden-agent reasoning in this version.

TorchVision provides pretrained RegNetY variants such as `regnet_y_800mf` and `regnet_y_1_6gf`; the official docs list pretrained-weight support plus reference parameter/GFLOP metadata, which is why the plan below uses them as the default low-data backbone family. ([PyTorch Documentation][1])

---

# 1. Operating design domain

## 1.1 Supported ODD

`Atlas-HA` should support:

```text
1. Highways / freeways.
2. Divided arterial roads.
3. Well-marked non-highway roads.
4. First-person hood-camera driving.
5. Fixed 16:9 game window.
6. Daytime and nighttime.
7. Dry and rainy weather.
8. Moderate GTA traffic.
9. Lane keeping.
10. Adaptive following / speed control.
11. Conservative lane changes.
12. Visible blocker / lane-closure handling by merging if safe or slowing/stopping if not.
13. Temporary vanilla-GTA UI occlusions, including minimap, top-left popups, phone, and small center reticle.
```

## 1.2 Unsupported or deferred

Do not optimize this first model for:

```text
1. Dense urban driving.
2. Intersections.
3. Stop signs.
4. Traffic lights.
5. Traffic-sign compliance.
6. Pedestrian-heavy areas.
7. Parking lots.
8. Off-road driving.
9. Dirt roads / forest trails.
10. Aggressive merges.
11. Robust blind-spot driving from a single forward camera.
12. Complex 90-degree or greater-than-90-degree urban turns.
13. Full map/lane graph extraction.
14. General actor forecasting.
15. Dense 3D occupancy.
16. Long-horizon speculative hidden-agent reasoning.
```

## 1.3 Highway-optimized, not highway-only

The model should **not** immediately disengage just because the road is not a highway.

A well-marked winding mountain road, for example, is outside the core highway ODD but still close enough to the task. The system should continue in **caution mode**:

```text
slow down,
disable discretionary lane changes,
follow visible lane/path,
increase safety margins.
```

Hard takeover should only trigger when the system has no viable safe behavior.

Examples where hard takeover is allowed:

```text
1. Dirt trail in a forest.
2. No visible usable road.
3. Vehicle is off-road.
4. Vehicle is crashed / overturned / physically invalid.
5. Camera is mostly blocked.
6. All trajectory candidates are invalid for a sustained period.
7. Fallback slow/stop path cannot be safely maintained.
```

---

# 2. Global notation and coordinate convention

## 2.1 Tensor notation

Use the following symbols consistently:

```text
B      = batch size
T      = number of recent image frames explicitly encoded during training
C      = image channels, usually 3 for RGB
H      = model input image height
W      = model input image width
D      = model hidden width / token dimension
F      = visual tokens per encoded frame
S_full = number of cached full-token context frames
S_sum  = number of cached summary context steps
M      = number of sampled action-history steps
A      = action vector dimension
R      = number of action-summary tokens
K      = number of semantic trajectory candidates
N      = number of future trajectory samples
P      = number of lane/road longitudinal sample points
Qf     = number of full-token temporal query tokens
Qs     = number of summary-context query tokens
Cbev   = number of optional BEV auxiliary channels/classes
Hbev   = BEV longitudinal grid cells
Wbev   = BEV lateral grid cells
```

## 2.2 Ego-frame ground-plane convention

Use explicit **longitudinal/lateral** naming everywhere.

```text
long_m = longitudinal position in meters, positive forward from ego
lat_m  = lateral position in meters, positive left of ego, negative right of ego
yaw_rad = yaw relative to current ego heading, positive left / counterclockwise
speed_mps = speed in meters per second
```

Coordinate frame:

```text
long_m > 0  means ahead of ego
lat_m  > 0  means left of ego
lat_m  < 0  means right of ego
yaw_rad > 0 means leftward / counterclockwise rotation
```

Do not expose ambiguous public output fields named `x`/`y` unless they are clearly documented as:

```text
x == long_m
y == lat_m
```

Prefer the names `long_m` and `lat_m`.

---

# 3. Final selected default configuration

Use this as the initial Codex implementation target.

```python
@dataclass
class AtlasHAConfig:
    # Source capture.
    source_h: int = 1080
    source_w: int = 1920
    source_capture_hz: float = 60.0

    # Model input.
    input_h: int = 864
    input_w: int = 1536
    input_channels: int = 3

    # Runtime rates.
    model_hz: float = 20.0
    controller_hz: float = 60.0
    ui_detector_hz: float = 30.0

    # Backbone.
    backbone_name: str = "regnet_y_800mf"
    pretrained_backbone: bool = True
    hidden_dim: int = 192
    fpn_dim: int = 128
    bifpn_repeats: int = 1

    # Visual tokenizer.
    tokens_per_frame: int = 128

    # Full-token temporal context.
    full_token_context_s: float = 30.0
    full_context_queries: int = 24
    full_context_blocks: int = 2
    cache_projected_kv: bool = True

    # Summary context.
    summary_context_s: float = 60.0
    summary_hz: float = 2.0
    summary_context_queries: int = 8
    summary_context_blocks: int = 1

    # Action history.
    action_context_s: float = 30.0
    action_sample_hz: float = 10.0
    action_dim: int = 6
    action_summary_tokens: int = 16
    action_dropout_prob: float = 0.15

    # Trajectory candidates.
    num_traj_candidates: int = 5
    num_traj_points: int = 30
    traj_horizon_s: float = 4.5

    # Lane geometry.
    num_lane_points: int = 24
    lane_long_min_m: float = 5.0
    lane_long_max_m: float = 120.0

    # Optional BEV auxiliary.
    enable_bev_aux: bool = False
    bev_long_min_m: float = 0.0
    bev_long_max_m: float = 120.0
    bev_lat_min_m: float = -24.0
    bev_lat_max_m: float = 24.0
    bev_resolution_m: float = 1.5
    bev_channels: int = 4

    # Runtime safety.
    enable_latency_compensation: bool = True
    nominal_control_latency_s: float = 0.10
    max_frame_staleness_caution_s: float = 0.10
    max_frame_staleness_fallback_s: float = 0.30
    enable_trajectory_sanitizer: bool = True
    enable_trajectory_legalizer: bool = True
    enable_minimum_risk_maneuver: bool = True

    # Vehicle/controller profile.
    default_vehicle_profile: str = "generic_gta_car_conservative"

    # UI handling.
    enable_ui_preprocessor: bool = True
    enable_phone_detector: bool = True
    enable_minimap_crop: bool = True

    # Command conditioning.
    use_command_conditioning: bool = True
    nav_cmd_dim: int = 6
```

## 3.1 Stretch profile

Also implement a stretch config:

```python
@dataclass
class AtlasHAStretchConfig(AtlasHAConfig):
    backbone_name: str = "regnet_y_1_6gf"
    fpn_dim: int = 160
    bifpn_repeats: int = 2
    model_hz: float = 20.0
```

Use `regnet_y_1_6gf` only if profiling confirms acceptable latency on the RTX 2080 Ti. Official TorchVision docs list `regnet_y_1_6gf` with pretrained weights, 11,202,430 parameters, and 1.61 GFLOPs at the 224×224 reference size. ([PyTorch Documentation][2])

---

# 4. Hardware budget and temporal-context feasibility

The RTX 2080 Ti target has 11 GB GDDR6, 616 GB/s memory bandwidth, 4352 CUDA cores, 544 Tensor Cores, and a listed peak FP32 compute figure of 14.2 TFLOPS. ([ServeTheHome][3])

The critical runtime rule is:

```text
Encode only the current RGB frame.
Never re-encode historical RGB frames during live inference.
Cache historical frame tokens.
Use bottleneck cross-attention over cached tokens.
Never use full self-attention over all cached historical tokens.
```

## 4.1 Full-token context default

Default runtime:

```text
model_hz = 20 Hz
full_token_context_s = 30 s
S_full = 20 × 30 = 600 cached frames
F = 128 tokens/frame
D = 192
```

Cached full-token tensor:

```python
full_token_cache : [B, 600, 128, 192]
```

For `B=1`, fp16 memory:

```text
raw token cache = 600 × 128 × 192 × 2 bytes
                = 29,491,200 bytes
                ≈ 28.1 MiB
```

If storing projected K/V for two full-context cross-attention blocks:

```text
K/V cache per block = 2 × raw token cache
                    ≈ 56.3 MiB

K/V cache for 2 blocks = 112.5 MiB

raw tokens + 2-block K/V = about 140.6 MiB
```

This is acceptable on an 11 GB RTX 2080 Ti.

## 4.2 Cross-attention compute

For full-context cross-attention:

```text
Qf = 24 query tokens
L  = 600 × 128 = 76,800 cached tokens
D  = 192
blocks = 2
```

Approximate attention score/value compute per block:

```text
2 × Qf × L × D
= 2 × 24 × 76,800 × 192
≈ 0.708 billion operations
```

For two blocks:

```text
≈ 1.42 billion operations
```

This is small relative to the current-frame visual encoder. The expensive part is the current-frame RegNet/BiFPN encode, not the 30-second token cache.

## 4.3 Why not re-encode raw historical frames?

At 20 Hz and 30 seconds:

```text
600 historical frames per model step
```

At 1536×864 with the default RegNetY-800MF + BiFPN/tokenizer, the current-frame encode is roughly 26 GOP/MAC-equivalent operations. Re-encoding 600 historical frames would be roughly:

```text
600 × 26 ≈ 15,600 GOP per model step
```

At 20 Hz that would imply an unrealistic sustained compute requirement. Cached tokens are mandatory.

---

# 5. Input contract

## 5.1 Runtime input tensors

The model forward signature should support:

```python
def forward(
    self,
    scene_rgb: torch.Tensor,              # [B, 3, H, W], current sanitized RGB
    actions_hist: torch.Tensor,           # [B, M, 6]
    dt_hist: torch.Tensor,                # [B, M, 1]
    ui_mask: torch.Tensor | None = None,  # [B, 1, H, W]
    nav_cmd: torch.Tensor | None = None,  # [B, 6]
    state: AtlasHAState | None = None,
) -> tuple[dict[str, torch.Tensor], AtlasHAState]:
    ...
```

At live inference, the model receives the **current frame only**. Historical visual context comes from `state`.

## 5.2 Training input tensors

Training may use several frames for end-to-end short-context updates, but the architecture should still emulate the cached-token runtime path.

Suggested training batch:

```python
{
    "scene_rgb_current": [B, 3, H, W],
    "scene_rgb_short":   [B, T_short, 3, H, W],       # optional for end-to-end short-context training
    "ui_mask_current":   [B, 1, H, W],
    "ui_mask_short":     [B, T_short, 1, H, W],       # optional

    "full_token_cache":  [B, S_full, F, D],           # often precomputed/no-grad in training
    "summary_cache":     [B, S_sum, D],
    "actions_hist":      [B, M, 6],
    "dt_hist":           [B, M, 1],
    "nav_cmd":           [B, 6],
}
```

---

# 6. Action vector convention

Action vector dimension:

```text
A = 6
```

Layout:

```text
[steer, throttle, brake, handbrake, reverse, pilot_active]
```

Definitions:

```text
steer        : normalized steering command, usually [-1, +1]
throttle     : normalized throttle, usually [0, 1]
brake        : normalized brake, usually [0, 1]
handbrake    : 0/1 or normalized
reverse      : 0/1
pilot_active : 0 = human/manual control, 1 = policy/autopilot active
```

Runtime action-history sampling:

```text
Raw action logging: 60 Hz.
Model action sample rate: 10 Hz.
Action context: 30 seconds.
M = 30 × 10 = 300 sampled action steps.
```

Input to action encoder:

```python
actions_hist : [B, 300, 6]
dt_hist      : [B, 300, 1]
```

Internally concatenate:

```python
action_dt_hist = cat([actions_hist, dt_hist], dim=-1)  # [B, 300, 7]
```

---

# 7. UI and camera assumptions

## 7.1 Camera assumptions

The supported camera mode is:

```text
first-person hood camera
fixed forward direction
fixed 16:9 window
fixed FOV under default game settings
no manual camera panning while pilot_active=1
```

Do not implement third-person/chase-camera support in `Atlas-HA`.

Required runtime checks:

```text
1. Capture resolution matches expected 1920×1080 or configured source size.
2. Crop/resize transform is fixed.
3. Hood-camera startup check passes.
4. Basic horizon/forward-view sanity check passes if implemented.
```

Vehicle-dependent hood-camera height/position may vary slightly. First version should train/test primarily on one or a small number of vehicle models.

## 7.2 Vanilla GTA UI artifacts

Vanilla GTA may include:

```text
1. Bottom-left minimap.
2. Top-left semi-transparent notification overlays.
3. Bottom-right phone popup.
4. Small center reticle for weaponized vehicles.
```

These should not be treated as real road pixels.

Implement:

```text
gtapilot/atlas/highway/ui.py
```

with:

```python
@dataclass
class UIState:
    minimap_present: bool
    phone_present: bool
    popup_present: bool
    reticle_present: bool
    ui_mask: torch.Tensor          # [1, H, W]
    phone_mask: torch.Tensor | None
    popup_mask: torch.Tensor | None
    reticle_mask: torch.Tensor | None
```

UI preprocessor output:

```python
scene_rgb : [3, H, W]
ui_mask   : [1, H, W]
minimap_crop : optional
ui_state  : UIState
```

### Minimap

For v1:

```text
Mask or neutralize minimap in the main scene branch.
Do not use minimap for control.
```

For v2:

```text
minimap_crop -> minimap_encoder -> nav_cmd / route tokens
```

Do not feed the minimap directly into the main road-scene backbone.

### Phone

Implement a phone detector. When the phone appears:

```text
1. Immediately send the GTA right-click / close-phone command.
2. Mark the phone region in ui_mask.
3. Replace phone pixels in scene_rgb.
4. Enter caution mode while phone is visible.
5. Disable or delay right-lane-change candidates while phone occludes right-side evidence.
```

Preferred phone-region fill:

```text
dynamic stale-fill + mask
```

Meaning:

```text
replace phone pixels with last known clean pixels from that region,
set ui_mask = 1 for the phone region,
set phone_present = True.
```

Fallback fill:

```text
neutral blank / blurred fill + mask.
```

Avoid:

```text
raw phone pixels with no mask.
```

Raw phone pixels are misleading non-scene evidence. Known missing pixels are safer than fake scene structure.

### Top-left popups

Detect in the fixed top-left ROI. While visible:

```text
mask/neutralize popup region,
set popup_present = True,
enter caution mode only if it materially affects road perception.
```

### Reticle

Erase or mask a small center patch. This should not require a special model head.

---

# 8. Visual backbone and tokenizer

## 8.1 Backbone choice

Default:

```text
RegNetY-800MF pretrained through torchvision
```

Stretch:

```text
RegNetY-1.6GF pretrained through torchvision
```

The TorchVision docs list `regnet_y_800mf` pretrained weights, 6,432,512 parameters, and 0.83 GFLOPs at reference size; they also list `regnet_y_1_6gf` pretrained weights, 11,202,430 parameters, and 1.61 GFLOPs at reference size. ([PyTorch Documentation][1])

Do not start with RegNetY-3.2GF for v1. It is likely unnecessary given the small-data target and will reduce latency margin.

## 8.2 Input size

Default model input:

```text
H = 864
W = 1536
```

Source capture remains:

```text
1080 × 1920 @ 60 Hz
```

Resize/crop:

```text
raw 1920×1080 -> sanitized RGB -> resize to 1536×864
```

Fallback if profiling fails:

```text
1280×720, optionally padded to stride-32 if necessary
```

Full 1080p model input should remain a config option, not the default.

## 8.3 Backbone feature extraction

Use the RegNet trunk without the classifier. Extract the four stage outputs:

```python
c2 : [B, C2, H/4,  W/4]
c3 : [B, C3, H/8,  W/8]
c4 : [B, C4, H/16, W/16]
c5 : [B, C5, H/32, W/32]
```

At `H=864, W=1536`:

```text
c2 spatial: 216 × 384
c3 spatial: 108 × 192
c4 spatial:  54 ×  96
c5 spatial:  27 ×  48
```

Expected RegNetY stage channels:

```text
RegNetY-800MF:
  C2 = 64
  C3 = 144
  C4 = 320
  C5 = 784

RegNetY-1.6GF:
  C2 = 48
  C3 = 120
  C4 = 336
  C5 = 888
```

Codex should confirm these by introspecting the actual torchvision model.

## 8.4 BiFPN neck

Project all four feature levels to `Cfpn`.

Default:

```text
Cfpn = 128
bifpn_repeats = 1
```

Stretch:

```text
Cfpn = 160
bifpn_repeats = 2
```

Use lateral projections:

```text
1×1 Conv2d, bias=False
GroupNorm or BatchNorm
activation
```

Use lightweight separable convolution blocks inside BiFPN:

```text
depthwise 3×3 conv
pointwise 1×1 conv
normalization
activation
```

BiFPN outputs:

```python
p2 : [B, Cfpn, H/4,  W/4]
p3 : [B, Cfpn, H/8,  W/8]
p4 : [B, Cfpn, H/16, W/16]
p5 : [B, Cfpn, H/32, W/32]
```

## 8.5 Fixed-grid visual tokenizer

Do **not** flatten full-resolution feature maps into a huge attention source.

Instead, adaptively pool each pyramid level to a fixed grid:

```text
p2 pooled grid: 24 × 42 = 1008 tokens
p3 pooled grid: 12 × 21 = 252 tokens
p4 pooled grid:  6 × 11 = 66 tokens
p5 pooled grid:  3 ×  6 = 18 tokens

total source tokens = 1344
```

Then project source tokens to `D=192`.

```python
source_tokens : [B, 1344, D]
```

Use learned frame queries:

```python
frame_queries : [F, D]
F = 128
```

Use one cross-attention block:

```python
frame_tokens : [B, 128, 192]
```

Also produce:

```python
frame_summary : [B, 192]
```

Recommended frame summary:

```text
mean-pool frame_tokens followed by LayerNorm/MLP
```

---

# 9. Temporal system: 30-second full-token cache

## 9.1 Runtime state

Implement:

```python
@dataclass
class AtlasHAState:
    full_token_cache: torch.Tensor        # [B, S_full, F, D]
    full_kv_cache: list[KVCache]          # projected K/V per temporal block, optional
    summary_cache: torch.Tensor           # [B, S_sum, D]
    action_ring: ActionRingBuffer
    dt_ring: DtRingBuffer

    previous_selected_traj: torch.Tensor | None
    previous_stable_lane: torch.Tensor | None
    previous_stable_path: torch.Tensor | None
    lane_memory_age_s: float

    ego_state: EgoState
    timing_state: TimingState
    frame_freshness_state: FrameFreshnessState
    supervisor_state: SupervisorState
    lane_change_fsm_state: LaneChangeFSMState
    ui_state: UIState
```

Default cache sizes:

```text
S_full = full_token_context_s × model_hz
       = 30 × 20
       = 600

S_sum = summary_context_s × summary_hz
      = 60 × 2
      = 120
```

Shapes:

```python
full_token_cache : [B, 600, 128, 192]
summary_cache    : [B, 120, 192]
```

## 9.2 Full-context reader

Use bottleneck cross-attention.

```python
full_context_queries : [Qf, D]
Qf = 24
D = 192
```

Input:

```python
cached_tokens_flat : [B, S_full * F, D]  # [B, 76800, 192]
```

Output:

```python
full_ctx_tokens : [B, 24, 192]
```

Use:

```text
2 cross-attention blocks
8 attention heads
head_dim = 24
MLP ratio = 2
pre-norm
```

Never perform full self-attention over `[B, 76800, D]`.

## 9.3 Summary-context reader

Input:

```python
summary_cache : [B, 120, 192]
```

Learned queries:

```python
summary_context_queries : [8, 192]
```

Output:

```python
summary_ctx_tokens : [B, 8, 192]
```

Use one cross-attention block.

## 9.4 Action-history encoder

Sample actions at 10 Hz over 30 seconds:

```python
actions_hist : [B, 300, 6]
dt_hist      : [B, 300, 1]
action_dt    : [B, 300, 7]
```

Use:

```text
Linear(7 -> D)
sinusoidal or learned time/age encoding
learned action queries [16, D]
one cross-attention block
```

Output:

```python
action_tokens  : [B, 16, 192]
action_summary : [B, 192]
```

Use action dropout/noise during training:

```text
action_dropout_prob = 0.15
```

Action history should strongly support ego-state estimation but should not dominate scene/policy decisions.

## 9.5 Head-fusion tokens

Concatenate:

```python
current_frame_tokens : [B, 128, 192]
full_ctx_tokens      : [B, 24, 192]
summary_ctx_tokens   : [B, 8, 192]
action_tokens        : [B, 16, 192]
optional nav token   : [B, 1, 192]
```

Then use a small head-fusion reader:

```python
head_queries : [8, 192]
```

Output:

```python
head_tokens : [B, 8, 192]
head_context : [B, 192]
```

`head_context` can be the mean of `head_tokens` or a dedicated first query token.

---

# 10. Expected parameter counts under the target implementation

These counts are for the defined implementation target above, excluding optional BEV and route/minimap adapters.

## 10.1 Common block parameter count

For a cross-attention block with:

```text
D = 192
MLP ratio = 2
```

Using:

```text
q_proj, k_proj, v_proj, out_proj
2-layer MLP: D -> 2D -> D
2 LayerNorms
```

Parameter count:

```text
297,024 params per block
```

## 10.2 Default RegNetY-800MF profile

```text
RegNetY-800MF trunk without classifier: 5,647,512 params
lateral projections:                    168,960
BiFPN:                                  106,767
frame tokenizer:                        346,368
full-context reader:                    598,656
summary-context reader:                 298,560
action encoder:                         301,632
head fusion:                            298,560
trajectory head:                        381,149
lane/road head:                         100,037
lead/adjacent head:                      53,006
ego head:                                25,220
confidence/takeover head:                25,607

Total without optional BEV/minimap:   8,352,034 params
```

## 10.3 Stretch RegNetY-1.6GF profile

```text
RegNetY-1.6GF trunk without classifier: 10,313,430 params
lateral projections:                       224,000
BiFPN:                                     328,350
frame tokenizer:                           352,512
full-context reader:                       598,656
summary-context reader:                    298,560
action encoder:                            301,632
head fusion:                               298,560
trajectory head:                           381,149
lane/road head:                            100,037
lead/adjacent head:                         53,006
ego head:                                   25,220
confidence/takeover head:                   25,607

Total without optional BEV/minimap:    13,300,719 params
```

Codex should add a utility:

```text
gtapilot/atlas/highway/tools/print_model_stats.py
```

that prints exact parameter counts and measured latency after implementation.

---

# 11. Model outputs

The main model output dict must include:

```python
{
    # Path / maneuver.
    "traj_candidates": torch.Tensor,          # [B, K, N, 4]
    "candidate_logits": torch.Tensor,         # [B, K]

    # Lane/path.
    "lane_lat_pred": torch.Tensor,            # [B, 3, P]
    "lane_valid_logit": torch.Tensor,         # [B, 3, P]
    "lane_conf": torch.Tensor,                # [B, 3]

    # Road edges, optional but cheap.
    "road_edge_lat_pred": torch.Tensor,       # [B, 2, P]
    "road_edge_conf": torch.Tensor,           # [B, 2]

    # Lead / adjacent lane.
    "lead_present_logit": torch.Tensor,       # [B, 1]
    "lead_state": torch.Tensor,               # [B, 5]
    "adjacent_left": torch.Tensor,            # [B, 4]
    "adjacent_right": torch.Tensor,           # [B, 4]

    # Ego.
    "ego_kinematics": torch.Tensor,           # [B, 4]

    # Confidence / fallback.
    "road_followable_conf": torch.Tensor,     # [B, 1]
    "lane_tracking_conf": torch.Tensor,       # [B, 1]
    "takeover_required_logit": torch.Tensor,  # [B, 1]
    "scene_type_logits": torch.Tensor,        # [B, 4], optional but recommended

    # Optional auxiliary.
    "bev_highway": torch.Tensor | None,       # [B, Cbev, Hbev, Wbev]
}
```

---

# 12. Trajectory head

## 12.1 Shape

```python
traj_candidates : [B, K, N, 4]
candidate_logits: [B, K]
```

Defaults:

```text
K = 5
N = 30
traj_horizon_s = 4.5
traj_dt_s = 4.5 / 30 = 0.15 seconds
```

Each trajectory point layout:

```text
[long_m, lat_m, yaw_rad, speed_mps]
```

## 12.2 Fixed semantic candidate slots

Use fixed semantic candidate meanings:

```text
k = 0: KEEP_LANE_CRUISE
k = 1: KEEP_LANE_SLOW_OR_FOLLOW
k = 2: CHANGE_LEFT
k = 3: CHANGE_RIGHT
k = 4: FALLBACK_SLOW_STOP
```

Meaning:

```python
traj_candidates[:, 0]  # normal keep-lane path
traj_candidates[:, 1]  # slower keep-lane/following path
traj_candidates[:, 2]  # left lane-change path
traj_candidates[:, 3]  # right lane-change path
traj_candidates[:, 4]  # conservative slow/stop path
```

`candidate_logits[:, k]` is the model’s preference/confidence for candidate `k`, but the runtime supervisor may veto or re-rank candidates.

## 12.3 Runtime interpretation

The model’s speed is only a proposal:

```text
proposed_speed_mps = raw model speed
authorized_speed_mps = speed after legalizer/controller caps
executed_speed_mps = actual speed after GTA physics/control
```

The controller must use `authorized_speed_mps`, not raw model speed.

---

# 13. Lane and road head

Use fixed longitudinal samples.

```python
lane_long_samples_m : [P]
P = 24
```

Default:

```python
lane_long_samples_m = linspace(5.0, 120.0, 24)
```

Outputs:

```python
lane_lat_pred    : [B, 3, P]
lane_valid_logit : [B, 3, P]
lane_conf        : [B, 3]

road_edge_lat_pred : [B, 2, P]
road_edge_conf     : [B, 2]
```

Lane index convention:

```text
lane index 0 = left adjacent lane center
lane index 1 = current lane center
lane index 2 = right adjacent lane center
```

Road-edge index convention:

```text
road edge index 0 = left road edge / shoulder / barrier boundary
road edge index 1 = right road edge / shoulder / barrier boundary
```

Interpretation:

```text
At longitudinal sample lane_long_samples_m[p],
predict lateral lane-center location lane_lat_pred[b, lane_idx, p].
```

`lane_conf` meaning:

```text
lane_conf[:, 0] = confidence left adjacent lane exists / is usable
lane_conf[:, 1] = confidence current lane/path is visible / usable
lane_conf[:, 2] = confidence right adjacent lane exists / is usable
```

Do not require a full lane graph in v1.

---

# 14. Lead and adjacent-lane head

Outputs:

```python
lead_present_logit : [B, 1]
lead_state         : [B, 5]

adjacent_left      : [B, 4]
adjacent_right     : [B, 4]
```

`lead_state` layout:

```text
[lead_long_m, lead_lat_m, lead_distance_m, lead_rel_speed_mps, lead_ttc_s]
```

Definitions:

```text
lead_long_m = lead longitudinal position in ego frame
lead_lat_m = lead lateral position in ego frame
lead_distance_m = distance to lead along current lane / forward corridor
lead_rel_speed_mps = lead_speed_mps - ego_speed_mps
lead_ttc_s = time-to-collision estimate, clipped to a max value
```

`adjacent_left` / `adjacent_right` layout:

```text
[available_prob, front_gap_m, rear_risk_proxy, confidence]
```

Important:

```text
rear_risk_proxy is not a true blind-spot detector.
```

With only a forward camera, adjacent/rear safety must be conservative. Use privileged side/rear actor state during training only for **policy/gap labels**, not as direct visible-state regression unless the actor was recently visible.

---

# 15. Ego head

Output:

```python
ego_kinematics : [B, 4]
```

Layout:

```text
[speed_mps, a_long_mps2, yaw_rate_radps, curvature_inv_m]
```

Definitions:

```text
speed_mps = ego speed estimate
a_long_mps2 = longitudinal acceleration
yaw_rate_radps = yaw rate
curvature_inv_m = approximate current curvature, 1/m
```

At inference, these are estimated from camera + action history. Privileged ego state is only a training label.

Runtime should filter this output using `HighwayEgoStateEstimator`.

---

# 16. Confidence and takeover head

Outputs:

```python
road_followable_conf      : [B, 1]
lane_tracking_conf        : [B, 1]
takeover_required_logit   : [B, 1]
scene_type_logits         : [B, 4]
```

Scene type classes:

```text
0: highway / multi-lane marked road
1: well-marked non-highway road
2: ambiguous but road-like
3: no usable road / invalid
```

Definitions:

```text
road_followable_conf:
  confidence that the visible road structure is interpretable enough to continue.

lane_tracking_conf:
  confidence that a current lane/path can be followed.

takeover_required_logit:
  extreme signal meaning no viable safe behavior is available.

scene_type_logits:
  descriptive scene type, not a direct disengagement command.
```

Hard takeover should be rare and hysteresis-gated.

---

# 17. Optional BEV auxiliary head

BEV is optional in v1. If enabled:

```python
bev_highway : [B, Cbev, Hbev, Wbev]
```

Default BEV range:

```text
longitudinal: 0 m to 120 m
lateral: -24 m to +24 m
resolution: 1.5 m/cell
```

Therefore:

```text
Hbev = 80
Wbev = 32
```

Recommended `Cbev = 4` channels/classes:

```text
0: drivable road
1: lane marking / lane boundary
2: vehicle occupied
3: road edge / barrier / unknown
```

BEV is for auxiliary supervision and visualization. It is not the primary planner state.

---

# 18. Runtime supervisor modes

Implement a runtime supervisor with four modes:

```text
NORMAL
CAUTION
MINIMUM_RISK
TAKEOVER_REQUESTED
```

## 18.1 NORMAL

Conditions:

```text
road_followable_conf high
lane_tracking_conf high
selected trajectory stable
frame fresh
UI occlusion not relevant to maneuver
lead/adjacent outputs sane
```

Behavior:

```text
normal target speed
lane centering
ACC/following
lane changes allowed if FSM accepts
```

## 18.2 CAUTION

Conditions:

```text
well-marked but non-highway road
ambiguous but road-like scene
temporary phone/popup occlusion
frame staleness > caution threshold
lane confidence degraded
sharp curvature
rain / lower grip
traffic ambiguity
```

Behavior:

```text
reduce speed cap
disable discretionary lane changes
increase following margin
prefer KEEP_LANE_SLOW_OR_FOLLOW
continue operating
```

## 18.3 MINIMUM_RISK

Conditions:

```text
low confidence but still enough path structure to slow safely
stale frames persist
candidate legality weak
fallback path needed
```

Behavior:

```text
disable lane changes
follow previous stable path or current lane if visible
smoothly decelerate
hold stable steering if path is lost
request takeover if condition persists
```

## 18.4 TAKEOVER_REQUESTED

Conditions:

```text
takeover_required_prob high
road_followable_conf very low
lane_tracking_conf very low
all candidates invalid
fallback slow/stop invalid or unsafe
condition persists through hysteresis
```

Recommended hysteresis:

```text
takeover_required_prob > 0.95
AND road_followable_conf < 0.15
AND lane_tracking_conf < 0.15
for 0.75 to 1.5 seconds
```

Do not immediately drop controls. Transition through minimum-risk behavior if possible.

---

# 19. Timing and latency compensation

Implement:

```text
gtapilot/atlas/highway/timing.py
gtapilot/atlas/highway/ego_predictor.py
```

Track timestamps:

```python
@dataclass
class TimingState:
    capture_time_ns: int
    model_start_time_ns: int
    model_output_time_ns: int
    control_send_time_ns: int
    estimated_apply_time_ns: int
    estimated_capture_to_control_s: float
```

The trajectory should be interpreted relative to the **estimated actuation time**, not the raw image-capture time.

Runtime flow:

```text
frame captured at t_capture
  -> model inference
  -> estimate t_apply
  -> forward-predict ego state from t_capture to t_apply
  -> transform/align trajectory to t_apply ego frame
  -> legalizer/controller act on latency-compensated state
```

Training target generation should support:

```python
nominal_control_latency_s = 0.10
```

Generate trajectory targets from:

```text
t + nominal_control_latency_s
```

rather than exactly from frame time.

---

# 20. Ego-state estimator

Implement:

```text
gtapilot/atlas/highway/ego_state.py
```

Runtime state:

```python
@dataclass
class EgoState:
    speed_mps: float
    a_long_mps2: float
    yaw_rate_radps: float
    curvature_inv_m: float
    confidence: float
    timestamp_ns: int
```

Estimator:

```python
class HighwayEgoStateEstimator:
    def update(
        self,
        model_ego_kinematics: torch.Tensor,
        previous_controls: dict,
        dt_s: float,
        model_confidence: float,
    ) -> EgoState:
        ...
```

Use smoothing/filtering. Do not use privileged ego state in deploy mode.

Debug mode may optionally compare against privileged ego state if available, but this must be clearly separated from deployable inference.

---

# 21. Vehicle dynamics profile and actuator calibration

Implement:

```text
gtapilot/atlas/highway/vehicle_profile.py
gtapilot/atlas/highway/actuator_calibration.py
```

## 21.1 Vehicle dynamics profile

```python
@dataclass
class VehicleDynamicsProfile:
    vehicle_model_hash: int | None = None

    wheelbase_eff_m: float = 2.6
    steer_max_rad: float = 0.55
    steer_rate_max_radps: float = 2.0

    accel_max_mps2: float = 2.5
    decel_comfort_mps2: float = 3.0
    decel_hard_mps2: float = 6.0
    jerk_max_mps3: float = 8.0

    a_lat_comfort_mps2: float = 3.0
    a_lat_hard_mps2: float = 5.0

    rain_lat_multiplier: float = 0.75
    caution_lat_multiplier: float = 0.75
    safety_margin: float = 0.80
```

These numbers are placeholders. Calibrate in GTA.

First version should use one conservative global profile:

```text
generic_gta_car_conservative
```

Later, specialize by vehicle model/class.

## 21.2 Actuator calibration

```python
@dataclass
class ActuatorCalibration:
    steer_deadzone: float
    steer_gain: float
    steer_nonlinearity: float

    throttle_deadzone: float
    throttle_gain: float

    brake_deadzone: float
    brake_gain: float

    input_lag_s: float
```

Create scripted calibration runs:

```text
1. Constant steering at multiple speeds.
2. Step steering response.
3. Full-throttle acceleration.
4. Brake deceleration.
5. Dry/rain variants if possible.
6. Repeat for the selected default vehicle.
```

---

# 22. Trajectory sanitizer, legalizer, and speed profiler

Implement:

```text
gtapilot/atlas/highway/trajectory_sanitizer.py
gtapilot/atlas/highway/trajectory_legalizer.py
gtapilot/atlas/highway/speed_profiler.py
gtapilot/atlas/highway/candidate_selector.py
```

## 22.1 Trajectory sanitizer

Before legality checks:

```text
1. Clamp speed_mps >= 0.
2. Enforce monotonic longitudinal progress for forward candidates.
3. Recompute yaw from geometry if yaw is invalid.
4. Smooth geometry with spline/Savitzky-Golay/cubic fit.
5. Reject pathological candidates.
```

Reject reason codes:

```text
reject_nonmonotonic_long
reject_negative_speed
reject_bad_yaw
reject_curvature_excess
reject_nan
reject_invalid_geometry
```

## 22.2 Dynamic feasibility

For each candidate, estimate curvature:

```text
kappa_i = curvature at trajectory point i, units 1/m
```

Lateral acceleration:

```text
a_lat_i = speed_i^2 * abs(kappa_i)
```

Curvature-derived speed cap:

```text
v_cap_curve_i = sqrt(a_lat_limit / (abs(kappa_i) + eps))
```

Steering estimate:

```text
steer_i = atan(wheelbase_eff_m * kappa_i)
```

Reject or penalize if:

```text
abs(steer_i) > steer_max_rad
abs(dsteer/dt) > steer_rate_max_radps
```

## 22.3 Speed profiler

Combine speed caps:

```text
authorized_speed = min(
    model_proposed_speed,
    user_max_speed,
    curvature_speed_cap,
    lead_speed_cap,
    odd/caution_speed_cap,
    ui_speed_cap,
    rain/weather_speed_cap
)
```

Then run acceleration/deceleration feasibility passes.

Forward pass:

```text
v[i] <= sqrt(v[i-1]^2 + 2 * accel_max * ds)
```

Backward pass:

```text
v[i] <= sqrt(v[i+1]^2 + 2 * decel_limit * ds)
```

Smooth jerk.

## 22.4 Legalized trajectory output

```python
@dataclass
class LegalizedTrajectorySet:
    traj: torch.Tensor                 # [K_runtime, N, 4]
    legal_mask: torch.Tensor           # [K_runtime]
    cost: torch.Tensor                 # [K_runtime]
    selected_idx: int
    reject_reasons: list[str]
    speed_cap_curve: torch.Tensor      # [K_runtime, N]
    speed_cap_lead: torch.Tensor       # [K_runtime, N]
    speed_cap_ui: torch.Tensor         # [K_runtime, N]
    speed_cap_odd: torch.Tensor        # [K_runtime, N]
    speed_cap_final: torch.Tensor      # [K_runtime, N]
```

## 22.5 Fallback guarantee

If no model candidate is legal, synthesize:

```text
CONTROL_FALLBACK_BRAKE_ALONG_CURRENT_PATH
```

Fallback path sources in priority order:

```text
1. Current high-confidence lane estimate.
2. Previous stable lane/path if recent.
3. Short straight braking path if lane is unavailable.
```

If the vehicle is already physically too fast for the curve, no controller can guarantee perfect tracking. In that case:

```text
brake,
reduce throttle,
track least-risk available path,
avoid abrupt steering,
enter minimum-risk / takeover-requested if necessary.
```

---

# 23. Candidate selector and lane-change FSM

## 23.1 Candidate cost

Do not select purely by `argmax(candidate_logits)`.

Compute:

```python
total_cost[k] =
    w_model   * (-candidate_logits[k])
  + w_legal   * legalizer_cost[k]
  + w_lane    * lane_deviation_cost[k]
  + w_lead    * lead_risk_cost[k]
  + w_ui      * ui_occlusion_cost[k]
  + w_switch  * candidate_switch_cost[k]
  + w_comfort * comfort_cost[k]
  + w_cmd     * command_mismatch_cost[k]
```

Pick the lowest-cost legal candidate.

## 23.2 Candidate hysteresis

Avoid candidate flicker:

```text
Do not switch selected candidate unless the new candidate is clearly better
or current candidate becomes illegal/unsafe.
```

## 23.3 Lane-change FSM

Implement:

```text
KEEP_LANE
PREPARE_LEFT
CHANGE_LEFT
SETTLE_LEFT
PREPARE_RIGHT
CHANGE_RIGHT
SETTLE_RIGHT
ABORT
```

Lane-change candidate acceptance requires:

```text
1. Candidate is CHANGE_LEFT or CHANGE_RIGHT.
2. nav_cmd or operational need supports lane change.
3. Target lane_conf is high.
4. Adjacent available_prob is high.
5. front_gap_m sufficient.
6. rear_risk_proxy low.
7. Lane estimate stable for several frames.
8. No relevant phone/popup occlusion.
9. Lead TTC acceptable.
10. Legalizer approves path/speed.
```

If rejected:

```text
choose KEEP_LANE_CRUISE,
KEEP_LANE_SLOW_OR_FOLLOW,
or FALLBACK_SLOW_STOP.
```

---

# 24. Classical controller

Implement:

```text
gtapilot/atlas/highway/controller.py
```

## 24.1 Lateral control

Use:

```text
pure pursuit or Stanley controller first
```

Input:

```python
selected_legalized_traj : [N, 4]
ego_state
vehicle_profile
```

Output:

```text
steer command
```

## 24.2 Longitudinal control

Use ACC-like control.

Inputs:

```text
authorized speed profile
ego speed
lead_present
lead_state
desired/user max speed
supervisor mode
```

Output:

```text
throttle
brake
```

Rules:

```text
1. Track authorized_speed_mps, not model speed.
2. Increase following distance in caution mode.
3. Brake if TTC unsafe.
4. Smooth throttle/brake to reduce jerk.
```

---

# 25. Command conditioning and route intent

Implement optional `nav_cmd`.

```python
nav_cmd : [B, 6]
```

Suggested command classes:

```text
0: KEEP / FOLLOW
1: PREFER_LEFT
2: PREFER_RIGHT
3: PREPARE_EXIT_LEFT
4: PREPARE_EXIT_RIGHT
5: SLOW / CAUTION
```

For v1, `nav_cmd` can be manually set or generated by a simple heuristic.

Rules:

```text
1. The model may score lane-change candidates.
2. The FSM should require nav_cmd or clear operational need for discretionary lane changes.
3. A slow/stopped blocker ahead can count as operational need.
```

Do not implement minimap route parsing in v1 unless explicitly needed later.

---

# 26. Frame freshness and capture health

Implement:

```text
gtapilot/atlas/highway/frame_freshness.py
```

Runtime metadata:

```python
@dataclass
class FrameFreshnessState:
    source_frame_id: int
    is_fresh: bool
    frame_age_s: float
    repeated_frame_count: int
    capture_to_model_latency_s: float
```

Rules:

```text
if frame_age_s > 0.10:
    enter CAUTION

if frame_age_s > 0.30:
    enter MINIMUM_RISK or fallback slow/stop

if stale persists and road/path cannot be confirmed:
    request TAKEOVER
```

Do not let a repeated frame be treated as a fresh observation.

---

# 27. Minimum-risk maneuver

Implement:

```text
gtapilot/atlas/highway/minimum_risk.py
```

Policy:

```text
1. Disable lane changes.
2. Reuse current lane or previous stable path if available.
3. Smoothly decelerate.
4. Increase following margin.
5. If lane/path disappears, brake while holding stable steering.
6. Request takeover if confidence remains low.
```

Do not immediately stop sending controls when takeover is requested.

---

# 28. Required package structure

Create:

```text
gtapilot/atlas/highway/
    __init__.py
    config.py
    model.py
    backbone.py
    bifpn.py
    tokenizer.py
    temporal.py
    action_encoder.py
    heads.py
    state.py

    ui.py
    timing.py
    ego_state.py
    ego_predictor.py
    frame_freshness.py

    vehicle_profile.py
    actuator_calibration.py
    trajectory_sanitizer.py
    trajectory_legalizer.py
    speed_profiler.py
    candidate_selector.py
    fsm.py
    controller.py
    minimum_risk.py

    dataset.py
    targets.py
    losses.py
    train_affordance.py
    train_policy.py
    train_dagger.py

    live_highway_assist.py
    visualization.py

    tools/
        print_model_stats.py
        profile_inference.py
        replay_highway.py
        calibrate_vehicle.py
        build_token_cache.py
        validate_dataset.py
```

Do not destroy or heavily mutate the existing full Atlas foundation model. Implement `Atlas-HA` as a separate variant/package.

---

# 29. Dataset requirements

## 29.1 Core data fields

Every collected run should store:

```text
RGB frames
source frame timestamps
source frame IDs
fresh/repeated-frame metadata
actions
action timestamps
dt
pilot_active flag
weather/time-of-day
vehicle model/class
UI metadata if available
```

## 29.2 Privileged labels

Required labels:

```python
target_traj             : [B, N, 4]
target_candidate        : [B]
target_ego              : [B, 4]
target_lead_present     : [B, 1]
target_lead_state       : [B, 5]
target_lane_lat         : [B, 3, P]
target_lane_valid       : [B, 3, P]
target_lane_conf        : [B, 3]
```

Recommended labels:

```python
target_road_edge_lat        : [B, 2, P]
target_road_edge_valid      : [B, 2, P]
target_adjacent_left_visible: [B, 4]
target_adjacent_right_visible: [B, 4]
target_lane_change_teacher_ok: [B, 2]
target_soft_scene_type      : [B]
target_takeover_required    : [B, 1]
target_legal_speed          : [B, N]
target_candidate_legal      : [B, K]
target_bev                  : [B, Cbev, Hbev, Wbev]  # optional
ui_mask                     : [B, 1, H, W]
```

## 29.3 Label derivation

### Future trajectory

From privileged ego pose:

```text
target_traj[t, n] = future ego pose at t + latency + n*traj_dt
                    expressed in ego frame at t + latency
```

Layout:

```text
[long_m, lat_m, yaw_rad, speed_mps]
```

### Candidate label

Heuristic:

```text
small lateral shift + steady speed -> KEEP_LANE_CRUISE
small lateral shift + braking/following -> KEEP_LANE_SLOW_OR_FOLLOW
large positive lateral shift -> CHANGE_LEFT
large negative lateral shift -> CHANGE_RIGHT
strong braking/stopping -> FALLBACK_SLOW_STOP
```

### Ego label

From privileged ego state:

```text
[speed_mps, a_long_mps2, yaw_rate_radps, curvature_inv_m]
```

### Lead label

From actor states:

```text
find closest relevant vehicle in current lane / forward corridor
derive lead_long_m, lead_lat_m, distance, rel_speed, TTC
```

### Adjacent-lane labels

Separate visible/inferable labels from omniscient privileged labels.

Use:

```text
target_adjacent_left_visible
target_adjacent_right_visible
```

for direct model regression.

Use:

```text
target_lane_change_teacher_ok
```

for policy/gap supervision from full privileged state.

Do not force the student to regress perfect hidden rear/side state that is not visible or recently observed.

### Lane labels

Do not require full lane graph extraction.

Use lane proxies:

```text
1. Clean expert trajectory as current-lane center in lane-keeping clips.
2. Repeated-pass averaging over fixed highway segments.
3. Lane-width offsets for adjacent lanes.
4. Optional road/path-node approximation.
5. Optional small manually curated highway seed map.
```

Important rule:

```text
Do not use future ego trajectory as lane-center label during lane changes,
merges, obstacle avoidance, or off-center recovery.
```

During lane changes, separate:

```text
target_traj = lane-change path
target_current_lane_lat = original/current lane
target_target_lane_lat = target lane
target_candidate = CHANGE_LEFT or CHANGE_RIGHT
```

Mask invalid lane samples with `target_lane_valid`.

### Road edges

Use approximate labels only:

```text
semantics/depth if available,
road/path width heuristic,
manual highway seed map,
drivable mask boundary.
```

Road edges are auxiliary, not primary safety truth.

### Traffic lights and signs

Do not require for `Atlas-HA` v1.

---

# 30. Data collection strategy

Do not collect endless passive highway loops only. Use scenario-balanced targeted collection.

Track coverage buckets:

```text
highway segment ID
road type
lane index
traffic density
lead distance
lead relative speed
curvature bucket
weather
time of day
UI state
phone state
frame staleness/repeat state
route/maneuver command
lane-change state
blocker/closure state
recovery/perturbation state
```

Initial collection target:

```text
3–5 effective hours for lane keeping + following
6–10 effective hours for conservative lane changes
8–15 effective hours for blocker / closure / merge-or-slow behavior
```

Suggested first dataset mix:

```text
1.0 h clear daytime lane keeping
0.5 h rainy lane keeping
0.5 h nighttime lane keeping
1.0 h following traffic at varied speeds
0.5 h lane changes in light traffic
0.5 h recovery perturbations
0.5 h UI/phone/minimap robustness cases
0.5 h winding well-marked road / tight-curve speed control
```

After the first closed-loop model, prioritize failure mining over more passive data.

---

# 31. Training curriculum

The original large foundation-model Stage 1A/1B/1C curriculum is not the target for this compact model. Use an HA-specific curriculum.

## 31.1 HA-0: data validation and target generation

Build:

```text
dataset validator
target builders
token-cache builder
vehicle calibration tools
UI mask generator
```

Required checks:

```text
timestamps monotonic
frame freshness sane
actions aligned to frames
trajectory targets valid
lane targets valid/masked
lead labels valid
UI masks valid
train/val/test split by episode/segment
```

## 31.2 HA-A: short-context affordance training

Goal:

```text
teach the visual backbone and core heads to see lanes, road structure, lead vehicles, ego state, and basic candidate trajectories.
```

Train end-to-end with:

```text
current frame
+ a few selected recent frames if desired
+ actions
```

Short training offsets can be:

```text
[0.0, -0.1, -0.2, -0.5, -1.0, -1.5]
```

Backprop through these selected RGB frames.

Train:

```text
RegNet/BiFPN backbone
tokenizer
action encoder
lane head
lead head
ego head
trajectory head
confidence head
```

Use pretrained RegNet weights.

## 31.3 HA-B: 30-second cached-token temporal training

Goal:

```text
teach the model to use 30 seconds of detailed frame-token history without backpropagating through 600 image encodes.
```

Procedure:

```text
1. Run trained/EMA visual backbone over the corpus.
2. Save frame_tokens [128,192] and frame_summary [192] per model-timestep frame.
3. Train full-context temporal reader, summary reader, action encoder, fusion, and heads.
4. Freeze backbone initially.
5. Optionally fine-tune current-frame path later.
```

Training input:

```python
full_token_cache : [B, 600, 128, 192]
summary_cache    : [B, 120, 192]
actions_hist     : [B, 300, 6]
dt_hist          : [B, 300, 1]
```

Add temporal regularization:

```text
random context truncation
old-frame token dropout
cache-age jitter
phone/UI burst augmentation
frame-staleness augmentation
```

## 31.4 HA-C: policy and legality training

Goal:

```text
make trajectory candidates executable and compatible with legalizer/controller.
```

Generate legalizer-derived targets:

```python
target_legal_speed     : [B, N]
target_candidate_legal : [B, K]
```

Losses include:

```text
trajectory loss
candidate classification loss
legal speed loss
lateral acceleration penalty
curvature/steering limit penalty
candidate legality loss if implemented
```

## 31.5 HA-D: closed-loop DAgger-lite

Run live in GTA and mine failures:

```text
lane departures
steering oscillation
late braking
speed too high for curve
bad lane-change attempts
lane-change aborts
phone UI events
top-left popup events
frame-staleness bursts
winding road caution-mode cases
blocker/lane-closure failures
takeover false positives
takeover false negatives
```

Relabel with:

```text
human correction
privileged teacher
scripted recovery teacher
legalizer/controller oracle
```

Fine-tune with emphasis on recovery and low-confidence cases.

---

# 32. Losses

Implement:

```text
gtapilot/atlas/highway/losses.py
```

## 32.1 Trajectory loss

If `target_candidate` is available:

```text
L_traj = Huber(traj_candidates[:, target_candidate], target_traj)
L_candidate = CE(candidate_logits, target_candidate)
```

If candidate label is missing, use min-over-candidates:

```text
L_traj = min_k Huber(traj_candidates[:, k], target_traj)
```

## 32.2 Lane loss

```text
L_lane_lat = Huber(lane_lat_pred, target_lane_lat) over target_lane_valid
L_lane_valid = BCE(lane_valid_logit, target_lane_valid)
L_lane_conf = BCE(lane_conf, target_lane_conf)
```

## 32.3 Road-edge loss

If road-edge labels exist:

```text
L_road_edge = masked Huber/BCE
```

## 32.4 Lead loss

```text
L_lead_present = BCE(lead_present_logit, target_lead_present)
L_lead_state = Huber(lead_state, target_lead_state) only when lead is present
```

## 32.5 Adjacent loss

```text
L_adjacent_visible = mixed BCE/Huber for visible adjacent labels
L_lane_change_teacher = BCE for teacher lane-change OK labels
```

Do not use omniscient side/rear privileged actor state as direct regression if not observable.

## 32.6 Ego loss

```text
L_ego = Huber(ego_kinematics, target_ego)
```

## 32.7 Confidence/takeover loss

```text
L_scene_type = CE(scene_type_logits, target_scene_type)
L_takeover = BCE(takeover_required_logit, target_takeover_required)
L_road_conf = BCE/soft target loss for road_followable_conf
L_lane_tracking_conf = BCE/soft target loss for lane_tracking_conf
```

Important:

```text
target_takeover_required = 1 only for extreme no-viable-path cases.
```

Most well-marked non-highway roads should have:

```text
target_takeover_required = 0
scene_type = well-marked non-highway / road-like
```

## 32.8 Legal speed and dynamics losses

```text
L_legal_speed = Huber(predicted_speed_for_target_candidate, target_legal_speed)
```

Curvature/lateral acceleration penalty:

```text
a_lat_pred = speed_pred^2 * abs(kappa_pred)
L_latacc = mean(relu(a_lat_pred - a_lat_limit_soft)^2)
```

Curvature/steering penalty:

```text
kappa_max_soft = tan(steer_max_rad * safety_margin) / wheelbase_eff_m
L_curvature_limit = mean(relu(abs(kappa_pred) - kappa_max_soft)^2)
```

Optional:

```text
L_candidate_legal = BCE(candidate_feasible_logit, target_candidate_legal)
```

## 32.9 Smoothness and continuity losses

```text
L_path_smooth
L_yaw_consistency
L_speed_smooth
L_temporal_candidate_consistency
```

## 32.10 Example total loss

Initial weights:

```text
L =
  3.0 * L_traj
+ 1.0 * L_candidate
+ 1.0 * L_lane
+ 0.7 * L_lead
+ 0.5 * L_adjacent
+ 0.5 * L_ego
+ 0.5 * L_legal_speed
+ 0.2 * L_latacc
+ 0.2 * L_curvature_limit
+ 0.3 * L_scene_type
+ 0.2 * L_takeover
+ 0.1 * L_smoothness
```

Tune empirically.

---

# 33. Dataset splits and leakage control

Because GTA’s map is fixed, do not randomly split by frame.

Split by:

```text
episode
highway segment
scenario type
weather/time bucket
traffic density
```

Maintain:

```text
train
validation-known-segments
validation-heldout-segments
closed-loop-test-scenarios
```

Validation must include:

```text
clear highway
rain highway
night highway
winding marked road
lead slowdown
stopped blocker
left lane change
right lane change
phone popup
top-left popup
stale frame burst
```

---

# 34. Offline and live evaluation

## 34.1 Offline metrics

```text
trajectory ADE/FDE in long/lat coordinates
candidate classification accuracy
lane lateral error at 20/40/80/120 m
lane confidence calibration
lead distance error
lead relative speed error
ego speed/yaw-rate error
legalizer rejection rate
takeover false-positive rate
takeover false-negative rate
confidence calibration / ECE
```

## 34.2 Live metrics

```text
mean distance/time between interventions
lane departure count
collision count
barrier/curb hit count
phantom braking events
unsafe lane-change attempts
speed-too-high-for-curve events
candidate switch frequency
steering oscillation
controller saturation frequency
phone/UI robustness failures
frame-staleness recovery behavior
minimum-risk activations
takeover requests and reasons
```

---

# 35. Visualization requirements

Implement:

```text
gtapilot/atlas/highway/visualization.py
gtapilot/atlas/highway/tools/replay_highway.py
```

Overlay and log:

```text
raw RGB
sanitized scene RGB
UI mask
phone/popup/reticle state
lane predictions
road-edge predictions
lead estimate
adjacent lane availability
all K trajectory candidates
candidate logits
candidate legal mask
candidate reject reasons
selected candidate
curvature speed cap
lead speed cap
UI speed cap
ODD/caution speed cap
final authorized speed
ego speed estimate
controller steer/throttle/brake
supervisor mode
takeover reason
frame freshness/staleness
latency estimate
```

This is mandatory for debugging. The legalizer and supervisor should not be black boxes.

---

# 36. Live runtime flow

The live system should run:

```text
1. Capture raw 1920×1080 GTA frame.
2. Record source_frame_id, timestamp, freshness.
3. Run UI detector/sanitizer.
4. Resize sanitized scene to 1536×864.
5. Encode current frame with RegNet/BiFPN/tokenizer.
6. Push frame_tokens and frame_summary into caches.
7. Sample action history.
8. Run temporal readers and heads.
9. Update ego-state estimator.
10. Estimate control latency and compensate ego/trajectory.
11. Sanitize candidate trajectories.
12. Legalize trajectories and speed profiles.
13. Select candidate with FSM/cost.
14. Run lateral and longitudinal controllers.
15. Send GTA action command.
16. Log visualization/debug state.
```

Model inference can run at 20 Hz. Controller can run at 60 Hz using the most recent legalized trajectory advanced through ego-state prediction.

---

# 37. Codex acceptance checklist

Implementation is not complete until the following are true:

```text
1. AtlasHA model builds from config.
2. RegNetY-800MF pretrained backbone loads.
3. Forward pass returns all required tensors with exact expected shapes.
4. Runtime state supports 30-second full-token cache.
5. Current-frame-only live inference works.
6. Historical RGB is never re-encoded during live inference.
7. Cross-attention uses bottleneck queries, not full self-attention over cached tokens.
8. Trajectory sanitizer rejects pathological candidates.
9. Legalizer computes curvature, speed caps, legality masks, and reject reasons.
10. Controller uses authorized speed, not raw model speed.
11. FSM vetoes unsafe lane changes.
12. UI preprocessor detects/masks minimap, phone, popups, and reticle.
13. Frame staleness triggers caution/minimum-risk behavior.
14. Takeover is hysteresis-gated and rare.
15. Dataset loader supports required labels and valid masks.
16. Target builders support trajectory, lane, lead, ego, candidate, legal-speed labels.
17. Training scripts exist for HA-A, HA-B, HA-C, HA-D.
18. Replay visualization renders all key model, legalizer, and controller outputs.
19. Model stats/profiling tool prints parameter counts, FLOPs estimates, latency, and VRAM use.
20. Closed-loop live inference can run through the GTA Pilot IPC/control system.
```

---

# 38. Final implementation priorities

Build in this order:

```text
1. config.py, state.py, model.py skeleton
2. RegNet/BiFPN/tokenizer
3. temporal cache + cross-attention readers
4. action encoder
5. output heads
6. losses and dataset target interfaces
7. trajectory sanitizer/legalizer/speed profiler
8. controller and FSM
9. UI preprocessor
10. live inference adapter
11. visualization/replay tools
12. training scripts
13. calibration/profiling tools
14. DAgger/failure-mining loop
```

The first closed-loop milestone should be:

```text
lane centering + legal speed through curves + simple following
```

Then add:

```text
conservative lane changes
```

Then add:

```text
blocker/lane-closure merge-or-slow behavior
```

---

# 39. Final design summary

`Atlas-HA` should be a small, explicit, controllable highway-assist system:

```text
pretrained compact visual backbone
+ 30-second cached full-token context
+ action history
+ lane/lead/ego/trajectory heads
+ semantic maneuver candidates
+ deterministic trajectory legality layer
+ classical controller
+ minimum-risk supervisor
```

The most important implementation constraints are:

```text
1. Use first-person hood-camera assumptions only.
2. Use long/lat coordinate naming.
3. Use 30 seconds of cached full frame tokens.
4. Never re-encode historical RGB during live inference.
5. Never run full self-attention over the historical token cache.
6. Keep privileged labels observability-aware.
7. Let the controller/legalizer enforce dynamic feasibility.
8. Make hard takeover rare.
9. Treat phone/UI as temporary sensor occlusion, not road structure.
10. Prefer targeted scenario data and closed-loop failure mining over passive mileage.
```

This specification should be implemented as a separate Highway Assist variant under `gtapilot/atlas-highway/`, leaving the larger Atlas foundation model intact.
