# PLWT Scaffold: Photon → Latent World → Trajectory

This artifact is a runnable code scaffold for the **PLWT-S1080** student and **PLWT-T1080-Priv** teacher discussed in the design spec.

It includes:

- config dataclasses for the 1080p student, the privileged teacher, and a tiny smoke-test variant
- explicit recurrent state objects
- module stubs / lightweight implementations for:
  - visual dual-path tokenizer
  - temporal camera mixer
  - action encoder
  - learned ego-state filter
  - geometry lifter
  - observation pool
  - rolling latent world memory
  - proposal + evaluator planner
  - auxiliary heads (lane segments, map elements, occupancy, BEV-lite, actors, ego)
  - route adapter
  - slow-loop reasoner bridge
  - inference decode scheduler
- stage-aware loss utilities
- example smoke test / usage script

## Important caveat

This scaffold is meant to be **explicit, modular, and runnable**. It is **not** claiming that every internal block here is the final best implementation.

In particular:

- the config / tensor contracts match the recommended architecture
- the module internals are intentionally compact enough to run and iterate on
- you should expect to swap in stronger internals over time:
  - stronger local-attention visual stages
  - richer world update blocks
  - more task-specific lane / map losses
  - teacher-student distillation code
  - GTA-specific label loaders and teacher planners

Think of this artifact as the **first implementation substrate**, not the final research endpoint.

## Package layout

```text
plwt/
  __init__.py
  config.py
  state.py
  scheduler.py
  losses.py
  model.py
  modules/
    common.py
    visual_dualpath.py
    temporal_mixer.py
    action_encoder.py
    ego_filter.py
    geometry_lifter.py
    observation_pool.py
    adapters.py
    world_memory.py
    planner.py
    aux_heads.py

configs/
  plwt_s1080.json
  plwt_t1080_priv.json
  plwt_smoke.json

examples/
  smoke_train_step.py

tests/
  smoke_test.py
```

## Canonical online inputs

```python
rgb_t_raw      : [B, 3, 1080, 1920]
action_prev    : [B, 6]   # [steer, throttle, brake, handbrake, reverse, pilot_active]
dt_t           : [B, 1]
route_polyline : optional [B, 32, 3]
nav_cmd        : optional [B, 6]
reasoner_tok   : optional [B, 8, D_reasoner]
state          : PLWTState
```

`pilot_active` convention:

- `0 = manual / human intervention`
- `1 = policy in control`

## Canonical outputs from `model.step(...)`

Always returned:

- `best_traj`
- `traj`
- `score`
- `reward_terms`
- `pose_delta`
- `kinematics`
- `depth_logits`
- `depth_mean`
- `ray_conf`
- `frustum_tokens`
- `static_grid`
- `persistent_tokens`

Conditionally returned by the decode scheduler:

- lane head:
  - `centerline`
  - `left_boundary`
  - `right_boundary`
  - lane attribute logits / topology logits
- map head:
  - `map_poly`
  - `map_elem_cls`
  - `lane_elem_bind`
- BEV-lite head:
  - `bev_lite`
  - `provenance`
- actor head:
  - `actor_cls`
  - `actor_box`
  - `actor_vel`
  - `actor_future`
- occupancy head:
  - `occ_state`
  - `occ_sem`
- ego head:
  - `ego_out`
  - `ego_logvar`

## Recommended runtime decode scheduling

### Drive mode
Use `mode="drive"`.

Cadence defaults:

- every step:
  - planner
  - ego
  - lane
  - map
  - BEV-lite / provenance
- every 2 steps:
  - actors
- every 3 steps:
  - heavy occupancy / semantics visualization

### Inspect mode
Use `mode="inspect"`.

All configured heads decode every step.

## Training stage expectations

### Stage 1A
Use only:

- visual tokenizer
- temporal mixer
- projector-space camera token losses

Recommended output keys to provide if you add JEPA projector heads:

- `cam_projector_pred`
- `cam_projector_target`
- optional `cam_mask`

### Stage 1B
Turn on:

- ego filter
- geometry lifter
- pseudo / GT targets for:
  - depth
  - tracks
  - ego motion

### Stage 1C
Turn on:

- world memory
- world-token JEPA projector heads

Expected keys if you add projector heads:

- `world_projector_pred`
- `world_projector_target`
- optional `world_mask`

### Stage 2
Turn on auxiliary GTA-supervised heads:

- lane / map
- occupancy
- BEV-lite / provenance
- actors
- ego

### Stage 3
Turn on planner:

- proposal generator
- evaluator
- teacher costs
- best / diverse candidate supervision

### Stage 4
Joint end-to-end training with aux losses retained.

## Lane label compatibility

The internal lane-segment schema was chosen to be broadly compatible with:

- OpenLane / OpenLane-V2 style 3D lane geometry and topology
- ONCE-3DLanes style monocular 3D lane geometry
- BDD100K lane attributes / visible lane-mark classes
- CULane visible lane curves
- ApolloScape lane segmentation

The scaffold does **not** ship public dataset converters, but the head output schema is already aligned for those converters.

## Suggested next implementation tasks

1. Replace the current visual stages with stronger local-attention blocks if runtime allows.
2. Add GTA-specific dataset loaders and privileged label generation.
3. Add teacher-planner candidate generation and ranking labels.
4. Add explicit projector heads for Stage 1A and Stage 1C JEPA training.
5. Add richer set losses for lane topology and map binding.
6. Add a distillation path from `PLWT-T1080-Priv` to `PLWT-S1080`.

## Quick start

```bash
python examples/smoke_train_step.py
python tests/smoke_test.py
```

These use the tiny `PLWT-SMOKE` config, not the full 1080p config.
