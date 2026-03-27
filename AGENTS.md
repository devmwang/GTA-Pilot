# AGENTS.md - GTA Pilot Coding Agent Playbook

Audience: coding agents working inside this repository.

Goal (current codebase): maintain and extend the Atlas-oriented runtime and data
collection stack:

display capture OR video override -> generic channel IPC -> visualization / blackbox
generalized manual input capture -> generic channel IPC -> visualization / blackbox

The current runtime only records and consumes inference-time sources:

- front RGB frames
- executed actions
- timestamps and source metadata

Privileged sensors and labels are a later ScriptHook-based project. Do not
present them as already implemented.

# When making changes: Always use hard-cutover changes, never have backwards compatibility.

## 1. Ground Truth

- Language: Python 3.13+ (`pyproject.toml` currently requires `~=3.13`).
- Package: `gtapilot`.
- Atlas scaffolding exists under `gtapilot/atlas`.
- PyTorch and torchvision are now project dependencies.
- Runtime stack is still multi-process and ZeroMQ-based.
- Live capture is primarily the native Windows DX11 executable in `bin/`.
- Python video override remains for offline testing, but live capture is the
  native DX11 path.

## 2. Entry Points

Main CLI entrypoint: `gtapilot/main.py`

Typical runs:

```bash
uv venv .venv
uv pip install -e .

# Live capture
uv run ./gtapilot/main.py

# Video override instead of live capture
uv run ./gtapilot/main.py --video-override path/to/video.mp4

# Select display index explicitly
uv run ./gtapilot/main.py --display-id 0
```

Shutdown triggers:

- ESC in the coordinator console on Windows
- Ctrl+C / SIGINT / SIGTERM to the coordinator
- any child process exiting

After the first trigger, the coordinator force-terminates the remaining child
processes. There is no shared shutdown event.

## 3. Active Processes

Coordinator: `gtapilot.coordinator.coordinator.main`

Current worker set:

1. `SettingsRuntime`
   `gtapilot.ipc.settings_runtime.main`
   Owns the runtime settings snapshot and serves it over a dedicated settings
   IPC plane.

2. `DisplayCaptureDX11`
   Native executable `bin/DisplayCaptureDX11.exe`
   Live desktop capture on Windows, publishes a fixed 60 Hz RGB stream to the
   `vision.frames` channel.

3. `DisplayOverride`
   `gtapilot.display_capture.display_override.main`
   Uses a video file instead of live capture and publishes a fixed 60 Hz RGB
   stream to the `vision.frames` channel.

4. `ActionCapture`
   `gtapilot.input_capture.input_capture.main`
   Polls keyboard state plus one XInput controller at 60 Hz, publishes
   generalized driving action packets to `input.actions`, and updates mutable
   runtime settings via the settings service.

5. `Visualization`
   `gtapilot.visualization.visualization.main`
   Displays the latest frame with FPS and action overlays.

6. `Blackbox`
   `gtapilot.blackbox.blackbox.main`
   Optional recorder enabled by `BLACKBOX_ENABLED` in `gtapilot/config.py`.

Deprecated runtime systems based on the old depth / YOLO / lane-mask pipeline
have been removed from the live process graph.

## 4. Generic Channel IPC

Public modules:

- `gtapilot.ipc.types`
- `gtapilot.ipc.codecs`
- `gtapilot.ipc.channel`
- `gtapilot.ipc.channels`

Pattern:

- ZeroMQ PUB/SUB
- publishers bind
- subscribers connect
- every stream uses the same multipart wire format:
  `[topic, envelope_json, payload_bytes]`

Current active channel specs:

- `VISION_FRAMES_CHANNEL`
  - name: `vision.frames`
  - port: `55550`
  - topic: `b"frames"`
  - codec: `RawRGBFrameCodec`
  - default buffer size / HWM tuned for 60 Hz runtime collection:
    buffer `16`, send `8`, receive `8`
- `INPUT_ACTIONS_CHANNEL`
  - name: `input.actions`
  - port: `55552`
  - topic: `b"actions"`
  - codec: `JsonDataclassCodec(ActionPacket)`

Shared envelope contract (version `1`):

- `v`
- `channel`
- `encoding`
- `sequence_id`
- `message_timestamp_ns`
- `publish_timestamp_ns`
- `source`
- `metadata`

Vision payload contract:

- raw RGB `uint8`
- shape `(H, W, 3)`
- encoding `raw_rgb_v1`
- metadata fields:
  - `w`
  - `h`
  - `channels`
  - `dtype`
  - `frame_id`
  - `nominal_fps`
  - `capture_timestamp_ns`
  - `is_repeat`

Action payload contract:

- `ActionPacket`
- action vector order:
  `[steer, throttle, brake, handbrake, reverse, pilot_active]`
- payload fields:
  - `steer`
  - `throttle`
  - `brake`
  - `handbrake`
  - `reverse`
  - `pilot_active`
  - `active_device`
  - `device_actions`
  - `device_inputs`

Current action source is keyboard plus one XInput controller. `reverse` is not
inferred reliably yet and is currently published as `0.0` until vehicle-state
integration lands.

Rules:

1. Preserve the generic envelope contract unless you intentionally version it.
2. New streams should be added by defining a `ChannelSpec` and payload codec or
   payload schema, not by adding one-off IPC modules.
3. Keep subscribers able to shut down cleanly: stop background thread, close
   socket, terminate context.
4. Do not silently switch to compression or a different transport without
   documentation and an intentional encoding change.

## 5. Runtime Settings IPC

Public modules:

- `gtapilot.ipc.settings_types`
- `gtapilot.ipc.settings_registry`
- `gtapilot.ipc.settings_client`
- `gtapilot.ipc.settings_runtime`

This is separate from the stream channel IPC. Use it for mutable runtime state
that late subscribers must be able to read immediately.

Transport:

- `settings.updates`
  - ZeroMQ PUB/SUB
  - port `55553`
- `settings.rpc`
  - ZeroMQ REQ/REP
  - port `55554`

Current runtime-visible settings:

- `blackbox.enabled`
  - seeded from `BLACKBOX_ENABLED`
  - read-only
- `blackbox.recording_enabled`
  - seeded from `BLACKBOX_RECORD_ON_START`
  - mutable
  - current writer: `manual_input`
- `blackbox.preroll_seconds`
  - seeded from `BLACKBOX_PREROLL_SECONDS`
  - mutable for trusted local tools only
- `blackbox.record_hotkey`
  - seeded from `BLACKBOX_RECORD_HOTKEY`
  - read-only in v1

Rules:

1. Do not tunnel runtime settings through stream channels.
2. `SettingsRuntime` is the only authority for accepted writes.
3. Clients must load a startup snapshot via RPC and then listen for updates.
4. Runtime settings are not persisted across fresh launches.

## 6. Blackbox Recorder

The blackbox currently records inference-time data only.

Outputs under `blackbox-recordings/`:

- `capture_<timestamp>_video.mkv`
- `capture_<timestamp>_metadata.json`

Current manifest schema version: `5`

The manifest records:

- session metadata
- session video settings
- per-frame metadata and video frame index
- per-frame envelope data
- frame-aligned action payload
- frame-aligned action envelope data
- frame-aligned action vector
- raw action stream entries

Behavior notes:

- blackbox starts idle by default even when `BLACKBOX_ENABLED = True`
- runtime recording is controlled by the `blackbox.recording_enabled` setting
- keyboard hotkey `F8` currently flips that setting via the input-capture
  process
- while idle, blackbox keeps a bounded in-memory pre-roll buffer
- each start/stop cycle produces a separate `capture_<timestamp>_*` pair
- frames are encoded into H.264 video in an MKV container via `ffmpeg`
- metadata is flushed incrementally during capture
- abrupt termination can still lose a small tail of in-memory state
- the JSON manifest is the authoritative timestamp/alignment source, not the
  container timestamps
- frame producers must publish `nominal_fps` in frame metadata; blackbox does
  not guess it
- current live capture and video override producers publish `nominal_fps=60.0`
  for the runtime collection path

Do not silently change the manifest format. If it must evolve, bump
`schema_version`.

## 7. Atlas Data Collection Scope

For now, only collect inference-time sources needed by Atlas:

- front RGB frames
- action vectors
- timestamps
- source identity / repeat flags

Do not add fake placeholders for privileged labels. Those belong in a later
ScriptHook-based data engine.

## 8. Atlas Temporal Architecture

Atlas no longer uses the old flat `num_frames`/`rgb_past` temporal contract.

The active temporal API is the three-tier context design:

- recent full-token bank
- older compressed-history bank
- sparse mid-summary bank
- persistent dynamic and speculative memory slots

Student baseline:

- `24 Hz`
- `32` recent full-token frames
- `64` older compressed frames
- `120` mid-summary steps at `6 Hz`
- `128` action-history steps
- `64` dynamic slots
- `16` speculative slots

Teacher target:

- `36 Hz`
- `48` recent full-token frames
- `96` older compressed frames
- `180` mid-summary steps at `6 Hz`
- `192` action-history steps
- `96` dynamic slots
- `24` speculative slots

Canonical training interface:

- `rgb_recent`, `dt_recent`
- `rgb_older`, `dt_older`
- `rgb_mid`, `dt_mid`
- `actions_hist`, `dt_hist`

Current Stage 1 data contract:

- blackbox source capture remains `60 Hz` for both frames and actions
- student training resamples that source timeline onto `24 Hz` recent/older/action grids
- teacher training resamples that source timeline onto `36 Hz` recent/older/action grids
- `mid_summary_hz = 6` is a sparse sampling grid over the same source timeline
- for each desired model timestamp, the loader uses the latest source frame or raw
  action packet at or before that timestamp
- `actions_hist` should prefer the raw blackbox `actions` stream and only fall back
  to frame-aligned `action_vector` for older recordings that do not have it
- privileged Stage 1B / 1C targets should live in the sibling
  `capture_<timestamp>_privileged/` directory and be indexed through
  `AtlasTemporalClipIndex.privileged_dir`
- build aligned privileged packages with
  `python -m gtapilot.atlas.data.build_stage1b_privileged_dataset ...`
- privileged manifests must carry source alignment metadata and should be
  validated against the source blackbox clip before training
- Stage 1B track supervision is sparse-lag only; keep
  `cfg.geometry.track_lag_indices`, sparse privileged targets, and geometry-head
  output dimensionality in sync

Atlas state must carry:

- `recent_cam_cache`, `older_cam_cache`, `mid_summary_cache`
- `recent_dt_cache`, `older_dt_cache`, `mid_dt_cache`
- `recent_valid`, `older_valid`, `mid_valid`
- `action_buffer`, `dt_buffer`
- `dynamic_slots`, `speculative_slots`
- `dynamic_slot_age_s`, `speculative_slot_age_s`
- `dynamic_slot_alive`, `speculative_slot_alive`

When changing Atlas temporal code:

1. Preserve the three-tier interface; do not reintroduce flat `rgb_past`.
2. Use exact `dt` values from data sources; do not fake all clips to `24 Hz`.
3. Ignore invalid temporal-bank entries with masks instead of attending to zero
   padding.
4. Keep hidden-actor reasoning explicit through `dynamic_slots`,
   `speculative_slots`, `dyn_flow_bev`, `occl_risk_bev`, and `provenance`.

## 9. Safe Extension Patterns

When adding a new worker:

1. Create `gtapilot/<module>/<file>.py` with `def main(...):`
2. Initialize heavy resources inside `main()`, not at import time
3. Use a simple `while True` loop and rely on process termination semantics
4. Register the worker in `build_processes`
5. Prefer additive, reversible changes over broad rewrites

If a worker needs graceful flushing, make it periodic and incremental. Do not
reintroduce a global shutdown event.

## 10. Deprecated Systems

These old systems are no longer part of the runtime contract:

- old perception worker stack based on YOLO / lane masks / drivable masks
- older IPC modules such as `gtapilot.ipc.messaging` and the deleted
  stream-specific frame/action IPC layer
- planner-facing visualization overlays driven by that stack

When cleaning up similar code in the future:

- remove the worker from the coordinator
- remove the IPC surface
- remove dead imports and docs in the same change

## 11. Testing Guidance

Do not automatically add tests, test files, or test scaffolding.

Only create or update tests if the user explicitly asks for them.

For early runtime work in this repo, prioritize implementation speed and manual
validation over adding unit or smoke tests by default.

## 12. Performance Notes

- target runtime collection FPS is currently 60
- teacher temporal support targets 36 Hz inputs offline
- visualization should avoid unnecessary resizes
- heavy work should not run in the display capture loop
- if latency rises, prefer conflation or bounded buffering over unbounded queues

For Atlas itself, keep the current runtime focused on data movement and
recording. Do not move training or inference into the capture workers.

## 13. Documentation Rules

`AGENTS.md` is the source-of-truth runtime playbook for this repository.

If you change the generic channel framework, blackbox schema, or the active
runtime graph, update `AGENTS.md` in the same change.

## 14. When Unsure

Inspect the runtime in data-flow order:

1. `gtapilot/ipc/settings_runtime.py`
2. `gtapilot/ipc/settings_client.py`
3. `gtapilot/display_capture/` or `gtapilot/native/display_capture/`
4. `gtapilot/ipc/channel.py`
5. `gtapilot/ipc/channels.py`
6. `gtapilot/input_capture/input_capture.py`
7. `gtapilot/visualization/visualization.py`
8. `gtapilot/blackbox/blackbox.py`
9. `gtapilot/atlas/`

Prefer small, focused changes that keep the frame/action capture contract
stable.
