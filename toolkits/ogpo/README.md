# OGPO Q rollout visualization

`visualize_q_rollout.py` generates successful LIBERO actor trajectories and plots
every critic head for the executed action chunk and its negation. A working actor
can come from an earlier checkpoint while the critic comes from a collapsed run.

Run from the repository root using the training Python environment. Replace the
example checkpoint paths with the two `global_step_N` directories to inspect:

```bash
python toolkits/ogpo/visualize_q_rollout.py \
  --actor-checkpoint /path/to/good_actor/global_step_20 \
  --critic-checkpoint /path/to/collapsed_run/global_step_200 \
  --episodes 10
```

`--checkpoint PATH` (alias `--ckpt`) remains the shared fallback for both models.
For example, `--checkpoint BAD --actor-checkpoint GOOD` uses `GOOD` for rollout
and `BAD` for Q. Either source can also be its `actor/model_state_dict/full_weights.pt`.
Both sources must be OpenPI RLinf OGPO checkpoints. Their action chunk length and
environment action dimension must match; Q head counts need not match.

Each model reads its own saved `tensorboard/config.yaml` and original normalization
assets. `--actor-config` and `--critic-config` override discovery independently;
`--config` explicitly supplies a shared fallback. Actor sampling uses the actor's
config. `--mode train` (default) uses training SDE sampling; `--mode eval` uses ODE.

The critic is loaded as a separate complete model, including its image/text encoder
and `state_proj`, because Q depends on these features as well as `q_head`. With
different models, the actor's decoded environment commands are encoded using the
critic's `encode_replay_actions` before scoring. Q therefore evaluates the commands
sent to LIBERO in the critic's own normalization coordinates. For identical weights,
config and device, the script reuses one model and keeps the previous exact normalized
action diagnostic, without a decode/encode round-trip.

By default the second curve negates every normalized critic action coordinate,
including the gripper. `--negate-space env` instead negates all decoded environment
action coordinates and re-encodes them with the critic. Only actor actions are
executed in the environment. Non-finite Q values are plotted as gaps, shown as
`nan`/`inf` in each head's title, and saved as JSON `null` with original values in
`q_value_text`, allowing a numerically collapsed critic to be inspected.

The environment comes from the **critic** log's `env.train` (or `env.eval` when
`--env-split eval` is selected). Its `specific_reset_id`, task filter, fixed/ordered
reset flags, seed, initial-state settings and episode limit are retained. In the
original run, global reset ID 0 means task 0 / initial-state index 0. The script
checks this restriction after every reset. `--episodes` counts successful videos,
with up to `100 * episodes` attempts by default (`--max-attempts` overrides it).

Outputs default to `<critic-checkpoint>/q_visualizations/<timestamp>/`; `--output-dir`
overrides this with a new directory. Each MP4 has the rollout on the left and two
curves per Q head on the right. Per-episode JSON and `manifest.json` record both
checkpoint sources and the actual action coordinates. Both training configs and
the effective visualization environment config are saved beside the videos.

Two different sources keep two complete models in memory. To place them on separate
GPUs, add `--device cuda:0 --critic-device cuda:1`. The default puts both on `--device`.

Regression cases were added to `tests/unit_tests/test_models.py` for checkpoint
selection, cross-model action normalization, critic feature ownership, both negation
modes, shared-model compatibility and incompatible action shapes. They have not been
executed; the changes were checked statically without loading models or environments.
