"""
traj_transforms.py

Contains trajectory transforms used in the orca data pipeline. Trajectory transforms operate on a dictionary
that represents a single trajectory, meaning each tensor has the same leading dimension (the trajectory length).
"""

import logging
from typing import Dict, Optional, Sequence

import tensorflow as tf


def chunk_act_obs(
    traj: Dict,
    window_size: int,
    future_action_window_size: int = 0,
    future_obs_window_size: int = 0,
) -> Dict:
    """
    Chunks actions and observations into the given window_size.

    "observation" keys are given a new axis (at index 1) of size `window_size` containing `window_size - 1`
    observations from the past and the current observation. "action" is given a new axis (at index 1) of size
    `window_size + future_action_window_size` containing `window_size - 1` actions from the past, the current
    action, and `future_action_window_size` actions from the future. "pad_mask" is added to "observation" and
    indicates whether an observation should be considered padding (i.e. if it had come from a timestep
    before the start of the trajectory).

    Every timestep in the trajectory becomes a training sample. When the current step is close enough to the
    end of the episode that the future-action window would run past the last frame, the out-of-range future
    indices are clamped to the final frame of the episode (i.e. the last action is repeated for those
    positions) instead of dropping the sample. A top-level "pad_mask_future_actions" tensor (shape
    [traj_len, window_size + future_action_window_size]) is added to indicate which action positions are
    backed by real data versus filled by clamping.
    """
    traj_len = tf.shape(traj["action"])[0]
    effective_traj_len = traj_len
    chunk_indices = tf.broadcast_to(tf.range(-window_size + 1, 1), [effective_traj_len, window_size]) + tf.broadcast_to(
        tf.range(effective_traj_len)[:, None], [effective_traj_len, window_size]
    )

    action_chunk_indices = tf.broadcast_to(
        tf.range(-window_size + 1, 1 + future_action_window_size),
        [effective_traj_len, window_size + future_action_window_size],
    ) + tf.broadcast_to(
        tf.range(effective_traj_len)[:, None],
        [effective_traj_len, window_size + future_action_window_size],
    )

    floored_chunk_indices = tf.maximum(chunk_indices, 0)

    goal_timestep = tf.fill([effective_traj_len], traj_len - 1)

    # Clamp out-of-range action indices: negatives -> first frame, beyond-end -> last frame (repeat last action).
    floored_action_chunk_indices = tf.minimum(tf.maximum(action_chunk_indices, 0), goal_timestep[:, None])

    traj["observation"] = tf.nest.map_structure(lambda x: tf.gather(x, floored_chunk_indices), traj["observation"])
    traj["action"] = tf.gather(traj["action"], floored_action_chunk_indices)

    # indicates whether an entire observation is padding
    traj["observation"]["pad_mask"] = chunk_indices >= 0

    # True for action positions backed by real data; False where the index was clamped to the last frame.
    traj["pad_mask_future_actions"] = action_chunk_indices <= goal_timestep[:, None]

    # Optionally chunk **future** observations for the predictive-token branch (future-vision target).
    # Indices for step t are [t+1, ..., t+future_obs_window_size]. Out-of-range indices are clamped to
    # the last frame; `pad_mask_future_obs` records which positions are real vs clamped.
    # NOTE: We store these at the top level (sibling to "observation") so the frame-level `dl.vmap`
    # pipeline that decodes/augments observation images does not see them (mixing leading dims would
    # break vmap). A dedicated frame transform decodes/resizes them later.
    if future_obs_window_size > 0 and "image_primary" in traj["observation"]:
        future_obs_indices = tf.broadcast_to(
            tf.range(1, 1 + future_obs_window_size),
            [effective_traj_len, future_obs_window_size],
        ) + tf.broadcast_to(
            tf.range(effective_traj_len)[:, None],
            [effective_traj_len, future_obs_window_size],
        )
        floored_future_obs_indices = tf.minimum(tf.maximum(future_obs_indices, 0), goal_timestep[:, None])
        # Only gather the primary image (wrist views are not used as prediction targets).
        traj["image_primary_future"] = tf.gather(
            traj["observation"]["image_primary"], floored_future_obs_indices
        )
        traj["pad_mask_future_obs"] = future_obs_indices <= goal_timestep[:, None]

    # Truncate other elements of the trajectory dict
    traj["task"] = tf.nest.map_structure(lambda x: tf.gather(x, tf.range(effective_traj_len)), traj["task"])
    traj["dataset_name"] = tf.gather(traj["dataset_name"], tf.range(effective_traj_len))
    traj["absolute_action_mask"] = tf.gather(traj["absolute_action_mask"], tf.range(effective_traj_len))

    return traj


def make_actions_relative_to_current_state(traj: Dict, relative_action_mask: Optional[Sequence[bool]]) -> Dict:
    """Convert action chunks to OpenPI-style deltas relative to the current state's proprio."""
    if relative_action_mask is None:
        return traj

    if "proprio" not in traj["observation"]:
        raise ValueError("Relative actions require observation['proprio'].")

    mask = tf.convert_to_tensor(relative_action_mask, dtype=tf.bool)
    dims = len(relative_action_mask)
    action_dim = traj["action"].shape[-1]
    proprio_dim = traj["observation"]["proprio"].shape[-1]
    if action_dim is not None and dims > action_dim:
        raise ValueError(f"Length of relative_action_mask ({dims}) exceeds action dimension ({action_dim}).")
    if proprio_dim is not None and dims > proprio_dim:
        raise ValueError(f"Length of relative_action_mask ({dims}) exceeds proprio dimension ({proprio_dim}).")

    # observation["proprio"] has shape [T, window_size, D] after chunking; the last window entry is current state.
    current_state = traj["observation"]["proprio"][:, -1, :dims]
    action_prefix = traj["action"][..., :dims]
    relative_prefix = tf.where(mask, action_prefix - current_state[:, None, :], action_prefix)
    traj["action"] = tf.concat([relative_prefix, traj["action"][..., dims:]], axis=-1)
    return traj


def subsample(traj: Dict, subsample_length: int) -> Dict:
    """Subsamples trajectories to the given length."""
    traj_len = tf.shape(traj["action"])[0]
    if traj_len > subsample_length:
        indices = tf.random.shuffle(tf.range(traj_len))[:subsample_length]
        traj = tf.nest.map_structure(lambda x: tf.gather(x, indices), traj)

    return traj


def add_pad_mask_dict(traj: Dict) -> Dict:
    """
    Adds a dictionary indicating which elements of the observation/task should be treated as padding.
        =>> traj["observation"|"task"]["pad_mask_dict"] = {k: traj["observation"|"task"][k] is not padding}
    """
    traj_len = tf.shape(traj["action"])[0]

    for key in ["observation", "task"]:
        pad_mask_dict = {}
        for subkey in traj[key]:
            # Handles "language_instruction", "image_*", and "depth_*"
            if traj[key][subkey].dtype == tf.string:
                pad_mask_dict[subkey] = tf.strings.length(traj[key][subkey]) != 0

            # All other keys should not be treated as padding
            else:
                pad_mask_dict[subkey] = tf.ones([traj_len], dtype=tf.bool)

        traj[key]["pad_mask_dict"] = pad_mask_dict

    return traj
