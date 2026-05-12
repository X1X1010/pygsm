"""Internal helpers for level-of-theory state and scratch management."""

from __future__ import annotations

import os


def normalize_states(states):
    singlets = [state for state in states if state[0] == 1]
    doublets = [state for state in states if state[0] == 2]
    triplets = [state for state in states if state[0] == 3]
    quartets = [state for state in states if state[0] == 4]
    quintets = [state for state in states if state[0] == 5]

    len_singlets = max(singlets, key=lambda state: state[1])[1] + 1 if singlets else 0
    len_doublets = len(doublets)
    len_triplets = len(triplets)
    len_quartets = len(quartets)
    len_quintets = len(quintets)
    expected_length = len_singlets + len_doublets + len_triplets + len_quartets + len_quintets

    if len(states) >= expected_length:
        return states, False

    fixed_states = []
    for index in range(len_singlets):
        fixed_states.append((1, index))
    for index in range(len_triplets):
        fixed_states.append((3, index))
    return fixed_states, True


def infer_gradient_states(states, gradient_states, calc_grad):
    if gradient_states is None and calc_grad:
        print(' Assuming gradient states are ', states)
        return states
    return gradient_states


def initialize_job_data(job_data):
    prepared = dict(job_data)
    prepared['orbfile'] = prepared.get('orbfile', '')
    prepared['lot'] = prepared.get('lot', None)
    return prepared


def node_scratch_dir(scratch_root, string_id, node_id):
    return os.path.join(scratch_root, f'{string_id:03}', str(node_id))


def energy_file_path(scratch_root, string_id, node_id):
    return os.path.join(scratch_root, f'{string_id:03}', f'E_{node_id}.txt')


def ensure_node_scratch_dir(scratch_root, string_id, node_id):
    scratch_dir = node_scratch_dir(scratch_root, string_id, node_id)
    print(f' making folder {scratch_dir}')
    os.makedirs(scratch_dir, exist_ok=True)
    return scratch_dir
