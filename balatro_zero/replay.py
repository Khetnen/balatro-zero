"""Ring replay buffer over (Obs, policy_target, z_win, z_prog) samples.

The policy target is a positional pi vector (np.ndarray, Discrete(500))
for v3/v4 nets, or a targets.CandidateSet for factored (V5) nets. The
buffer is format-agnostic; sample() stacks positional targets into one
array and returns factored ones as a list for collate_candidate_sets.
"""

from __future__ import annotations

import numpy as np

from balatro_zero.state import Obs, stack_obs


class ReplayBuffer:
    def __init__(self, capacity: int = 60_000) -> None:
        self.capacity = capacity
        self._obs: list[Obs] = []
        self._pi: list[np.ndarray] = []
        self._z_win: list[float] = []
        self._z_prog: list[float] = []

    def add(self, samples: list[tuple[Obs, np.ndarray, float, float]]) -> None:
        for obs, pi, z_win, z_prog in samples:
            self._obs.append(obs)
            self._pi.append(pi)
            self._z_win.append(z_win)
            self._z_prog.append(z_prog)
        overflow = len(self._obs) - self.capacity
        if overflow > 0:
            del self._obs[:overflow]
            del self._pi[:overflow]
            del self._z_win[:overflow]
            del self._z_prog[:overflow]

    def __len__(self) -> int:
        return len(self._obs)

    def sample(self, batch_size: int, rng: np.random.Generator):
        idx = rng.integers(0, len(self._obs), size=min(batch_size, len(self._obs)))
        flat, jid, cid, mid = stack_obs([self._obs[i] for i in idx])
        pis = [self._pi[i] for i in idx]
        return (
            flat,
            jid,
            cid,
            mid,
            np.stack(pis) if isinstance(pis[0], np.ndarray) else pis,
            np.asarray([self._z_win[i] for i in idx], dtype=np.float32),
            np.asarray([self._z_prog[i] for i in idx], dtype=np.float32),
        )
