import random

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx
import topology_conf

class Request:
    def __init__(self, src: int, dst: int, req_id: int, arrival_t: int):
        self.src = src
        self.dst = dst
        self.id = req_id
        self.arrival_t = arrival_t

    def as_tuple(self):
        return (self.src, self.dst, self.id, self.arrival_t)

class QKDNSchedulingEnv(gym.Env):
    """
    QKDN Scheduling Environment (Gymnasium-style)

    - Each time step, n requests arrive (n in [1, max_requests_per_step]).
    - The agent selects exactly one of them (action = index in [0, R_max-1]).
    - The selected request is served via a weighted shortest path (WSP).
      Edge weight = 1 / (remaining_keys + epsilon) + alpha * hop_cost.
    - If a feasible path exists with remaining_keys >= consume_key_size along all edges:
        success (+1 reward), keys consumed.
      Else:
        failure (+0 reward by default, optional penalty).

    Observation Dict:
      num_key: (N,N) float32 normalized [0..1] remaining-keys matrix
      requests: (R_max, 4) float32 = [src, dst, path_len_hops, path_min_keys]
      mask: (R_max,) bool for valid request slots
      t: float32 normalized time progress in [0,1]

    Action:
      Discrete(R_max). Choosing a masked=False index yields a small penalty.

    Episode ends at max_time_steps.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        topology_conf: dict,
        max_time_steps: int = 200,
        max_requests_per_step: int = 8,
        consume_key_size: int = 1,
        key_pool_size: int = 64,
        initial_key_lifetime: int = 32,
        generate_key_size: int = 2,
        key_gen_noise_std: float = 1.0,
        alpha_hop_weight: float = 0.0,
        invalid_action_penalty: float = -0.05,
        max_backlog: int = 512,
        fixed_requests_per_step: bool = True,
        backlog_drop_policy: str = 'drop_oldest',
        seed: int = 42,
    ):
        super().__init__()
        self.topology_conf = topology_conf
        self.N = int(self.topology_conf.get("NUM_QKD_NODE", 8))
        self.max_time_steps = int(max_time_steps)
        self.max_requests_per_step = int(max_requests_per_step)
        self.consume_key_size = int(consume_key_size)
        self.key_pool_size = int(key_pool_size)
        self.initial_key_lifetime = int(initial_key_lifetime)
        self.generate_key_size = int(generate_key_size)
        self.key_gen_noise_std = float(key_gen_noise_std)
        self.alpha_hop_weight = float(alpha_hop_weight)
        self.invalid_action_penalty = float(invalid_action_penalty)
        self.max_backlog = int(max_backlog)
        self.fixed_requests_per_step = bool(fixed_requests_per_step)
        self.backlog_drop_policy = str(backlog_drop_policy)

        self._rng = np.random.default_rng(seed)

        # Build topology
        self.G = self._build_topology(self.topology_conf)

        # Key pool: dict[(u,v)] -> list[int lifetimes]
        self.key_pool = {}
        for u, v in self.G.edges():
            e = self._ekey(u, v)
            self.key_pool[e] = []

        # Observation/Action spaces
        # num_key matrix in [0,1], shape (N,N)
        # requests (R_max, 4): src, dst, path_len_hops, path_min_keys (all normalized-ish floats)
        # mask (R_max,): bool
        # t: scalar in [0,1]
        self.observation_space = spaces.Dict(
            {
                "num_key": spaces.Box(low=0.0, high=1.0, shape=(self.N, self.N), dtype=np.float32),
                "requests": spaces.Box(low=0.0, high=1.0, shape=(self.max_requests_per_step, 4), dtype=np.float32),
                "mask": spaces.MultiBinary(self.max_requests_per_step),
                "t": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            }
        )
        self.action_space = spaces.Discrete(self.max_requests_per_step)

        # Runtime
        self.time_step = 0
        self.current_requests = []  # list[Request]
        self.total_success = 0
        self.total_blocking = 0

    # ------------------------- core gym API -------------------------

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Reset key pool
        for e in list(self.key_pool.keys()):
            self.key_pool[e] = []

        # Optionally pre-fill some keys
        self._warmup_keys()

        self.time_step = 0
        self.total_success = 0
        self.total_blocking = 0

        # First batch of requests
        self.current_requests = self._generate_requests(self.time_step)
        obs = self._build_observation()
        info = {"num_requests": len(self.current_requests)}
        return obs, info

    def step(self, action: int):
        terminated = False
        truncated = False
        reward = 0.0

        # 1) Apply lifetime decay & key generation for this step
        self._lifetime_and_generate_keys()

        # 2) Handle action (choose one request among current)
        valid_mask = self._request_mask()
        if action < 0 or action >= self.max_requests_per_step or not valid_mask[action]:
            # invalid action
            reward += self.invalid_action_penalty
        else:
            req = self.current_requests[action]
            success = self._serve_request(req)
            if success:
                reward += 1.0
                self.total_success += 1
            else:
                # No extra penalty; blocking metric increments
                self.total_blocking += 1

        # 3) Advance time and generate new batch
        self.time_step += 1
        if self.time_step >= self.max_time_steps:
            terminated = True
        else:
            self.current_requests = self._generate_requests(self.time_step)

        obs = self._build_observation()
        info = {
            "num_requests": len(self.current_requests),
            "success_total": self.total_success,
            "blocking_total": self.total_blocking,
        }
        return obs, reward, terminated, truncated, info

    # ------------------------- internals -------------------------

    def _build_topology(self, topo: dict):
        """
        Builds an undirected simple graph. If 'EDGES' present in topology_conf, use that.
        Otherwise create a random connected graph.
        """
        G = nx.Graph()
        G.add_nodes_from(range(self.N))

        if "EDGES" in topo:
            for (u, v) in topo["EDGES"]:
                if u == v: 
                    continue
                G.add_edge(int(u), int(v))
        else:
            # build a simple ring + some chords for connectivity
            for i in range(self.N):
                G.add_edge(i, (i + 1) % self.N)
            # add random chords
            m = max(1, self.N // 3)
            added = 0
            while added < m:
                u, v = self._rng.integers(0, self.N, size=2)
                if u != v and not G.has_edge(u, v):
                    G.add_edge(int(u), int(v))
                    added += 1
        return G

    def _ekey(self, u: int, v: int):
        a, b = (u, v) if u < v else (v, u)
        return (a, b)

    def _warmup_keys(self):
        # Initially seed each edge with a small number of keys
        for u, v in self.G.edges():
            e = self._ekey(u, v)
            init_k = self._rng.integers(low=0, high=min(8, self.key_pool_size))
            self.key_pool[e] = [self.initial_key_lifetime] * int(init_k)

    def _lifetime_and_generate_keys(self):
        # Age keys by -1; remove expired
        for e in list(self.key_pool.keys()):
            if not self.key_pool[e]:
                continue
            self.key_pool[e] = [k - 1 for k in self.key_pool[e] if (k - 1) > 0]

        # Generate keys with slight noise
        gen = int(np.round(self._rng.normal(self.generate_key_size, self.key_gen_noise_std)))
        gen = max(0, gen)
        if gen == 0:
            return

        for u, v in self.G.edges():
            e = self._ekey(u, v)
            space = self.key_pool_size - len(self.key_pool[e])
            if space <= 0:
                continue
            to_add = min(space, gen)
            self.key_pool[e].extend([self.initial_key_lifetime] * to_add)

    def _remaining_keys_matrix(self):
        M = np.zeros((self.N, self.N), dtype=np.float32)
        for u, v in self.G.edges():
            e = self._ekey(u, v)
            cnt = len(self.key_pool[e])
            M[u, v] = cnt
            M[v, u] = cnt
        # normalize to [0,1]
        mx = M.max()
        if mx > 0:
            M = M / mx
        return M

    def _weighted_shortest_path(self, src: int, dst: int):
        """
        Weight = 1/(remaining_keys + eps) + alpha*1 (hop-cost per edge)
        Returns: list of nodes path or None
        """
        eps = 1e-6
        def edge_weight(u, v, _attr):
            e = self._ekey(u, v)
            rk = len(self.key_pool[e])
            return 1.0 / (rk + eps) + self.alpha_hop_weight * 1.0

        try:
            path = nx.shortest_path(self.G, source=src, target=dst, weight=edge_weight, method="dijkstra")
            return path
        except nx.NetworkXNoPath:
            return None

    def _path_min_keys(self, path):
        if not path or len(path) < 2:
            return 0
        mins = []
        for i in range(len(path) - 1):
            e = self._ekey(path[i], path[i + 1])
            mins.append(len(self.key_pool[e]))
        return min(mins) if mins else 0

    def _serve_request(self, req: Request):
        path = self._weighted_shortest_path(req.src, req.dst)
        if path is None:
            return False
        # Check capacity
        for i in range(len(path) - 1):
            e = self._ekey(path[i], path[i + 1])
            if len(self.key_pool[e]) < self.consume_key_size:
                return False
        # Consume
        for i in range(len(path) - 1):
            e = self._ekey(path[i], path[i + 1])
            # oldest-first
            del self.key_pool[e][: self.consume_key_size]
        return True

    
    def _generate_requests(self, t: int, limit: int | None = None):
            # Randomly sample up to max_requests_per_step unique pairs
            Rmax = self.max_requests_per_step if limit is None else min(limit, self.max_requests_per_step)
            iu, ju = np.triu_indices(self.N, k=1)
            # if fewer pairs than R_max (small graphs), sample with replacement
            replace = len(iu) < Rmax
            idx = self._rng.choice(len(iu), size=Rmax, replace=replace)
            reqs = []
            for ridx, k in enumerate(idx):
                src = int(iu[k])
                dst = int(ju[k])
                reqs.append(Request(src, dst, req_id=(t * 1000 + ridx), arrival_t=t))
            # Decide count
            if self.fixed_requests_per_step:
                n_valid = Rmax
            else:
                n_valid = int(self._rng.integers(low=1, high=min(Rmax, len(reqs)) + 1))
            return reqs[:n_valid]


    def _request_mask(self):
        mask = np.zeros((self.max_requests_per_step,), dtype=bool)
        n = len(self.current_requests)
        mask[:n] = True
        return mask

    def _requests_features(self):
        # Build (R_max, 4) features; pad invalid rows with zeros
        feats = np.zeros((self.max_requests_per_step, 4), dtype=np.float32)
        for i, req in enumerate(self.current_requests):
            path = self._weighted_shortest_path(req.src, req.dst)
            if path is None:
                hops = 0.0
                min_keys = 0.0
            else:
                hops = float(len(path) - 1)
                min_keys = float(self._path_min_keys(path))
            # Normalize hops by (N-1)
            hops_norm = hops / max(1.0, float(self.N - 1))
            # Normalize min_keys by key_pool_size
            min_keys_norm = min_keys / max(1.0, float(self.key_pool_size))
            # src,dst normalized by N-1
            feats[i, 0] = req.src / max(1.0, float(self.N - 1))
            feats[i, 1] = req.dst / max(1.0, float(self.N - 1))
            feats[i, 2] = hops_norm
            feats[i, 3] = min_keys_norm
        return feats

    def _build_observation(self):
        num_key = self._remaining_keys_matrix()
        req_feats = self._requests_features()
        mask = self._request_mask().astype(np.int8)  # MultiBinary expects {0,1}
        t_norm = np.array([self.time_step / max(1, self.max_time_steps - 1)], dtype=np.float32)
        return {
            "num_key": num_key.astype(np.float32),
            "requests": req_feats,
            "mask": mask,
            "t": t_norm,
        }

    # ------------------------- utils -------------------------

    def render(self):
        print(f"t={self.time_step}, success={self.total_success}, blocking={self.total_blocking}")

    def close(self):
        pass


if __name__ == "__main__":
    # Small smoke test
    topo = {"NUM_QKD_NODE": 6}
    env = QKDNSchedulingEnv(topo, max_time_steps=5, max_requests_per_step=5, seed=0)
    obs, info = env.reset()
    print("Reset obs keys:", list(obs.keys()), "info:", info)
    done = False
    total_reward = 0.0
    while True:
        # random valid action
        mask = obs["mask"].astype(bool)
        valid_idxs = np.where(mask)[0]
        # a = int(valid_idxs[0]) if len(valid_idxs) > 0 else 0
        a = int(random.choice(valid_idxs)) if len(valid_idxs) > 0 else 0
        obs, r, term, trunc, info = env.step(a)
        total_reward += r
        env.render()
        print("action: ", a)
        print(env.key_pool)
        print()
        if term or trunc:
            break
    print("Total reward:", total_reward)
    def _enforce_backlog_cap(self):
        """Ensure carryover_queue length <= max_backlog according to policy."""
        if self.max_backlog <= 0:
            # no backlog allowed
            self.dropped_backlog += len(self.carryover_queue)
            self.carryover_queue = []
            return

        excess = len(self.carryover_queue) - self.max_backlog
        if excess <= 0:
            return

        if self.backlog_drop_policy == 'drop_oldest':
            # drop from the front (oldest first)
            self.carryover_queue = self.carryover_queue[excess:]
            self.dropped_backlog += excess
        elif self.backlog_drop_policy == 'drop_random':
            # randomly keep max_backlog items
            keep_idx = set(self._rng.choice(len(self.carryover_queue), size=self.max_backlog, replace=False).tolist())
            self.carryover_queue = [r for i, r in enumerate(self.carryover_queue) if i in keep_idx]
            self.dropped_backlog += excess
        else:
            # default fallback: drop_oldest
            self.carryover_queue = self.carryover_queue[excess:]
            self.dropped_backlog += excess

