import random

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx
import topology_conf


class Request:
    """
    Represents a single traffic request.
    Episode-based waiting is tracked per-request:
      - wait_left: how many episode boundaries this request can survive.
      - max_wait: original waiting budget (for logging/analytics).
    """
    def __init__(self, src: int, dst: int, req_id: int, arrival_t: int, wait_left: int):
        self.src = int(src)
        self.dst = int(dst)
        self.id = int(req_id)
        self.arrival_t = int(arrival_t)
        self.wait_left = int(wait_left)
        self.max_wait = int(wait_left)

    def dec_episode_wait(self) -> bool:
        """
        Decrease remaining episode-based waiting budget by 1.
        Returns True if the request may still carry over (>= 0), else False.
        """
        self.wait_left -= 1
        return self.wait_left >= 0

    def remaining_wait(self) -> int:
        return self.wait_left

    def as_tuple(self):
        return (self.src, self.dst, self.id, self.arrival_t, self.wait_left, self.max_wait)

    def __repr__(self):
        return f"Request(src={self.src}, dst={self.dst}, id={self.id}, t={self.arrival_t}, wait_left={self.wait_left})"


class QKDNSchedulingEnv(gym.Env):
    """
    QKDN Scheduling Environment (Gymnasium-style)

    Episode dynamics:
      - At the END of each episode:
          (1) lifetime of remaining keys decreases by 1 and expired keys are removed,
          (2) new keys are generated on each edge (to be used next episode),
          (3) unserved requests' waiting TTL decreases by 1 via Request.dec_episode_wait().
              Only requests that return True (still have wait budget) carry over.
      - Keys are NOT aged or generated during steps.

    Step dynamics:
      - Each time step, exactly R_max requests (configurable) are presented.
      - The agent selects exactly one of them (action = index in [0, R_max-1]).
      - The selected request is served via a weighted shortest path (WSP).
        Edge weight = 1 / (remaining_keys + epsilon) + alpha * hop_cost.
      - If a feasible path exists AND every edge on the path has at least ONE key:
          success (+1 reward) and we decrement the key count by exactly ONE per path edge.
        Else:
          failure (+0 reward by default, optional penalty).

    Observation Dict:
      num_key: (N,N) float32 normalized [0..1] remaining-keys matrix
      requests: (R_max, 4) float32 = [src_norm, dst_norm, path_len_norm, path_min_keys_norm]
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
        max_requests_per_step: int = 10,
        consume_key_size: int = 1,
        key_pool_size: int = 64,
        initial_key_lifetime: int = 2,
        generate_key_size: int = 1,
        key_gen_noise_std: float = 1.0,

        alpha_hop_weight: float = 0.0,
        invalid_action_penalty: float = -0.05,
        max_backlog: int = 512,
        fixed_requests_per_step: bool = True,
        backlog_drop_policy: str = 'drop_oldest',
        auto_continue: bool = True,
        request_wait_episodes: int = 3,
        seed: int = 42,
    ):
        super().__init__()
        self.topology_conf = topology_conf
        self.N = int(self.topology_conf.get("NUM_QKD_NODE", 8))
        self.max_time_steps = int(max_time_steps)
        self.max_requests_per_step = int(max_requests_per_step)
        # Keep parameter but enforce ONE key per edge consumption in _serve_request:
        self.consume_key_size = int(consume_key_size)
        self.key_pool_size = int(key_pool_size)
        self.initial_key_lifetime = int(initial_key_lifetime)
        self.generate_key_size = int(generate_key_size)
        self.key_gen_noise_std = float(key_gen_noise_std)
        self.expired_keys_last = 0
        self.expired_keys_total = 0
        self.alpha_hop_weight = float(alpha_hop_weight)
        self.invalid_action_penalty = float(invalid_action_penalty)
        self.pending_reward = 0.0
        self.decision_in_slot = 0

        # Scheduling / backlog
        self.max_backlog = int(max_backlog)
        self.fixed_requests_per_step = bool(fixed_requests_per_step)
        self.backlog_drop_policy = str(backlog_drop_policy)
        self.auto_continue = bool(auto_continue)

        # Request TTL in episodes
        self.request_wait_episodes = int(request_wait_episodes)

        self._rng = np.random.default_rng(seed)

        # Build topology
        self.G = self._build_topology(self.topology_conf)

        # Key pool: dict[(u,v)] -> list[int lifetimes]
        self.key_pool = {}
        for u, v in self.G.edges():
            e = self._ekey(u, v)
            self.key_pool[e] = []

        # Observation/Action spaces
        self.observation_space = spaces.Dict(
            {
                "num_key": spaces.Box(low=0.0, high=1.0, shape=(self.N, self.N), dtype=np.float32),
                "requests": spaces.Box(low=0.0, high=1.0, shape=(self.max_requests_per_step, 5), dtype=np.float32),
                "mask": spaces.MultiBinary(self.max_requests_per_step),
                "t": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            }
        )
        self.action_space = spaces.Discrete(self.max_requests_per_step)

        # Runtime
        self.time_step = 0
        self.episode_idx = 0
        self._is_first_reset = True
        self.current_requests = []  # list[Request]
        self.total_success = 0
        self.total_blocking = 0

        # Queues
        self.temp_queue = []          # accumulates within an episode
        self.carryover_queue = []     # persists across episodes
        self.dropped_backlog = 0      # stats: dropped due to backlog cap
        self.dropped_wait_expired = 0 # stats: dropped due to wait TTL expiration

    # ------------------------- core gym API -------------------------

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Keys persist across episodes; warmup only once (first reset)
        if self._is_first_reset:
            for e in list(self.key_pool.keys()):
                self.key_pool[e] = []
            self._warmup_keys()
            self._is_first_reset = False

        # Reset episode runtime stats
        self.time_step = 0
        self.total_success = 0
        self.total_blocking = 0
        self.pending_reward = 0.0
        self.decision_in_slot = 0
        self.temp_queue = []  # new episode starts with empty temp queue
        self.expired_keys_last = 0

        # Prepare initial requests from carryover first, then fill with fresh generation
        self._start_next_episode()  # uses carryover first then fresh

        obs = self._build_observation()
        info = {"num_requests": len(self.current_requests), "episode_idx": self.episode_idx}
        return obs, info

    def step(self, action: int):
        terminated = False
        truncated = False
        reward = 0.0

        # NOTE: No lifetime aging or key generation here (episode boundary only).

        # 1) Handle action (choose one request among current)
        valid_mask = self._request_mask()
        chosen_idx = None
        served_success = False
        path_len = 0.0

        if action < 0 or action >= self.max_requests_per_step or not valid_mask[action]:
            # invalid action
            reward += self.invalid_action_penalty
            path_len = 0.0
        else:
            chosen_idx = int(action)
            req = self.current_requests[chosen_idx]
            served_success, path = self._serve_request(req)  # decrements ONE key per edge on success
            if served_success:
                self.pending_reward += 1.0 - (0.0 * (req.wait_left / self.request_wait_episodes))
                self.total_success += 1
                del self.current_requests[chosen_idx]
                path_len = float(len(path) - 1)
            else:
                reward += 0.0
                self.total_blocking += 1
                path_len = 0.0

        # 2) Advance time and generate next batch of candidates (fresh only)
        self.time_step += 1
        self.decision_in_slot += 1
        if self.decision_in_slot >= self.max_time_steps:
            if self.current_requests:
                self.temp_queue.extend(self.current_requests)
                self.current_requests = []
                # Episode end: age remaining keys and then generate new keys for the next episode

            reward = self.pending_reward
            self.pending_reward = 0.0

            if self.time_step >= self.max_time_steps:
                self._end_of_episode_housekeeping()

                if self.auto_continue:
                    # Immediately start next episode; do not set terminated
                    self._start_next_episode()
                    terminated = False
                else:
                    terminated = True
        else:
            pass

        obs = self._build_observation()
        info = {
            "num_requests": len(self.current_requests),
            "success_total": self.total_success,
            "blocking_total": self.total_blocking,
            "episode_idx": self.episode_idx,
            "dropped_wait_expired": self.dropped_wait_expired,
            "path_length": path_len,
            "expired_keys_last_episode": self.expired_keys_last,
            "expired_keys_total": self.expired_keys_total,
        }
        return obs, reward, terminated, truncated, info

    # ------------------------- internals -------------------------

    def _build_topology(self, topo: dict):
        """
        Builds an undirected simple graph. If 'EDGES' present in topology_conf, use that.
        Otherwise create a random connected graph.
        """
        if self.N == 14:
            G = nx.from_numpy_array(np.array(topology_conf.nsfnet_topo["QKD_TOPOLOGY"]))
        elif self.N == 28:
            G = nx.from_numpy_array(np.array(topology_conf.cost266_topo["QKD_TOPOLOGY"]))
        else:
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

    def _generate_keys_episode(self):
        """Generate new keys for the *next* episode."""
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

    def _end_of_episode_housekeeping(self):
        """
        Called when an episode terminates.
        - Decrease lifetime of all remaining keys by 1 and remove expired.
        - Generate new keys for the next episode.
        - Decrease waiting TTL for unserved requests; keep only those with wait_left >= 0.
        - Move surviving to carryover_queue; clear temp_queue.
        - Enforce backlog cap.
        - Increment episode counter.
        """
        # Lifetime aging (-1) & expiry
        expired_this = 0  # ★ 추가: 이번 에피소드에서 만료된 키 수
        for e in list(self.key_pool.keys()):
            L = self.key_pool[e]
            if not L:
                continue
            aged = [k - 1 for k in L]  # 수명 -1
            keep = [k for k in aged if k > 0]
            expired_this += (len(aged) - len(keep))
            self.key_pool[e] = keep

        self.expired_keys_last = expired_this
        self.expired_keys_total += expired_this

        # Generate keys per episode
        self._generate_keys_episode()

        # Process unserved requests: decrease TTL and decide carryover
        if self.temp_queue:
            survivors = []
            for req in self.temp_queue:
                if req.dec_episode_wait():
                    survivors.append(req)
                else:
                    self.dropped_wait_expired += 1
            self.carryover_queue.extend(survivors)
            self.temp_queue = []

        # Enforce backlog cap
        self._enforce_backlog_cap()

        self.episode_idx += 1

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

    def _start_next_episode(self):
        """Start a new episode WITHOUT calling reset().
        - time_step -> 0
        - per-episode stats reset
        - current_requests = carryover first, then fresh requests to fill R_max
        """
        self.time_step = 0
        self.total_success = 0
        self.total_blocking = 0
        self.decision_in_slot = 0
        # temp_queue should already be cleared in housekeeping; keep safety clear
        self.temp_queue = []

        carry = []
        while self.carryover_queue and len(carry) < self.max_requests_per_step:
            carry.append(self.carryover_queue.pop(0))

        need = self.max_requests_per_step - len(carry)
        fresh = self._generate_requests(self.time_step, limit=need) if need > 0 else []
        self.current_requests = carry + fresh

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
            return False, path
        # Check capacity: require at least ONE key per edge
        for i in range(len(path) - 1):
            e = self._ekey(path[i], path[i + 1])
            if len(self.key_pool[e]) < 1:
                return False, path
        # Consume exactly ONE key per edge (oldest-first)
        for i in range(len(path) - 1):
            e = self._ekey(path[i], path[i + 1])
            if self.key_pool[e]:
                del self.key_pool[e][0]
        return True, path

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
            # Assign episode TTL
            reqs.append(Request(src, dst, req_id=(t * 1000 + ridx), arrival_t=t, wait_left=self.request_wait_episodes))
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
        # Build (R_max, 5) features; pad invalid rows with zeros
        feats = np.zeros((self.max_requests_per_step, 5), dtype=np.float32)
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
            feats[i, 4] = req.wait_left / int(max(1, req.max_wait))
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
    # Small smoke test (may require gymnasium to be installed)
    topo = {"NUM_QKD_NODE": 6}
    env = QKDNSchedulingEnv(topo, max_time_steps=10, max_requests_per_step=10, seed=0,
                            auto_continue=True, request_wait_episodes=2)
    obs, info = env.reset()
    print("Reset obs keys:", list(obs.keys()), "info:", info)
    total_reward = 0.0
    target_episodes = 5
    start_ep = env.episode_idx
    while True:
        # random valid action
        mask = obs["mask"].astype(bool)
        valid_idxs = np.where(mask)[0]
        a = int(random.choice(valid_idxs)) if len(valid_idxs) > 0 else 0
        obs, r, term, trunc, info = env.step(a)
        total_reward += r
        env.render()
        print("action:", a, "| dropped_wait_expired:", info.get("dropped_wait_expired"))
        print("reward:", r)
        print("current requests: ", len(env.current_requests))
        print()
        if env.episode_idx - start_ep >= target_episodes:
            break
    print("Episodes run:", env.episode_idx - start_ep, "Total reward:", total_reward)
