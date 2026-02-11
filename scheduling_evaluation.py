# eval_qkdn_ppo.py
import os
import numpy as np
import pulp
import torch

from train_transformer_ppo import (
    make_env, PPOAgent, PPOConfig, SetTransformerPolicy
)

# ---------- robust checkpoint loader ----------
def _maybe_extract_state_dict(ckpt_obj):
    """
    지원 포맷:
      - raw state_dict (dict[str, Tensor])
      - {'model': state_dict}, {'state_dict': state_dict}, {'model_state_dict': state_dict}, {'policy': ...}, {'net': ...}
    """
    if isinstance(ckpt_obj, dict):
        for k in ("model", "state_dict", "model_state_dict", "policy", "net"):
            v = ckpt_obj.get(k, None)
            if isinstance(v, dict) and v and all(isinstance(t, torch.Tensor) for t in v.values()):
                return v
        # 이미 state_dict인 경우
        if ckpt_obj and all(isinstance(t, torch.Tensor) for t in ckpt_obj.values()):
            return ckpt_obj
    raise RuntimeError(
        f"Unsupported checkpoint format. Top-level keys: "
        f"{list(ckpt_obj.keys()) if isinstance(ckpt_obj, dict) else type(ckpt_obj)}"
    )

# ---------- obs -> torch (단일 샘플용) ----------
@torch.no_grad()
def _to_torch_obs(obs: dict, device: torch.device) -> dict:
    def t(x, dtype=torch.float32):
        return torch.as_tensor(x, dtype=dtype, device=device)
    num_key = t(obs["num_key"]).unsqueeze(0)           # [1,N,N]
    requests = t(obs["requests"]).unsqueeze(0)         # [1,R,5]
    mask = torch.as_tensor(obs["mask"], dtype=torch.bool, device=device).unsqueeze(0)  # [1,R]
    t_feat = t(obs["t"])
    if t_feat.ndim == 1:
        t_feat = t_feat.unsqueeze(0)                   # [1,1]
    elif t_feat.ndim == 2 and t_feat.size(1) != 1:
        t_feat = t_feat[:, :1]
    return {"num_key": num_key, "requests": requests, "mask": mask, "t": t_feat}

# ---------- 정책: 마스크 적용 argmax ----------
@torch.no_grad()
def _select_action_argmax(agent: PPOAgent, obs: dict, device: torch.device) -> int:
    obs_t = _to_torch_obs(obs, device)
    logits, _ = agent.net(obs_t)                       # [1,R], [1,1]
    mask = obs_t["mask"]                               # [1,R] (bool)
    very_neg = torch.finfo(logits.dtype).min
    masked_logits = torch.where(mask, logits, very_neg)
    a = int(torch.argmax(masked_logits, dim=-1)[0].item())
    return a

# (옵션) 무작위 정책 비교용
def _select_action_random(obs: dict) -> int:
    mask = obs["mask"].astype(bool)
    valid = np.where(mask)[0]
    return int(np.random.choice(valid)) if len(valid) else 0

def _select_action_fifo(obs: dict) -> int:
    """
    단순 FIFO: 현재 유효 요청 중 '가장 앞' 인덱스를 선택.
    유효 요청이 하나도 없으면 0으로 폴백.
    """
    mask = np.asarray(obs["mask"]).astype(bool)  # [R]
    valid = np.flatnonzero(mask)
    return int(valid[0]) if valid.size > 0 else 0

def _select_action_min_path(obs: dict) -> int:
    """
    단순 FIFO: 현재 유효 요청 중 '가장 앞' 인덱스를 선택.
    유효 요청이 하나도 없으면 0으로 폴백.
    """
    mask = np.asarray(obs["mask"]).astype(bool)  # [R]
    if not mask.any():
        return 0
    req = np.asarray(obs["requests"])
    if req.ndim < 2 or req.shape[1] < 3:
        return int(np.flatnonzero(mask)[0])
    # invalid는 +inf로 가려서 argmin 대상 제외, 동률이면 가장 앞 인덱스 선택
    scored = np.where(mask, req[:, 2], np.inf)
    return int(np.argmin(scored))

def _select_action_ilp(obs: dict) -> int:
    """
    1-step ILP(=MILP) oracle selector.
    - Decision: choose at most one request among valid (mask==True)
    - Feasibility proxy: min_keys_norm > 0  (from env: path bottleneck keys / key_pool_size)
    - Objective: prioritize feasible + higher min_keys_norm + shorter hops + urgent TTL

    Requires: pip install pulp
    Uses PuLP's CBC solver if available; otherwise falls back to greedy.
    """
    mask = np.asarray(obs["mask"]).astype(bool)  # [R]
    valid = np.flatnonzero(mask)
    if valid.size == 0:
        return 0

    req = np.asarray(obs["requests"], dtype=np.float32)  # [R,5] per env
    # columns per env:
    # 0: src_norm, 1: dst_norm, 2: hops_norm, 3: min_keys_norm, 4: wait_left_norm
    if req.ndim != 2 or req.shape[1] < 5:
        # fallback: first valid
        return int(valid[0])

    hops = req[:, 2]
    min_keys = req[:, 3]
    wait = req[:, 4]

    # Feasibility proxy: min_keys_norm > 0
    feasible = (min_keys > 0.0) & mask
    feasible_idx = np.flatnonzero(feasible)

    # If nothing looks feasible, fall back to a safe heuristic (e.g., min-path among valid)
    if feasible_idx.size == 0:
        # Same as your min-path heuristic: argmin hops_norm among valid
        scored = np.where(mask, hops, np.inf)
        return int(np.argmin(scored))

    # --- MILP: choose <= 1 among feasible requests ---
    try:
        import pulp
    except Exception:
        # PuLP not installed -> greedy fallback
        # Score: prefer more min_keys, shorter hop, more urgent(wait high)
        score = 10.0 * min_keys - 1.0 * hops + 0.5 * wait
        score = np.where(feasible, score, -np.inf)
        return int(np.argmax(score))

    # Problem
    prob = pulp.LpProblem("QKDN_1step_select", pulp.LpMaximize)

    x = {}
    for i in feasible_idx:
        x[i] = pulp.LpVariable(f"x_{i}", lowBound=0, upBound=1, cat=pulp.LpBinary)

    # at most one request
    prob += pulp.lpSum([x[i] for i in feasible_idx]) <= 1

    # Objective:
    # Big reward for feasibility is already enforced by only allowing feasible_idx.
    # Now optimize tie-breakers:
    #   + min_keys_norm (prefer safer/bigger bottleneck)
    #   - hops_norm (prefer shorter)
    #   + wait_left_norm (prefer urgent)
    #
    # You can tune these weights.
    W_KEYS = 10.0
    W_HOPS = 1.0
    W_WAIT = 0.5

    prob += pulp.lpSum([
        x[i] * (W_KEYS * float(min_keys[i]) - W_HOPS * float(hops[i]) + W_WAIT * float(wait[i]))
        for i in feasible_idx
    ])

    # Solve (CBC if available)
    try:
        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=0.05)  # keep it fast per-step
        prob.solve(solver)
    except Exception:
        # solver failed -> greedy fallback
        score = 10.0 * min_keys - 1.0 * hops + 0.5 * wait
        score = np.where(feasible, score, -np.inf)
        return int(np.argmax(score))

    # Extract chosen action
    chosen = None
    best_val = -1

    for i in feasible_idx:
        v = int(round(pulp.value(x[i]) or 0))
        if v == 1:
            chosen = int(i)
            break
        # fallback if solver returns fractional / none:
        val = pulp.value(x[i])
        if val is not None and val > best_val:
            best_val = val
            chosen = int(i)

    if chosen is None:
        # final fallback
        scored = np.where(mask, hops, np.inf)
        return int(np.argmin(scored))

    return chosen



# ---------- 메인 평가 루틴 ----------
def evaluate_checkpoint(
    checkpoint_path: str,
    *,
    episodes: int = 10,
    max_time_step=5,
    R_max: int = 5,
    N: int = 14,
    seed: int = 0,
    device: str | torch.device = "cuda",
    use_random_policy: bool = False,
) -> dict:
    device = torch.device(device if (isinstance(device, str) and device == "cuda" and torch.cuda.is_available()) else "cpu")

    # Env & Agent 준비 (train과 동일한 생성자 경로 사용)
    env = make_env(max_time_step=max_time_step, R_max=R_max, N=N, seed=seed)
    agent = PPOAgent(R_max, PPOConfig(device=device))
    agent.net.eval().to(device)

    # 체크포인트 로드
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = _maybe_extract_state_dict(ckpt)
    agent.net.load_state_dict(state)
    print(f"[load] loaded weights from: {checkpoint_path}")

    # ------------ Eval loop ------------
    rewards = []
    obs, info = env.reset()
    cur_ep = info.get("episode_idx", getattr(env, "episode_idx", 0))

    ep_reward = 0.0
    ep_success = 0
    ep_blocking = 0
    if use_random_policy:
        print("RL agent evaluation")

    while len(rewards) < episodes:
        # 유효 액션만 선택 (invalid penalty 피하기)
        if use_random_policy and agent is not None:
            a = _select_action_argmax(agent, obs, device)
        else:
            # a = _select_action_random(obs)
            # a = _select_action_fifo(obs)
            # a = _select_action_min_path(obs)
            a = _select_action_ilp(obs)

        prev_ep = cur_ep
        # print(a)
        # print(obs['mask'])
        obs, r, terminated, truncated, info = env.step(a)
        cur_ep = info.get("episode_idx", getattr(env, "episode_idx", prev_ep))
        # print(a, info['path_length'])

        # 로컬 집계: r>0이면 성공, 아니면 블로킹으로 셈
        ep_reward += float(r)
        if r > 0:
            ep_success += 1
        else:
            ep_blocking += 1
        expired_key = info.get("expired_keys_last_episode")

        # 에피소드 경계 감지: episode_idx가 바뀌면 이전 에피소드 종료
        if cur_ep != prev_ep:
            ep_idx_print = len(rewards) + 1
            print(
                f"[eval] episode {ep_idx_print}/{episodes} "
                f"reward={ep_reward:.3f} success_total={ep_success} "
                f"blocking_total={ep_blocking} expired_key={expired_key}"
            )
            rewards.append(ep_reward)

            # 다음 에피소드용 초기화
            ep_reward = 0.0
            ep_success = 0
            ep_blocking = 0

    # 최종 통계
    rewards_np = np.array(rewards, dtype=np.float32)
    stats = {
        "episodes": float(episodes),
        "avg_reward": float(rewards_np.mean() if rewards_np.size else 0.0),
        "std_reward": float(rewards_np.std() if rewards_np.size > 1 else 0.0),
        "total_expired key": int(info.get("expired_keys_total")),
        "device": str(device),
    }
    print("===== Evaluation Summary =====")
    for k, v in stats.items():
        print(f"{k}: {v}")
    return stats


if __name__ == "__main__":
    # 예시 실행: 경로/파라미터를 프로젝트 설정에 맞게 바꾸세요
    ckpt = "./checkpoints/setppo_update99000.pt"
    if os.path.exists(ckpt):
        evaluate_checkpoint(
            ckpt,
            episodes=100,
            max_time_step=10,
            R_max=10,
            N=14,
            seed=0,
            device="cuda",
            use_random_policy=True,  # 랜덤 정책 비교시 False
        )
    else:
        print("No checkpoint found at", ckpt)