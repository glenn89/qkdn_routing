# eval_set_transformer_ppo.py
"""
QKDN 스케줄링 - Set Transformer PPO 평가 코드

- 수정본 학습 코드(Actor/Critic 분리, req_feat_dim = 5 + E)의 체크포인트를 로드해 평가
- 같은 env seed에서 PPO와 휴리스틱(random / fifo / min_hop / min_hop_feasible / edf_feasible)을 비교
- warm-up 에피소드를 제외하고 지표 집계 (env의 누적 카운터를 스냅샷 차분으로 계산)
- 정책별 CSV + 전체 비교표(summary.csv) 저장

예)
  python eval_set_transformer_ppo.py --ckpt ./checkpoints/grid6_setppo_final.pt \
      --N 6 --R_max 50 --max_time_step 50 --episodes 1000 --warmup 20 --seeds 101 202 303
"""
import os
import json
import time
import argparse
import importlib
from typing import Callable, Dict, List, Optional

import networkx as nx
import numpy as np
import pandas as pd
import torch

# ---------- 학습 코드 import (수정본: Actor/Critic 분리 버전) ----------
T = None
for _name in ("train_set_transformer_ppo_1", "train_transformer_ppo"):
    try:
        _m = importlib.import_module(_name)
    except ImportError:
        continue
    if hasattr(_m, "Actor"):          # 수정본에만 있는 클래스
        T = _m
        break
if T is None:
    raise ImportError("수정본 학습 코드(Actor/Critic 분리 버전)를 찾지 못했습니다. "
                      "train_transformer_ppo.py(또는 train_set_transformer_ppo.py)를 같은 폴더에 두세요.")


# ---------- robust checkpoint loader ----------
def _maybe_extract_state_dict(ckpt_obj):
    """지원 포맷: raw state_dict, {'model'|'state_dict'|'model_state_dict'|'policy'|'net': state_dict}"""
    if isinstance(ckpt_obj, dict):
        for k in ("model", "state_dict", "model_state_dict", "policy", "net"):
            v = ckpt_obj.get(k, None)
            if isinstance(v, dict) and v and all(isinstance(t, torch.Tensor) for t in v.values()):
                return v
        if ckpt_obj and all(isinstance(t, torch.Tensor) for t in ckpt_obj.values()):
            return ckpt_obj
    raise RuntimeError(
        f"Unsupported checkpoint format. Top-level keys: "
        f"{list(ckpt_obj.keys()) if isinstance(ckpt_obj, dict) else type(ckpt_obj)}")


_warned_cpu = False


def _resolve_device(device) -> torch.device:
    global _warned_cpu
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        if not _warned_cpu:
            print("[warn] CUDA 사용 불가 → CPU로 평가합니다.")
            _warned_cpu = True
        return torch.device("cpu")
    return device


def load_agent(checkpoint_path: str, env, device: torch.device):
    """env 설정(N, R_max, 요청 feature 차원)으로 네트워크를 만들고 가중치 로드"""
    agent = T.PPOAgent(env.max_requests_per_step, env.N, env.req_feat_dim,
                       T.PPOConfig(device=str(device)))
    state = _maybe_extract_state_dict(torch.load(checkpoint_path, map_location=device))
    try:
        agent.net.load_state_dict(state)
    except RuntimeError as e:
        raise RuntimeError(
            f"체크포인트와 네트워크 구조가 맞지 않습니다. 학습 때와 같은 N / R_max / "
            f"include_path_edges 설정인지 확인하세요. (현재 env: N={env.N}, "
            f"R_max={env.max_requests_per_step}, req_feat_dim={env.req_feat_dim})\n{e}") from e
    agent.net.eval()
    print(f"[load] {checkpoint_path}")
    return agent


# ---------- 정책들 (obs -> action) ----------
# requests feature: [src, dst, hops_norm, min_keys_norm, wait_left_ratio, (path edge one-hot ...)]
HOPS, MIN_KEYS, WAIT = 2, 3, 4


def make_ppo_policy(agent, device, greedy: bool = True) -> Callable:
    @torch.no_grad()
    def policy(obs):
        a, _, _ = agent.net.act(T.obs_to_tensors(obs, device), greedy=greedy)
        return int(a)
    return policy


def make_random_policy(rng: np.random.Generator) -> Callable:
    def policy(obs):
        valid = np.flatnonzero(np.asarray(obs["mask"]).astype(bool))
        return int(rng.choice(valid)) if valid.size else 0
    return policy


def policy_fifo(obs) -> int:
    """유효 요청 중 가장 앞 인덱스 (env는 carryover → fresh 순으로 채우므로 오래된 요청 우선)"""
    valid = np.flatnonzero(np.asarray(obs["mask"]).astype(bool))
    return int(valid[0]) if valid.size else 0


def policy_min_hop(obs) -> int:
    """유효 요청 중 경로 hop 수가 가장 짧은 요청 (동률이면 앞 인덱스)"""
    mask = np.asarray(obs["mask"]).astype(bool)
    if not mask.any():
        return 0
    return int(np.argmin(np.where(mask, obs["requests"][:, HOPS], np.inf)))


def policy_min_hop_feasible(obs) -> int:
    """지금 서비스 가능한(경로 최소 키 > 0) 요청 중 hop 최소. 없으면 min_hop으로 폴백.
    env가 feature 계산과 실제 서비스에 같은 최단경로를 쓰므로 min_keys>0 ⇔ 성공."""
    mask = np.asarray(obs["mask"]).astype(bool)
    feas = mask & (obs["requests"][:, MIN_KEYS] > 0)
    if not feas.any():
        return policy_min_hop(obs)
    return int(np.argmin(np.where(feas, obs["requests"][:, HOPS], np.inf)))


def policy_edf_feasible(obs) -> int:
    """서비스 가능한 요청 중 남은 대기 TTL이 가장 적은 요청(EDF), 동률이면 hop 짧은 쪽."""
    mask = np.asarray(obs["mask"]).astype(bool)
    req = obs["requests"]
    feas = mask & (req[:, MIN_KEYS] > 0)
    if not feas.any():
        return policy_min_hop(obs)
    score = np.where(feas, req[:, WAIT] * 10.0 + req[:, HOPS], np.inf)
    return int(np.argmin(score))


HEURISTICS = {
    "fifo": policy_fifo,
    "min_hop": policy_min_hop,
    "min_hop_feasible": policy_min_hop_feasible,
    "edf_feasible": policy_edf_feasible,
}


# ---------- env 누적 카운터 스냅샷 (warm-up 제외용) ----------
def _snapshot(env) -> dict:
    return {
        "dropped_wait_expired": env.dropped_wait_expired,
        "dropped_backlog": env.dropped_backlog,
        "expired_keys": env.expired_keys_total,
        "generated_keys": env.total_generation_keys,
        "consumed_keys": env.total_consumed_keys,
        "served": {k: dict(v) for k, v in env.served_requests.items()},
        "link_consumed": dict(env.key_pool_consume),
        "n_delays": len(env.queueing_delays),
        "n_ep_delays": len(env.delay_per_episode),
        "pair_n": {k: len(v) for k, v in env.pair_queueing_delays.items()},
        # 공정성 cohort 계산용: 평가 구간의 첫 에피소드 번호와, 스냅샷 시점에 이미 생성된 그 에피소드의 신규 요청
        "episode_idx": env.episode_idx,
        "fresh_at_snap": _count_pairs(r for r in env.current_requests if r.arrival_episode == env.episode_idx),
    }


def _count_pairs(reqs) -> dict:
    out = {}
    for r in reqs:
        out[(r.src, r.dst)] = out.get((r.src, r.dst), 0) + 1
    return out


# ---------- 공정성(fairness): 노드 쌍(src,dst)별 할당 개수와 Jain's Fairness Index ----------
def jain_index(x) -> float:
    """JFI = (Σx)² / (n·Σx²).  1 = 완전히 균등, 1/n = 한 쌍이 전부 독점. 모두 0이면 정의되지 않음(nan)."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0 or not np.any(x > 0):
        return float("nan")
    return float(x.sum() ** 2 / (x.size * (x ** 2).sum()))


def compute_pair_fairness(env, snap, served_log: List[tuple]) -> tuple:
    """warm-up 이후 구간에서 노드 쌍별 생성·할당(서비스) 개수와 공정성 지표를 계산.

    served_log: 평가 중 서비스된 요청의 (src, dst, arrival_episode) 기록
    개수 정의:
      - served        : 평가 구간 동안 쌍 i에 할당(서비스)된 요청 수 (warm-up 중 도착한 요청 포함)
      - generated     : 평가 구간에 도착한 쌍 i의 요청 수
      - served_cohort : 그중 서비스된 수 (도착 에피소드 기준으로 집계 → service_ratio ≤ 1 보장)
    두 가지 JFI:
      - jfi_served_count : x_i = served. 도착이 쌍마다 다르면 수요 차이까지 불공정으로 잡힘.
      - jfi_service_ratio: x_i = served_cohort / generated (수요로 정규화).
                           정책이 특정 쌍(예: 먼 쌍)을 차별하는지를 봄.
    ※ 끝부분에 도착해 아직 대기 중인 요청은 서비스되지 않은 것으로 잡혀 ratio가 약간 낮게 나옴(긴 평가에선 무시 가능).
    """
    t0_ep = snap["episode_idx"]
    window, cohort = {}, {}
    for s, t, a in served_log:
        window[(s, t)] = window.get((s, t), 0) + 1
        if a >= t0_ep:
            cohort[(s, t)] = cohort.get((s, t), 0) + 1
    rows = []
    for (s, t), v in env.served_requests.items():
        gen = v["generated"] - snap["served"][(s, t)]["generated"] + snap["fresh_at_snap"].get((s, t), 0)
        srv = window.get((s, t), 0)
        srv_c = cohort.get((s, t), 0)
        ds = env.pair_queueing_delays.get((s, t), [])[snap["pair_n"].get((s, t), 0):]
        try:
            hops = nx.shortest_path_length(env.G, s, t)        # 토폴로지상 최소 hop 거리
        except nx.NetworkXNoPath:
            hops = -1
        rows.append({
            "src": s, "dst": t, "hops": hops,
            "generated": int(gen), "served": int(srv), "served_cohort": int(srv_c),
            "service_ratio": srv_c / gen if gen > 0 else np.nan,
            "avg_delay": float(np.mean(ds)) if ds else np.nan,
            "max_delay": float(np.max(ds)) if ds else np.nan,
        })
    df = pd.DataFrame(rows)

    active = df[df["generated"] > 0]                      # 구간 내 요청이 한 번이라도 생성된 쌍
    ratios = active["service_ratio"].to_numpy()

    # hop 거리별 서비스율 (먼 쌍이 차별받는지 확인)
    by_hop = (df[df["hops"] >= 0].groupby("hops")[["generated", "served", "served_cohort"]].sum())
    by_hop["service_ratio"] = by_hop["served_cohort"] / by_hop["generated"].replace(0, np.nan)
    by_hop["n_pairs"] = df[df["hops"] >= 0].groupby("hops").size()

    stats = {
        "n_pairs": int(len(df)),
        "n_active_pairs": int(len(active)),
        "jfi_served_count": jain_index(df["served"].to_numpy()),
        "jfi_service_ratio": jain_index(ratios),
        "min_pair_service_ratio": float(np.nanmin(ratios)) if ratios.size else float("nan"),
        "max_pair_service_ratio": float(np.nanmax(ratios)) if ratios.size else float("nan"),
        "starved_pairs": int(((active["served_cohort"] == 0)).sum()),  # 요청은 있었지만 한 번도 할당 못 받은 쌍
        "service_ratio_by_hop": {int(h): float(r) for h, r in by_hop["service_ratio"].items()},
    }
    return df, by_hop.reset_index(), stats


def _new_ep_counter():
    return {"reward": 0.0, "steps": 0, "success": 0, "blocking": 0, "invalid": 0, "hops": 0.0}


# ---------- 단일 정책 평가 ----------
def evaluate_policy(
    policy_name: str,
    *,
    checkpoint_path: Optional[str] = None,
    episodes: int = 1000,
    warmup_episodes: int = 20,
    max_time_step: int = 50,
    R_max: int = 50,
    N: int = 28,
    seed: int = 0,
    device="cuda",
    ppo_sample: bool = False,
    out_dir: Optional[str] = None,
    tag: str = "COST266",
    verbose_every: int = 0,
    ppo_ckpts: Optional[Dict[str, str]] = None,
) -> dict:
    """policy_name: 'ppo' | 'ppo_05' | 'random' | 'fifo' | 'min_hop' | 'min_hop_feasible' | 'edf_feasible'
    ppo_ckpts: PPO 정책 이름 → 체크포인트 경로 (예: {"ppo": "...final.pt", "ppo_05": "..._05_final.pt"})"""
    device = _resolve_device(device)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = T.make_env(max_time_step=max_time_step, R_max=R_max, N=N, seed=seed)
    assert env.auto_continue, "auto_continue=True 환경을 가정합니다 (reset()이 누적 카운터를 초기화하므로)."

    # PPO 계열: 이름으로 체크포인트를 찾음 ("ppo"는 기존처럼 checkpoint_path도 허용)
    ppo_ckpts = dict(ppo_ckpts or {})
    if checkpoint_path and "ppo" not in ppo_ckpts:
        ppo_ckpts["ppo"] = checkpoint_path

    if policy_name in ppo_ckpts:
        ckpt = ppo_ckpts[policy_name]
        if not ckpt or not os.path.exists(ckpt):
            raise FileNotFoundError(f"checkpoint not found for {policy_name}: {ckpt}")
        policy = make_ppo_policy(load_agent(ckpt, env, device), device, greedy=not ppo_sample)
    elif policy_name == "random":
        policy = make_random_policy(np.random.default_rng(seed))
    elif policy_name in HEURISTICS:
        policy = HEURISTICS[policy_name]
    else:
        raise ValueError(f"unknown policy: {policy_name}")

    obs, info = env.reset()
    cur_ep = info["episode_idx"]
    snap = _snapshot(env) if warmup_episodes == 0 else None

    per_ep: List[dict] = []
    served_log: List[tuple] = []
    ep = _new_ep_counter()
    total_eps = warmup_episodes + episodes
    t0 = time.time()

    while len(per_ep) < total_eps:
        mask = np.asarray(obs["mask"]).astype(bool)
        a = policy(obs)
        valid_action = 0 <= a < mask.size and mask[a]
        chosen = env.current_requests[a] if valid_action else None   # 공정성 집계용: 선택된 요청

        obs, r, terminated, truncated, info = env.step(a)

        # 평가 구간(warm-up 이후)에 서비스된 요청의 (src, dst, 도착 에피소드) 기록
        if snap is not None and chosen is not None and info["path_length"] > 0:
            served_log.append((chosen.src, chosen.dst, chosen.arrival_episode))

        # step 단위 집계: path_length>0 ⇔ 서비스 성공 (env는 성공 시에만 경로 길이를 채움)
        ep["reward"] += float(r)
        ep["steps"] += 1
        if not valid_action:
            ep["invalid"] += 1
        elif info["path_length"] > 0:
            ep["success"] += 1
            ep["hops"] += info["path_length"]
        else:
            ep["blocking"] += 1

        # 에피소드 경계 (env의 success_total/blocking_total은 경계 step에서 이미 0으로 리셋되므로 직접 집계)
        if info["episode_idx"] != cur_ep:
            cur_ep = info["episode_idx"]
            ep["expired_keys"] = info["expired_keys_last_episode"]
            ep["backlog_after"] = len(env.carryover_queue)
            ep["avg_hops_served"] = ep["hops"] / ep["success"] if ep["success"] else 0.0
            per_ep.append(ep)
            ep = _new_ep_counter()

            n_done = len(per_ep)
            if n_done == warmup_episodes:
                snap = _snapshot(env)
            if verbose_every and n_done > warmup_episodes and (n_done - warmup_episodes) % verbose_every == 0:
                e = per_ep[-1]
                print(f"[{policy_name}|seed {seed}] episode {n_done - warmup_episodes}/{episodes} "
                      f"reward={e['reward']:.2f} success={e['success']} blocking={e['blocking']} "
                      f"invalid={e['invalid']} expired_keys={e['expired_keys']} backlog={e['backlog_after']}")

    eval_eps = per_ep[warmup_episodes:]
    df_ep = pd.DataFrame(eval_eps)
    df_ep.insert(0, "episode", range(len(eval_eps)))

    # ---- warm-up 이후 구간의 누적 지표 (스냅샷 차분) ----
    gen = sum(v["generated"] - snap["served"][k]["generated"] for k, v in env.served_requests.items())
    srv = sum(v["success"] - snap["served"][k]["success"] for k, v in env.served_requests.items())
    delays = np.asarray(env.queueing_delays[snap["n_delays"]:], dtype=np.float32)
    n_succ, n_block = int(df_ep["success"].sum()), int(df_ep["blocking"].sum())

    stats = {
        "policy": policy_name,
        "seed": seed,
        "episodes": episodes,
        "warmup_episodes": warmup_episodes,
        "avg_reward": float(df_ep["reward"].mean()),
        "std_reward": float(df_ep["reward"].std(ddof=0)),
        "avg_success_per_ep": float(df_ep["success"].mean()),
        "avg_blocking_per_ep": float(df_ep["blocking"].mean()),
        "invalid_actions": int(df_ep["invalid"].sum()),
        # 서비스 시도 대비 성공률 (선택한 요청이 실제로 서비스된 비율)
        "attempt_success_rate": n_succ / max(1, n_succ + n_block),
        # 새로 생성된 요청 대비 서비스된 비율 (구간 경계의 이월 요청 때문에 근사치)
        "request_success_rate": srv / max(1, gen),
        "generated_requests": int(gen),
        "served_requests": int(srv),
        "expired_requests": env.dropped_wait_expired - snap["dropped_wait_expired"],
        "dropped_backlog_requests": env.dropped_backlog - snap["dropped_backlog"],
        "pending_requests_at_end": len(env.carryover_queue) + len(env.current_requests),
        "generated_keys": int(env.total_generation_keys - snap["generated_keys"]),
        "consumed_keys": int(env.total_consumed_keys - snap["consumed_keys"]),
        "expired_keys": int(env.expired_keys_total - snap["expired_keys"]),
        "avg_hops_served": float(df_ep["hops"].sum() / max(1, n_succ)),
        "avg_queueing_delay": float(delays.mean()) if delays.size else 0.0,   # 단위: episode
        "p95_queueing_delay": float(np.percentile(delays, 95)) if delays.size else 0.0,
        "max_queueing_delay": float(delays.max()) if delays.size else 0.0,
        "eval_seconds": round(time.time() - t0, 1),
    }
    stats["key_utilization"] = stats["consumed_keys"] / max(1, stats["generated_keys"])

    # ---- 공정성 (노드 쌍별 할당 개수, JFI) ----
    df_pair, df_hop, fair = compute_pair_fairness(env, snap, served_log)
    stats.update(fair)

    # ---- CSV 저장 ----
    if out_dir:
        d = os.path.join(out_dir, policy_name, f"seed{seed}")
        os.makedirs(d, exist_ok=True)
        p = f"{tag}_{policy_name}"
        df_ep.to_csv(os.path.join(d, f"{p}_episodes.csv"), index=False)

        pd.DataFrame([[u, v, c - snap["link_consumed"].get((u, v), 0)]
                      for (u, v), c in env.key_pool_consume.items()],
                     columns=["src", "dst", "consumed_keys"]
                     ).to_csv(os.path.join(d, f"{p}_links_consumed_keys.csv"), index=False)

        # 노드 쌍별: hop 거리, 생성 수, 할당(서비스) 수, 서비스율, 대기시간
        df_pair.to_csv(os.path.join(d, f"{p}_served_requests.csv"), index=False)
        # hop 거리별 서비스율
        df_hop.to_csv(os.path.join(d, f"{p}_service_ratio_by_hop.csv"), index=False)

        pd.DataFrame({"episode": range(len(env.delay_per_episode) - snap["n_ep_delays"]),
                      "avg_delay": env.delay_per_episode[snap["n_ep_delays"]:]}
                     ).to_csv(os.path.join(d, f"{p}_queueing_delay_per_episode.csv"), index=False)

        rows = []
        for (s, t), ds in env.pair_queueing_delays.items():
            ds = ds[snap["pair_n"].get((s, t), 0):]
            if ds:
                rows.append([s, t, float(np.mean(ds)), float(np.max(ds)), len(ds)])
        pd.DataFrame(rows, columns=["src", "dst", "avg_delay", "max_delay", "count"]
                     ).to_csv(os.path.join(d, f"{p}_pair_queueing_delay.csv"), index=False)

        with open(os.path.join(d, f"{p}_summary.json"), "w") as f:
            json.dump(stats, f, indent=2)

    df_pair.insert(0, "seed", seed)
    df_pair.insert(0, "policy", policy_name)
    stats["_pair_table"] = df_pair          # compare_policies에서 정책·seed 통합 표로 모은 뒤 제거
    return stats


# ---------- 여러 정책 × 여러 seed 비교 ----------
SUMMARY_COLS = ["avg_reward", "avg_success_per_ep", "attempt_success_rate", "request_success_rate",
                "expired_requests", "avg_hops_served", "key_utilization", "expired_keys",
                "avg_queueing_delay", "p95_queueing_delay",
                "jfi_served_count", "jfi_service_ratio", "min_pair_service_ratio", "starved_pairs"]
DISPLAY_COLS = ["avg_success_per_ep", "request_success_rate", "attempt_success_rate", "expired_requests", "avg_hops_served"]
FAIR_DISPLAY_COLS = ["jfi_served_count", "jfi_service_ratio", "min_pair_service_ratio", "starved_pairs"]


def _fmt_table(agg: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    return pd.DataFrame({c: [f"{agg.loc[p, (c, 'mean')]:.3f} ± {np.nan_to_num(agg.loc[p, (c, 'std')]):.3f}"
                             for p in agg.index] for c in cols}, index=agg.index)


def compare_policies(policies: List[str], seeds: List[int], out_dir: Optional[str] = None, **kwargs) -> pd.DataFrame:
    rows, pair_tables = [], []
    for name in policies:
        for s in seeds:
            st = evaluate_policy(name, seed=s, out_dir=out_dir, **kwargs)
            pair_tables.append(st.pop("_pair_table"))
            rows.append(st)
            print(f"  done: {name:17s} seed={s}  avg_reward={st['avg_reward']:.3f}  success_rate={st['attempt_success_rate']:.3f}  "
                  f"JFI(ratio)={st['jfi_service_ratio']:.3f}  ({st['eval_seconds']}s)", flush=True)
    df = pd.DataFrame(rows)
    agg = df.groupby("policy", sort=False)[SUMMARY_COLS].agg(["mean", "std"])

    # 정책·seed 통합 노드 쌍 표 → 정책별로 seed를 합산한 쌍별 할당 개수/서비스율
    pairs_all = pd.concat(pair_tables, ignore_index=True)
    pair_sum = (pairs_all.groupby(["policy", "src", "dst", "hops"], sort=False)
                [["generated", "served", "served_cohort"]].sum().reset_index())
    pair_sum["service_ratio"] = pair_sum["served_cohort"] / pair_sum["generated"].replace(0, np.nan)
    # hop 거리별 서비스율 (정책 × hop)
    hop_tbl = pair_sum.groupby(["policy", "hops"], sort=False)[["generated", "served_cohort"]].sum()
    hop_tbl = (hop_tbl["served_cohort"] / hop_tbl["generated"].replace(0, np.nan)).unstack("hops")
    hop_tbl.columns = [f"hop{h}" for h in hop_tbl.columns]

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        df.drop(columns=["service_ratio_by_hop"]).to_csv(os.path.join(out_dir, "all_runs.csv"), index=False)
        agg.to_csv(os.path.join(out_dir, "summary.csv"))
        pairs_all.to_csv(os.path.join(out_dir, "pair_allocation_all_runs.csv"), index=False)
        pair_sum.to_csv(os.path.join(out_dir, "pair_allocation_by_policy.csv"), index=False)
        hop_tbl.to_csv(os.path.join(out_dir, "service_ratio_by_hop.csv"))

    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print("\n===== Evaluation Summary (mean ± std over seeds) =====")
        print(_fmt_table(agg, DISPLAY_COLS))
        print("\n===== Fairness over (src,dst) pairs (mean ± std over seeds) =====")
        print(_fmt_table(agg, FAIR_DISPLAY_COLS))
        print("\n===== Service ratio by hop distance (seeds pooled) =====")
        print(hop_tbl.round(3))
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="./checkpoints/NSFNET_setppo_final.pt", help="'ppo' 체크포인트")
    ap.add_argument("--ckpt_05", default="./checkpoints/NSFNET_setppo_05_final.pt", help="'ppo_05' 체크포인트")
    ap.add_argument("--ckpt_10", default="./checkpoints/NSFNET_setppo_10_final.pt", help="'ppo_10' 체크포인트")
    ap.add_argument("--policies", nargs="+",
                    default=["SetTransformer", "SetTransformer_05", "SetTransformer_10", "random", "fifo", "min_hop_feasible"])
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=20, help="집계에서 제외할 초기 에피소드 수")
    ap.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303, 404, 505],  # [101, 202, 303, 404, 505]
                    help="학습 seed와 다른 값을 쓰세요")
    ap.add_argument("--N", type=int, default=14)
    ap.add_argument("--R_max", type=int, default=60)
    ap.add_argument("--max_time_step", type=int, default=60)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ppo_sample", action="store_true", help="PPO를 argmax 대신 확률적으로 샘플링")
    ap.add_argument("--out_dir", default="results/NSFNET")
    ap.add_argument("--tag", default="NSFNET")
    ap.add_argument("--verbose_every", type=int, default=0)
    args = ap.parse_args()

    # PPO 정책 이름 → 체크포인트 (PPO 모델을 더 추가하려면 여기에 한 줄 + --ckpt_xx 인자 추가)
    ppo_ckpts = {
        "SetTransformer": args.ckpt,
        "SetTransformer_05": args.ckpt_05,
        "SetTransformer_10": args.ckpt_10,
    }

    policies = list(args.policies)
    for name, ckpt in ppo_ckpts.items():
        if name in policies and not os.path.exists(ckpt):
            print(f"[warn] checkpoint 없음 ({ckpt}) → {name}은(는) 건너뜁니다.")
            policies.remove(name)

    compare_policies(policies, args.seeds, out_dir=args.out_dir,
                     ppo_ckpts=ppo_ckpts, episodes=args.episodes, warmup_episodes=args.warmup,
                     max_time_step=args.max_time_step, R_max=args.R_max, N=args.N,
                     device=args.device, ppo_sample=args.ppo_sample, tag=args.tag,
                     verbose_every=args.verbose_every)


if __name__ == "__main__":
    main()
