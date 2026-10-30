import logging
import os
import math
import random
from dataclasses import dataclass
from typing import Dict, Tuple, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from qkdn_scheduling_env_1 import QKDNSchedulingEnv

# -----------------------------
# Utils
# -----------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def to_tensor(x, device):
    return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)

def obs_to_tensors(obs, device):
    return {k: to_tensor(v, device) for k, v in obs.items()}

def stack_obs(obs_list: List[Dict[str, np.ndarray]], device):
    keys = obs_list[0].keys()
    return {k: to_tensor(np.stack([o[k] for o in obs_list], axis=0), device) for k in keys}

# -----------------------------
# Attention & Set Transformer blocks
# -----------------------------

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, q, k, v, key_mask=None):
        # q: [B,Q,D], k/v: [B,K,D], key_mask: [B,K] (True=실제 원소) or None
        B, Q, D = q.shape
        K = k.shape[1]
        q = self.q_proj(q).view(B, Q, self.n_head, self.d_head).transpose(1, 2)
        k = self.k_proj(k).view(B, K, self.n_head, self.d_head).transpose(1, 2)
        v = self.v_proj(v).view(B, K, self.n_head, self.d_head).transpose(1, 2)

        attn_logits = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)        # [B,H,Q,K]
        if key_mask is not None:
            attn_logits = attn_logits.masked_fill(~key_mask[:, None, None, :], float("-inf"))
        attn = attn_logits.softmax(dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)   # 요청이 0개인 배치(전부 마스킹) 방어
        attn = self.drop(attn)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, Q, D)
        return self.o_proj(out)


class MAB(nn.Module):
    """Multihead Attention Block (Pre-LN): MAB(X, Y) = X + Attn(X, Y) + FFN"""
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_head, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, X, Y, key_mask=None):
        Yn = self.ln_kv(Y)
        X = X + self.attn(self.ln_q(X), Yn, Yn, key_mask)
        X = X + self.ffn(self.ln2(X))
        return X


class SAB(nn.Module):
    """Set Attention Block: SAB(X) = MAB(X, X), 패딩 원소는 key에서 제외"""
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        self.mab = MAB(d_model, n_head, dropout)

    def forward(self, X, key_mask=None):
        return self.mab(X, X, key_mask)


class PMA(nn.Module):
    """Pooling by Multihead Attention: 학습 가능한 seed k개가 집합을 요약 (순열 불변)"""
    def __init__(self, d_model: int, n_head: int, k: int = 1, dropout: float = 0.0):
        super().__init__()
        self.seed = nn.Parameter(torch.randn(1, k, d_model) * 0.02)
        self.mab = MAB(d_model, n_head, dropout)

    def forward(self, Z, key_mask=None):
        S = self.seed.expand(Z.size(0), -1, -1)
        return self.mab(S, Z, key_mask).flatten(1)   # [B, k*D]

# -----------------------------
# Encoders & Policy
# -----------------------------

class GlobalEncoder(nn.Module):
    """num_key(N×N) + t → 전역 벡터.
    고정 토폴로지(Grid6)에서는 링크별 키 수 자체가 중요하므로 공간 정보를 평균내지 않고 flatten MLP 사용.
    (토폴로지 일반화가 필요하면 GNN/edge-token 방식으로 교체 권장)"""
    def __init__(self, N: int, d_out: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(N * N + 1, d_out), nn.ReLU(),
            nn.Linear(d_out, d_out), nn.ReLU(),
        )

    def forward(self, num_key, t):          # [B,N,N], [B,1]
        return self.mlp(torch.cat([num_key.flatten(1), t], dim=-1))


class RequestSetEncoder(nn.Module):
    """(R, F) → row MLP → SAB×2(마스크 적용) → final LN → (R, d)"""
    def __init__(self, f_in: int, d_req: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        self.row_mlp = nn.Sequential(
            nn.Linear(f_in, d_req), nn.ReLU(),
            nn.Linear(d_req, d_req),
        )
        self.sab1 = SAB(d_req, n_head, dropout)
        self.sab2 = SAB(d_req, n_head, dropout)
        self.ln_out = nn.LayerNorm(d_req)

    def forward(self, requests, key_mask):
        X = self.row_mlp(requests)
        X = self.sab1(X, key_mask)
        X = self.sab2(X, key_mask)
        return self.ln_out(X)


class Actor(nn.Module):
    """요청별 logit (순열 등변)"""
    def __init__(self, N, f_req, d_global, d_req, n_head, dropout):
        super().__init__()
        self.global_enc = GlobalEncoder(N, d_global)
        self.req_enc = RequestSetEncoder(f_req, d_req, n_head, dropout)
        self.head = nn.Sequential(nn.Linear(d_global + d_req, 128), nn.ReLU(), nn.Linear(128, 1))
        nn.init.orthogonal_(self.head[-1].weight, gain=0.01)   # 초기 정책 ≈ 균등
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, num_key, t, requests, mask):
        g = self.global_enc(num_key, t)
        Z = self.req_enc(requests, mask)
        R = requests.shape[1]
        return self.head(torch.cat([g.unsqueeze(1).expand(-1, R, -1), Z], -1)).squeeze(-1)


class Critic(nn.Module):
    """상태 가치 (순열 불변, PMA 풀링)"""
    def __init__(self, N, f_req, d_global, d_req, n_head, dropout):
        super().__init__()
        self.global_enc = GlobalEncoder(N, d_global)
        self.req_enc = RequestSetEncoder(f_req, d_req, n_head, dropout)
        self.pma = PMA(d_req, n_head, k=1, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(d_global + d_req, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, num_key, t, requests, mask):
        g = self.global_enc(num_key, t)
        Z = self.req_enc(requests, mask)
        return self.head(torch.cat([g, self.pma(Z, mask)], -1)).squeeze(-1)


class SetTransformerPolicy(nn.Module):
    """[수정] Actor/Critic 인코더 분리.
    공유 인코더에서는 value loss gradient가 policy gradient보다 수백 배 커서
    전체 norm 기준 grad clipping 후 정책이 거의 갱신되지 않았음."""
    def __init__(self, R_max: int, N: int, f_req: int,
                 d_global: int = 128, d_req: int = 128, n_head: int = 4, dropout: float = 0.0):
        super().__init__()
        self.R_max = R_max
        self.actor = Actor(N, f_req, d_global, d_req, n_head, dropout)
        self.critic = Critic(N, f_req, d_global, d_req, n_head, dropout)

    @staticmethod
    def _ensure_batch(obs):
        if obs["requests"].dim() == 2:
            return {k: v.unsqueeze(0) for k, v in obs.items()}, True
        return obs, False

    def forward(self, obs: Dict[str, torch.Tensor]):
        """returns masked_logits [B,R], value [B]"""
        num_key, requests = obs["num_key"], obs["requests"]
        mask = obs["mask"].to(torch.bool)                 # env의 mask = 실제 요청 슬롯(패딩 마스크)
        t = obs["t"].view(-1, 1)
        R = requests.shape[1]
        assert R == self.R_max

        logits = self.actor(num_key, t, requests, mask)   # [B,R]
        value = self.critic(num_key, t, requests, mask)   # [B]

        # 행동 마스킹 (전부 무효인 배치는 action 0 강제)
        all_invalid = (~mask).all(dim=-1, keepdim=True)
        act_mask = mask | (all_invalid & (torch.arange(R, device=mask.device) == 0))
        masked_logits = torch.where(act_mask, logits, torch.full_like(logits, -1e9))
        return masked_logits, value

    @torch.no_grad()
    def act(self, obs: Dict[str, torch.Tensor], greedy: bool = False):
        obs_b, added = self._ensure_batch(obs)
        logits, value = self.forward(obs_b)
        dist = torch.distributions.Categorical(logits=logits)
        action = logits.argmax(-1) if greedy else dist.sample()
        logp = dist.log_prob(action)
        if added:
            return action[0], logp[0], value[0]
        return action, logp, value

# -----------------------------
# PPO
# -----------------------------

@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    update_epochs: int = 2
    minibatch_size: int = 256
    target_kl: float = 0.02     # approx KL이 넘으면 이번 update의 남은 epoch 중단
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class PPOAgent:
    def __init__(self, R_max: int, N: int, f_req: int, cfg: PPOConfig = PPOConfig()):
        self.cfg = cfg
        self.net = SetTransformerPolicy(R_max, N, f_req).to(cfg.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr, eps=1e-5)

    def select_action(self, obs_dict):
        a, logp, v = self.net.act(obs_to_tensors(obs_dict, self.cfg.device))
        return int(a.item()), float(logp.item()), float(v.item())

    def ppo_update(self, obs_t, act_t, old_logp_t, ret_t, adv_t):
        cfg = self.cfg
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        n = adv_t.size(0)
        stats = {"pg_loss": [], "v_loss": [], "entropy": [], "approx_kl": [], "clipfrac": []}

        for _ in range(cfg.update_epochs):
            idx = torch.randperm(n, device=cfg.device)          # [수정] epoch마다 셔플
            epoch_kl = []
            for start in range(0, n, cfg.minibatch_size):
                mb = idx[start:start + cfg.minibatch_size]
                logits, value = self.net({k: v[mb] for k, v in obs_t.items()})
                dist = torch.distributions.Categorical(logits=logits)

                logp = dist.log_prob(act_t[mb])
                log_ratio = logp - old_logp_t[mb]
                ratio = log_ratio.exp()
                adv = adv_t[mb]

                pg_loss = -torch.min(ratio * adv,
                                     ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv).mean()
                v_loss = F.mse_loss(value, ret_t[mb])
                ent = dist.entropy().mean()
                loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * ent

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                # [수정] actor/critic 따로 clipping: value gradient가 policy gradient를 잠식하지 않도록
                nn.utils.clip_grad_norm_(self.net.actor.parameters(), cfg.max_grad_norm)
                nn.utils.clip_grad_norm_(self.net.critic.parameters(), cfg.max_grad_norm)
                self.opt.step()

                with torch.no_grad():
                    kl = ((ratio - 1) - log_ratio).mean().item()
                    stats["approx_kl"].append(kl)
                    stats["clipfrac"].append(((ratio - 1).abs() > cfg.clip_eps).float().mean().item())
                    stats["pg_loss"].append(pg_loss.item())
                    stats["v_loss"].append(v_loss.item())
                    stats["entropy"].append(ent.item())
                    epoch_kl.append(kl)
            if cfg.target_kl is not None and np.mean(epoch_kl) > cfg.target_kl:
                break                                            # [추가] KL 조기 종료
        return {k: float(np.mean(v)) for k, v in stats.items()}

# -----------------------------
# GAE
# -----------------------------

def compute_gae(rewards, values, dones, last_value, gamma, lam):
    """rewards/values/dones: [T], last_value: V(s_T). dones[t]=True면 t 이후 bootstrap 안 함."""
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    lastgaelam = 0.0
    for t in reversed(range(T)):
        nonterminal = 1.0 - float(dones[t])
        next_v = last_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_v * nonterminal - values[t]
        lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        adv[t] = lastgaelam
    return adv, adv + values

# -----------------------------
# Training Loop
# -----------------------------

@dataclass
class TrainConfig:
    seed: int = 0
    total_updates: int = 1_000
    rollout_steps: int = 2048         # [수정] 100 → 2048 (≈41 에피소드/업데이트)
    log_interval: int = 10
    save_interval: int = 100
    anneal_lr: bool = True
    save_dir: str = "./checkpoints"
    N: int = 28
    R_max: int = 50
    max_time_step: int = 50


def make_env(max_time_step=50, R_max=50, N=6, seed=0):
    return QKDNSchedulingEnv(
        {"NUM_QKD_NODE": N},
        max_time_steps=max_time_step,
        max_requests_per_step=R_max,
        auto_continue=True,
        request_wait_episodes=5,
        seed=seed,
        immediate_reward=True,
        include_path_edges=True,
    )


def train(tcfg: TrainConfig = TrainConfig()):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(tcfg.seed)

    env = make_env(tcfg.max_time_step, tcfg.R_max, tcfg.N, tcfg.seed)
    obs, info = env.reset()
    prev_ep_idx = info["episode_idx"]

    agent = PPOAgent(tcfg.R_max, tcfg.N, env.req_feat_dim, PPOConfig(device=device))
    os.makedirs(tcfg.save_dir, exist_ok=True)

    ep_return, ep_returns_hist, update_idx, stats_hist = 0.0, [], [], []
    global_step = 0

    for update in range(1, tcfg.total_updates + 1):
        if tcfg.anneal_lr:
            for pg in agent.opt.param_groups:
                pg["lr"] = agent.cfg.lr * (1.0 - (update - 1) / tcfg.total_updates)

        obs_buf, act_buf, logp_buf, val_buf, rew_buf, done_buf = [], [], [], [], [], []
        finished_returns = []

        for _ in range(tcfg.rollout_steps):
            obs_buf.append(obs)       # env가 매 step 새 배열을 만들어 반환하므로 복사 불필요
            a, logp, v = agent.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(a)

            crossed = info["episode_idx"] != prev_ep_idx
            prev_ep_idx = info["episode_idx"]
            done = bool(terminated or truncated or crossed)

            act_buf.append(a); logp_buf.append(logp); val_buf.append(v)
            rew_buf.append(float(reward)); done_buf.append(done)

            ep_return += reward
            if done:
                finished_returns.append(ep_return)
                ep_return = 0.0
            global_step += 1

        with torch.no_grad():
            _, last_v = agent.net(SetTransformerPolicy._ensure_batch(obs_to_tensors(obs, device))[0])
        adv_np, ret_np = compute_gae(np.array(rew_buf, np.float32), np.array(val_buf, np.float32),
                                     np.array(done_buf), float(last_v.item()),
                                     agent.cfg.gamma, agent.cfg.lam)

        stats = agent.ppo_update(
            stack_obs(obs_buf, device),
            torch.tensor(act_buf, dtype=torch.long, device=device),
            torch.tensor(logp_buf, dtype=torch.float32, device=device),
            torch.tensor(ret_np, dtype=torch.float32, device=device),
            torch.tensor(adv_np, dtype=torch.float32, device=device),
        )

        mean_ep_ret = float(np.mean(finished_returns)) if finished_returns else float("nan")
        ep_returns_hist.append(mean_ep_ret); update_idx.append(update); stats_hist.append(stats)

        if update % tcfg.log_interval == 0:
            print(f"[Update {update:04d}] steps={global_step} ep_return={mean_ep_ret:.2f} "
                  f"(n_ep={len(finished_returns)}) ent={stats['entropy']:.3f} "
                  f"kl={stats['approx_kl']:.4f} clipfrac={stats['clipfrac']:.3f} "
                  f"v_loss={stats['v_loss']:.3f}", flush=True)

        if update % tcfg.save_interval == 0:
            path = os.path.join(tcfg.save_dir, f"COST266_setppo_10_update{update:04d}.pt")
            torch.save({"model": agent.net.state_dict(), "optimizer": agent.opt.state_dict(),
                        "update": update, "global_step": global_step}, path)

    final = os.path.join(tcfg.save_dir, "COST266_setppo_10_final.pt")
    torch.save({"model": agent.net.state_dict(), "optimizer": agent.opt.state_dict(),
                "update": tcfg.total_updates, "global_step": global_step}, final)
    print(f"Training finished. Final checkpoint saved to {final}")

    logging.basicConfig(filename="COST266_reward.log", level=logging.INFO, format="%(asctime)s - %(message)s")
    for u, r, s in zip(update_idx, ep_returns_hist, stats_hist):
        logging.info({"update": u, "ep_return": r, **s})

    try:
        fig, ax = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
        ax[0].plot(update_idx, ep_returns_hist); ax[0].set_ylabel("Mean episode return")
        ax[1].plot(update_idx, [s["entropy"] for s in stats_hist]); ax[1].set_ylabel("Entropy")
        ax[2].plot(update_idx, [s["approx_kl"] for s in stats_hist]); ax[2].set_ylabel("Approx KL")
        ax[2].set_xlabel("Update")
        for a in ax: a.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(tcfg.save_dir, "COST266_training_10_curves.png"), dpi=150)
        plt.close(fig)
    except Exception as e:
        print("Plotting failed:", e)
    return agent


if __name__ == "__main__":
    train()
