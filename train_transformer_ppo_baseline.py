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
# Standard Transformer encoder (Set Transformer 비교용 baseline)
# -----------------------------
# Set Transformer 버전과의 차이 (나머지 구조·크기·하이퍼파라미터는 동일하게 유지):
#   1) 위치 인코딩 사용: 요청 슬롯 순서(index)를 입력에 더함 → 순열 등변성이 깨짐 (표준 Transformer의 특징)
#   2) 인코더: SAB 대신 표준 nn.TransformerEncoderLayer (Pre-LN, GELU, FFN 4×d, 2층, 4 head)
#   3) Value 풀링: PMA 대신 [CLS] 토큰 (BERT 방식)
# 공통: 패딩 슬롯은 key_padding_mask로 attention에서 제외, 전역 인코더·출력 head·초기화 동일

class GlobalEncoder(nn.Module):
    """num_key(N×N) + t → 전역 벡터 (Set Transformer 버전과 동일)"""
    def __init__(self, N: int, d_out: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(N * N + 1, d_out), nn.ReLU(),
            nn.Linear(d_out, d_out), nn.ReLU(),
        )

    def forward(self, num_key, t):          # [B,N,N], [B,1]
        return self.mlp(torch.cat([num_key.flatten(1), t], dim=-1))


class SinusoidalPositionalEncoding(nn.Module):
    """Vaswani et al. (2017)의 고정 sinusoidal 위치 인코딩"""
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, L: int):
        return self.pe[:L]


class RequestTransformerEncoder(nn.Module):
    """(R, F) → row MLP → + 위치 인코딩 → TransformerEncoder ×2 (패딩 마스크) → final LN → (R, d)
    use_cls=True면 맨 앞에 [CLS] 토큰을 붙이고 (cls_out [B,d], tokens [B,R,d])를 반환."""
    def __init__(self, f_in: int, d_req: int, n_head: int, R_max: int, dropout: float = 0.0,
                 n_layers: int = 2, pos_encoding: str = "learned", use_cls: bool = False):
        super().__init__()
        assert pos_encoding in ("learned", "sinusoidal", "none")
        self.row_mlp = nn.Sequential(
            nn.Linear(f_in, d_req), nn.ReLU(),
            nn.Linear(d_req, d_req),
        )
        self.pos_encoding = pos_encoding
        if pos_encoding == "learned":
            self.pos_emb = nn.Embedding(R_max, d_req)
            nn.init.normal_(self.pos_emb.weight, std=0.02)
        elif pos_encoding == "sinusoidal":
            self.pos_emb = SinusoidalPositionalEncoding(d_req, R_max)

        self.use_cls = use_cls
        if use_cls:
            self.cls = nn.Parameter(torch.randn(1, 1, d_req) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_req, nhead=n_head, dim_feedforward=4 * d_req, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        # enable_nested_tensor=False: train/eval 모드에서 패딩 위치 출력이 달라지지 않도록 fast path 비활성화
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers,
                                             norm=nn.LayerNorm(d_req), enable_nested_tensor=False)

    def forward(self, requests, mask):           # requests [B,R,F], mask [B,R] (True=실제 요청)
        B, R, _ = requests.shape
        X = self.row_mlp(requests)
        if self.pos_encoding == "learned":
            X = X + self.pos_emb(torch.arange(R, device=X.device)).unsqueeze(0)
        elif self.pos_encoding == "sinusoidal":
            X = X + self.pos_emb(R).unsqueeze(0)

        pad = ~mask                               # PyTorch 규약: True = 무시(패딩)
        if self.use_cls:
            X = torch.cat([self.cls.expand(B, -1, -1), X], dim=1)
            pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=pad.device), pad], dim=1)
        else:
            # 요청이 0개인 배치: 모든 key가 마스킹되면 softmax가 NaN → 슬롯 0만 열어 둠
            # (이 경우 행동도 0으로 강제되므로 결과에 영향 없음)
            all_pad = pad.all(dim=-1)
            if all_pad.any():
                pad = pad.clone()
                pad[all_pad, 0] = False

        Z = self.encoder(X, src_key_padding_mask=pad)
        if self.use_cls:
            return Z[:, 0], Z[:, 1:]
        return Z


class Actor(nn.Module):
    """요청별 logit (위치 인코딩 때문에 순열 등변이 아님)"""
    def __init__(self, N, f_req, d_global, d_req, n_head, dropout, R_max, pos_encoding):
        super().__init__()
        self.global_enc = GlobalEncoder(N, d_global)
        self.req_enc = RequestTransformerEncoder(f_req, d_req, n_head, R_max, dropout,
                                                 pos_encoding=pos_encoding, use_cls=False)
        self.head = nn.Sequential(nn.Linear(d_global + d_req, 128), nn.ReLU(), nn.Linear(128, 1))
        nn.init.orthogonal_(self.head[-1].weight, gain=0.01)   # 초기 정책 ≈ 균등
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, num_key, t, requests, mask):
        g = self.global_enc(num_key, t)
        Z = self.req_enc(requests, mask)
        R = requests.shape[1]
        return self.head(torch.cat([g.unsqueeze(1).expand(-1, R, -1), Z], -1)).squeeze(-1)


class Critic(nn.Module):
    """상태 가치 ([CLS] 토큰 풀링)"""
    def __init__(self, N, f_req, d_global, d_req, n_head, dropout, R_max, pos_encoding):
        super().__init__()
        self.global_enc = GlobalEncoder(N, d_global)
        self.req_enc = RequestTransformerEncoder(f_req, d_req, n_head, R_max, dropout,
                                                 pos_encoding=pos_encoding, use_cls=True)
        self.head = nn.Sequential(nn.Linear(d_global + d_req, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, num_key, t, requests, mask):
        g = self.global_enc(num_key, t)
        cls, _ = self.req_enc(requests, mask)
        return self.head(torch.cat([g, cls], -1)).squeeze(-1)


class TransformerPolicy(nn.Module):
    """표준 Transformer 기반 Actor/Critic (인터페이스는 SetTransformerPolicy와 동일)"""
    def __init__(self, R_max: int, N: int, f_req: int,
                 d_global: int = 128, d_req: int = 128, n_head: int = 4, dropout: float = 0.0,
                 pos_encoding: str = "learned"):
        super().__init__()
        self.R_max = R_max
        self.actor = Actor(N, f_req, d_global, d_req, n_head, dropout, R_max, pos_encoding)
        self.critic = Critic(N, f_req, d_global, d_req, n_head, dropout, R_max, pos_encoding)

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
    def __init__(self, R_max: int, N: int, f_req: int, cfg: PPOConfig = PPOConfig(),
                 pos_encoding: str = "learned"):
        self.cfg = cfg
        self.net = TransformerPolicy(R_max, N, f_req, pos_encoding=pos_encoding).to(cfg.device)
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
    N: int = 14
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
            _, last_v = agent.net(TransformerPolicy._ensure_batch(obs_to_tensors(obs, device))[0])
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
            path = os.path.join(tcfg.save_dir, f"NSFNET_tfppo_update{update:04d}.pt")
            torch.save({"model": agent.net.state_dict(), "optimizer": agent.opt.state_dict(),
                        "update": update, "global_step": global_step}, path)

    final = os.path.join(tcfg.save_dir, "NSFNET_tfppo_final.pt")
    torch.save({"model": agent.net.state_dict(), "optimizer": agent.opt.state_dict(),
                "update": tcfg.total_updates, "global_step": global_step}, final)
    print(f"Training finished. Final checkpoint saved to {final}")

    logging.basicConfig(filename="NSFNET_tf_reward.log", level=logging.INFO, format="%(asctime)s - %(message)s")
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
        fig.savefig(os.path.join(tcfg.save_dir, "NSFNET_tf_training_curves.png"), dpi=150)
        plt.close(fig)
    except Exception as e:
        print("Plotting failed:", e)
    return agent


if __name__ == "__main__":
    train()
