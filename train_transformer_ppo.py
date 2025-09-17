import os
import time
import math
import random
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from qkdn_scheduling_env import QKDNSchedulingEnv

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
    return torch.tensor(x, dtype=torch.float32, device=device)

def obs_to_tensors(obs, device):
    # obs is dict of numpy arrays
    return {k: to_tensor(v, device) for k, v in obs.items()}

def stack_obs(obs_list: List[Dict[str, np.ndarray]], device):
    keys = obs_list[0].keys()
    out = {}
    for k in keys:
        out[k] = to_tensor(np.stack([o[k] for o in obs_list], axis=0), device)
    return out

# -----------------------------
# Attention & Set Transformer blocks
# -----------------------------

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_head == 0
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        # q: [B, Q, D], k/v: [B, K, D], mask: [B, Q, K] (True=keep, False=mask-out) or None
        B, Q, D = q.shape
        _, K, _ = k.shape

        q = self.q_proj(q).view(B, Q, self.n_head, self.d_head).transpose(1, 2)  # [B,H,Q,d]
        k = self.k_proj(k).view(B, K, self.n_head, self.d_head).transpose(1, 2)  # [B,H,K,d]
        v = self.v_proj(v).view(B, K, self.n_head, self.d_head).transpose(1, 2)  # [B,H,K,d]

        attn_logits = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)         # [B,H,Q,K]
        if mask is not None:
            attn_logits = attn_logits.masked_fill(~mask[:, None, :, :], float('-inf'))
        attn = attn_logits.softmax(dim=-1)                                       # [B,H,Q,K]
        attn = self.drop(attn)
        out = attn @ v                                                           # [B,H,Q,d]
        out = out.transpose(1, 2).contiguous().view(B, Q, D)                     # [B,Q,D]
        return self.o_proj(out)


class SAB(nn.Module):
    """Set Attention Block"""
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_head, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, X):  # X: [B, R, D]
        h = self.attn(self.ln1(X), self.ln1(X), self.ln1(X))
        X = X + h
        X = X + self.ffn(self.ln2(X))
        return X

# -----------------------------
# Encoders & Policy
# -----------------------------

class GlobalEncoder(nn.Module):
    """num_key(NxN) via CNN + t(1) embedding -> global vector"""
    def __init__(self, d_out: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)  # -> [B, 32, 1, 1]
        )
        self.t_proj = nn.Linear(1, 16)
        self.proj = nn.Linear(32 + 16, d_out)

    def forward(self, num_key, t):  # num_key: [B,N,N], t: ideally [B,1]
        # Ensure num_key has batch dim: [B,N,N] -> [B,1,N,N]
        if num_key.dim() == 2:                  # [N,N]
            num_key = num_key.unsqueeze(0)      # [1,N,N]
        x = num_key.unsqueeze(1).float()        # [B,1,N,N]
        g = self.conv(x).flatten(1)             # [B,32]

        # ---- Force t to [B,1] and match batch with g ----
        B = g.size(0)
        if t is None:
            t = torch.zeros(B, 1, device=g.device, dtype=g.dtype)
        else:
            if t.dim() == 0:                    # [] -> [1,1]
                t = t.view(1, 1)
            elif t.dim() == 1:                  # [B] or [1] -> [B,1]/[1,1]
                t = t.view(-1, 1)
            elif t.dim() == 2:                  # [B,1] ok, [B,K] -> slice first
                if t.size(1) != 1:
                    t = t[:, :1]
            else:                                # [*,*,*] -> [B,1]
                b = t.size(0)
                t = t.view(b, -1)[:, :1]

            # If batch doesn't match g, expand or reshape to [B,1]
            if t.size(0) == 1 and B > 1:
                t = t.expand(B, 1)
            elif t.size(0) != B:
                if t.size(0) < B:
                    reps = (B + t.size(0) - 1) // t.size(0)
                    t = t.repeat(reps, 1)[:B, :]
                else:
                    t = t[:B, :]

        t = t.to(g.device, dtype=g.dtype)
        t_feat = F.relu(self.t_proj(t))         # [B,16]
        g = torch.cat([g, t_feat], dim=-1)      # [B,48]
        g = self.proj(g)                        # [B,d_out]
        return g
class RequestSetEncoder(nn.Module):
    """(R,5) -> row MLP -> SABx2 -> (R,d_req)"""
    def __init__(self, d_req: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        self.row_mlp = nn.Sequential(
            nn.Linear(5, d_req), nn.ReLU(),
            nn.Linear(d_req, d_req), nn.ReLU(),
        )
        self.sab1 = SAB(d_req, n_head, dropout)
        self.sab2 = SAB(d_req, n_head, dropout)

    def forward(self, requests):  # [B,R,5]
        X = self.row_mlp(requests)
        X = self.sab1(X)
        X = self.sab2(X)
        return X


class SetTransformerPolicy(nn.Module):
    def __init__(self, R_max: int, d_global: int = 128, d_req: int = 128, n_head: int = 4, dropout: float = 0.0):
        super().__init__()
        self.R_max = R_max
        self.global_enc = GlobalEncoder(d_out=d_global)
        self.req_enc    = RequestSetEncoder(d_req=d_req, n_head=n_head, dropout=dropout)

        self.policy_head = nn.Sequential(
            nn.Linear(d_global + d_req, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.value_head = nn.Sequential(
            nn.Linear(d_global + d_req, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, obs: Dict[str, torch.Tensor]):
        obs, added = self._ensure_batch(obs)

        num_key = obs["num_key"]     # [B,N,N]
        requests = obs["requests"]   # [B,R,5]
        t = obs["t"]                 # [B,1]
        B, R, _ = requests.shape
        assert R == self.R_max

        g = self.global_enc(num_key, t)                 # [B,dg]
        Z = self.req_enc(requests)                      # [B,R,dr]

        g_expand = g.unsqueeze(1).expand(-1, R, -1)     # [B,R,dg]
        policy_in = torch.cat([g_expand, Z], dim=-1)    # [B,R,dg+dr]
        logits = self.policy_head(policy_in).squeeze(-1) # [B,R]

        Z_pool = Z.mean(dim=1)                          # [B,dr]
        v_in = torch.cat([g, Z_pool], dim=-1)           # [B,dg+dr]
        value = self.value_head(v_in)                   # [B,1]
        return logits, value

    def act(self, obs: Dict[str, torch.Tensor]):
        obs_fixed, added = self._ensure_batch(obs)
        logits, value = self.forward(obs_fixed)
        mask = obs_fixed["mask"].to(torch.bool)   # [B,R]
        masked_logits = logits.masked_fill(~mask, float('-inf'))
        dist = torch.distributions.Categorical(logits=masked_logits)
        action = dist.sample()                # [B]
        logp = dist.log_prob(action)          # [B]
        if added:
            # 단일 샘플이면 배치 차원 제거
            action = action.squeeze(0)
            logp   = logp.squeeze(0)
            value  = value.squeeze(0)
        return action, logp, value.squeeze(-1)

    def _ensure_batch(self, obs: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], bool]:
        """Ensure tensors have a batch dim ONLY if missing.
        Expected shapes (B is batch):
          - num_key: [B,N,N] or [N,N] (single)
          - requests: [B,R,5] or [R,5] (single)
          - mask: [B,R] or [R] (single)
          - t: [B,1] or [1] (single)
        """
        out = {}
        added = False

        # num_key
        nk = obs.get("num_key")
        if nk is not None:
            if nk.dim() == 2:                    # [N,N] -> add batch
                out["num_key"] = nk.unsqueeze(0) # [1,N,N]
                added = True
            else:                                # [B,N,N]
                out["num_key"] = nk

        # requests
        rq = obs.get("requests")
        if rq is not None:
            if rq.dim() == 2:                    # [R,5] -> add batch
                out["requests"] = rq.unsqueeze(0) # [1,R,5]
                added = True
            else:                                 # [B,R,5]
                out["requests"] = rq

        # mask
        mk = obs.get("mask")
        if mk is not None:
            if mk.dim() == 1:                     # [R] -> add batch
                out["mask"] = mk.unsqueeze(0)     # [1,R]
                added = True
            else:                                  # [B,R]
                out["mask"] = mk

        # t
        tt = obs.get("t")
        if tt is not None:
            if tt.dim() == 1:                     # [1] or [B] -> [B,1]
                if tt.numel() == 1:               # scalar in tensor
                    out["t"] = tt.view(1, 1)      # [1,1]
                    added = True
                else:
                    out["t"] = tt.view(-1, 1)     # [B,1]
            elif tt.dim() == 2:                   # [B,1] or [B,K]
                out["t"] = tt[:, :1]              # keep first channel
            else:                                  # [*,*,*] -> [B,1]
                b = tt.size(0)
                out["t"] = tt.view(b, -1)[:, :1]

        return out, added

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
    update_epochs: int = 4
    minibatch_size: int = 64
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class PPOAgent:
    def __init__(self, R_max: int, cfg: PPOConfig = PPOConfig()):
        self.cfg = cfg
        self.net = SetTransformerPolicy(R_max).to(cfg.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)

    @torch.no_grad()
    def select_action(self, obs_dict):
        tens = obs_to_tensors(obs_dict, self.cfg.device)
        a, logp, v = self.net.act(tens)
        return a.cpu().numpy(), logp.cpu().numpy(), v.cpu().numpy()

    def ppo_update(self, obs_t, act_t, old_logp_t, ret_t, adv_t):
        cfg = self.cfg
        # normalize advantages
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        num_samples = adv_t.size(0)
        idx = torch.randperm(num_samples, device=cfg.device)

        for _ in range(cfg.update_epochs):
            for start in range(0, num_samples, cfg.minibatch_size):
                mb_idx = idx[start:start+cfg.minibatch_size]

                obs_mb = {k: v[mb_idx] for k, v in obs_t.items()}
                act_mb = act_t[mb_idx]
                old_logp_mb = old_logp_t[mb_idx]
                ret_mb = ret_t[mb_idx]
                adv_mb = adv_t[mb_idx]

                logits, value = self.net(obs_mb)
                mask = obs_mb["mask"].to(torch.bool)
                masked_logits = logits.masked_fill(~mask, float('-inf'))
                dist = torch.distributions.Categorical(logits=masked_logits)

                logp = dist.log_prob(act_mb)
                ratio = torch.exp(logp - old_logp_mb)

                pg_unclipped = ratio * adv_mb
                pg_clipped = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv_mb
                pg_loss = -torch.min(pg_unclipped, pg_clipped).mean()

                v_loss = F.mse_loss(value.squeeze(-1), ret_mb)
                ent = dist.entropy().mean()

                loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * ent

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                self.opt.step()

# -----------------------------
# Rollout & GAE
# -----------------------------

def compute_gae(rewards, values, dones, gamma, lam):
    """
    rewards, values, dones: shape [T]
    returns advantages, returns_t: shape [T]
    """
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    lastgaelam = 0.0
    for t in reversed(range(T)):
        nextnonterminal = 1.0 - float(dones[t])
        nextvalue = values[t + 1] if t + 1 < len(values) else 0.0
        delta = rewards[t] + gamma * nextvalue * nextnonterminal - values[t]
        lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
        adv[t] = lastgaelam
    ret = adv + values[:T]
    return adv, ret

# -----------------------------
# Training Loop
# -----------------------------

@dataclass
class TrainConfig:
    seed: int = 0
    total_updates: int = 200
    rollout_steps: int = 1024   # steps per update
    log_interval: int = 10
    save_interval: int = 50
    save_dir: str = "./checkpoints"

def make_env(max_time_step=5, R_max=5, N=6, seed=0):
    topo = {"NUM_QKD_NODE": N}
    env = QKDNSchedulingEnv(
        topo,
        max_time_steps=max_time_step,               # longer episodes for training
        max_requests_per_step=R_max,
        auto_continue=True,
        request_wait_episodes=1,
        seed=seed
    )
    return env

def train():
    tcfg = TrainConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(tcfg.seed)

    # Build env and agent
    R_max = 10
    env = make_env(R_max=R_max, N=14, seed=tcfg.seed)
    obs, info = env.reset()

    agent = PPOAgent(R_max, PPOConfig(device=device))
    os.makedirs(tcfg.save_dir, exist_ok=True)

    ep_returns = []
    global_step = 0

    for update in range(1, tcfg.total_updates + 1):
        # Storage
        obs_buf = []
        act_buf = []
        logp_buf = []
        val_buf = []
        rew_buf = []
        done_buf = []

        # Rollout
        steps = 0
        while steps < tcfg.rollout_steps:
            obs_buf.append(obs)
            act, logp, val = agent.select_action(obs)
            # env expects scalar action
            a = int(act[0]) if np.ndim(act) > 0 else int(act)
            obs, reward, terminated, truncated, info = env.step(a)

            act_buf.append(a)
            logp_buf.append(float(logp[0]) if np.ndim(logp) > 0 else float(logp))
            val_buf.append(float(val[0]) if np.ndim(val) > 0 else float(val))
            rew_buf.append(float(reward))
            done = bool(terminated or truncated)  # with auto_continue, usually False
            done_buf.append(done)

            global_step += 1
            steps += 1

        # Bootstrap value for GAE (last obs)
        last_v = agent.net.forward(obs_to_tensors(obs, device))[1].detach().cpu().numpy().squeeze()
        val_np = np.array(val_buf + [last_v], dtype=np.float32)
        rew_np = np.array(rew_buf, dtype=np.float32)
        done_np = np.array(done_buf, dtype=np.bool_)

        adv_np, ret_np = compute_gae(rew_np, val_np, done_np, agent.cfg.gamma, agent.cfg.lam)

        # Prepare tensors
        obs_t = stack_obs(obs_buf, device)
        act_t = torch.tensor(np.array(act_buf), dtype=torch.long, device=device)
        old_logp_t = torch.tensor(np.array(logp_buf), dtype=torch.float32, device=device)
        adv_t = torch.tensor(adv_np, dtype=torch.float32, device=device)
        ret_t = torch.tensor(ret_np, dtype=torch.float32, device=device)

        # PPO update
        agent.ppo_update(obs_t, act_t, old_logp_t, ret_t, adv_t)

        # Logging
        if update % tcfg.log_interval == 0:
            avg_rew = float(np.mean(rew_np)) if len(rew_np) > 0 else 0.0
            print(f"[Update {update:04d}] steps={global_step} avg_step_reward={avg_rew:.3f} "
                  f"adv_mean={adv_np.mean():.3f} ret_mean={ret_np.mean():.3f} "
                  f"success_total={info.get('success_total', -1)} ep_idx={info.get('episode_idx', -1)}")

        # Save
        if update % tcfg.save_interval == 0:
            ckpt_path = os.path.join(tcfg.save_dir, f"setppo_update{update:04d}.pt")
            torch.save({
                "model": agent.net.state_dict(),
                "optimizer": agent.opt.state_dict(),
                "update": update,
                "global_step": global_step,
            }, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

    # Final save
    ckpt_path = os.path.join(tcfg.save_dir, f"setppo_final.pt")
    torch.save({
        "model": agent.net.state_dict(),
        "optimizer": agent.opt.state_dict(),
        "update": tcfg.total_updates,
        "global_step": global_step,
    }, ckpt_path)
    print(f"Training finished. Final checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    train()