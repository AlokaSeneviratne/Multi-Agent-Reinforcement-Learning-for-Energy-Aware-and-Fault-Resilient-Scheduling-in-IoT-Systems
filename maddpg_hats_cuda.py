"""
MADDPG-style CTDE for the Hats Puzzle (CUDA-forced, MLP actors)
with Automatic Atomic Checkpointing & Auto-Resume, plus Windows sleep prevention.

Drop-in file: maddpg_hats_cuda.py
"""

import os
import json
import pickle
import random
import tempfile
import io
import ctypes
from collections import deque, namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
import matplotlib.pyplot as plt

# -------------------- Hyperparams --------------------
class HParams:
    # Force CUDA: fail early if not available
    assert torch.cuda.is_available(), "CUDA is not available. Install CUDA-enabled PyTorch or use a GPU machine."
    try:
        torch.set_default_device('cuda')
    except Exception:
        pass
    device = torch.device('cuda')

    seed = 42
    curriculum = [5, 7, 10, 15, 20]          # curriculum stages
    steps_per_curriculum = 10000             # gradient steps per curriculum stage
    envs_per_fill = 512                      # episodes collected per buffer-fill iteration
    buffer_capacity = 200_000
    batch_size = 1024

    gamma = 0.99
    tau = 0.01
    actor_lr = 1e-4
    critic_lr = 1e-3
    actor_hidden = 128
    critic_hidden = 256

    gumbel_temp_init = 1.0
    gumbel_anneal = 0.99999
    gumbel_min = 0.1

    eval_episodes = 400
    save_dir = "./maddpg_hats_ckpt"
    print_every_steps = 500   # also the periodic checkpoint interval

# -------------------- Hats environment --------------------
class HatsEnv:
    def __init__(self, N=10):
        self.N = N
        self.reset()

    def reset(self):
        self.hats = np.random.randint(0, 2, size=(self.N,), dtype=np.int64)
        self.announced = -1 * np.ones(self.N, dtype=np.int64)  # -1 unknown
        self.step_idx = 0
        self.done = False
        return self._get_obs()

    def _get_obs(self):
        agent = self.N - 1 - self.step_idx
        front = np.zeros(self.N, dtype=np.float32)
        if agent > 0:
            front[:agent] = self.hats[:agent]
        announced = self.announced.copy().astype(np.float32)
        announced_mask = (self.announced != -1).astype(np.float32)
        return {
            'agent_idx': float(agent),
            'front': front,
            'announced': announced,
            'announced_mask': announced_mask,
            'n': float(self.N)
        }

    def step(self, action):
        assert not self.done
        agent = self.N - 1 - self.step_idx
        correct = int(action == self.hats[agent])
        self.announced[agent] = int(action)
        self.step_idx += 1
        if self.step_idx >= self.N:
            self.done = True
            obs = None
        else:
            obs = self._get_obs()
        return obs, float(correct), self.done, {'agent': agent, 'correct': correct}

    def full_state(self):
        return self.hats.copy()

# ---------- parity baseline helper ----------
def parity_baseline(hats):
    total_parity = int(hats.sum() % 2)
    announced = -1 * np.ones_like(hats)
    announced[-1] = total_parity
    for t in range(1, len(hats)):
        agent = len(hats) - 1 - t
        front = hats[:agent]
        behind_announced = announced[agent+1:]
        hat = (total_parity - int(front.sum() % 2) - int(behind_announced.sum() % 2)) % 2
        announced[agent] = int(hat)
    return announced

# ---------- Replay buffer ----------
Transition = namedtuple('Transition', [
    'hats', 'announced', 'announced_mask', 'agent_idx', 'obs_vec',
    'joint_actions', 'action_onehot', 'reward', 'next_hats', 'next_announced', 'done'
])

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)

    # atomic pickle save; store container as plain list
    def save(self, path):
        try:
            payload = {
                "capacity": self.buffer.maxlen,
                "data": list(self.buffer)
            }
            tmp_fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", dir=os.path.dirname(path) or ".")
            with os.fdopen(tmp_fd, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        except Exception as e:
            print(f"[WARNING] Could not save replay buffer to {path}: {e}")

    # safe load: tolerant to missing/corrupt files
    def load(self, path):
        if not os.path.exists(path):
            print(f"[INFO] Replay buffer file not found: {path}. Starting with empty buffer.")
            self.buffer = deque(maxlen=self.buffer.maxlen)
            return
        try:
            with open(path, "rb") as f:
                payload = pickle.load(f)
            cap = payload.get("capacity", self.buffer.maxlen)
            data = payload.get("data", [])
            self.buffer = deque(data, maxlen=cap)
            print(f"[INFO] Replay buffer loaded from {path} with {len(self.buffer)} transitions (cap={cap}).")
        except EOFError:
            print(f"[WARNING] Replay buffer {path} incomplete/corrupted (EOF). Starting empty.")
            self.buffer = deque(maxlen=self.buffer.maxlen)
        except Exception as e:
            print(f"[WARNING] Could not load replay buffer from {path}: {e}. Starting empty.")
            self.buffer = deque(maxlen=self.buffer.maxlen)

# ---------- Utility functions ----------
def obs_to_vector(obs, N):
    return np.concatenate([
        obs['front'],
        obs['announced'],
        obs['announced_mask'],
        np.array([obs['agent_idx']], dtype=np.float32)
    ])

def actions_to_onehot_matrix(actions_np, N):
    """
    Convert actions (-1, 0, 1) for each agent into a one-hot encoding of shape (batch, N*2).
    Unknown actions (-1 or anything not 0/1) are encoded as zeros for that agent.
    """
    batch = actions_np.shape[0]
    out = np.zeros((batch, N*2), dtype=np.float32)
    for b in range(batch):
        for i in range(N):
            try:
                a = int(actions_np[b, i])
            except (ValueError, TypeError):
                continue  # skip if cannot cast
            if a == 0 or a == 1:
                out[b, i*2 + a] = 1.0
    return out

def action_to_onehot(a):
    v = np.zeros(2, dtype=np.float32)
    v[int(a)] = 1.0
    return v

def gumbel_softmax_sample(logits, temperature):
    U = torch.rand_like(logits)
    g = -torch.log(-torch.log(U + 1e-20) + 1e-20)
    y = (logits + g) / temperature
    return torch.softmax(y, dim=-1)

def hard_onehot_from_relaxed(y):
    ind = y.argmax(dim=-1, keepdim=True)
    y_hard = torch.zeros_like(y).scatter_(-1, ind, 1.0)
    return (y_hard - y).detach() + y

# ---------- Networks ----------
class ActorMLP(nn.Module):
    def __init__(self, N, hidden=128):
        super().__init__()
        self.in_dim = N*3 + 1
        self.mlp = nn.Sequential(
            nn.Linear(self.in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2)
        )

    def forward(self, x):
        return self.mlp(x)

class CriticMLP(nn.Module):
    def __init__(self, N, hidden=256):
        super().__init__()
        self.in_dim = N + N + N*2 + 1
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, hats, announced, joint_actions_onehot, stepidx):
        x = torch.cat([hats, announced, joint_actions_onehot, stepidx], dim=-1)
        return self.net(x).squeeze(-1)

# ---------- MADDPG trainer ----------
class MADDPG:
    def __init__(self, N, hp: HParams):
        self.N = N
        self.hp = hp
        self.device = hp.device
        self.actors = [ActorMLP(N, hidden=hp.actor_hidden).to(self.device) for _ in range(N)]
        self.target_actors = [ActorMLP(N, hidden=hp.actor_hidden).to(self.device) for _ in range(N)]
        self.critics = [CriticMLP(N, hidden=hp.critic_hidden).to(self.device) for _ in range(N)]
        self.target_critics = [CriticMLP(N, hidden=hp.critic_hidden).to(self.device) for _ in range(N)]
        self.actor_opts = [optim.Adam(a.parameters(), lr=hp.actor_lr) for a in self.actors]
        self.critic_opts = [optim.Adam(c.parameters(), lr=hp.critic_lr) for c in self.critics]

        for i in range(N):
            self.target_actors[i].load_state_dict(self.actors[i].state_dict())
            self.target_critics[i].load_state_dict(self.critics[i].state_dict())

        self.replay = ReplayBuffer(hp.buffer_capacity)
        self.gumbel_temp = hp.gumbel_temp_init
        self.train_step = 0

    # state helpers for models + optimizers
    def state_dict(self):
        return {
            'N': self.N,
            'gumbel_temp': self.gumbel_temp,
            'train_step': self.train_step,
            'actors': [a.state_dict() for a in self.actors],
            'target_actors': [a.state_dict() for a in self.target_actors],
            'critics': [c.state_dict() for c in self.critics],
            'target_critics': [c.state_dict() for c in self.target_critics],
            'actor_opts': [opt.state_dict() for opt in self.actor_opts],
            'critic_opts': [opt.state_dict() for opt in self.critic_opts],
        }

    def load_state_dict(self, sd):
        assert sd['N'] == self.N, f"Checkpoint N={sd['N']} does not match current N={self.N}"
        self.gumbel_temp = sd.get('gumbel_temp', self.gumbel_temp)
        self.train_step = sd.get('train_step', 0)
        for i in range(self.N):
            self.actors[i].load_state_dict(sd['actors'][i])
            self.target_actors[i].load_state_dict(sd['target_actors'][i])
            self.critics[i].load_state_dict(sd['critics'][i])
            self.target_critics[i].load_state_dict(sd['target_critics'][i])
            self.actor_opts[i].load_state_dict(sd['actor_opts'][i])
            self.critic_opts[i].load_state_dict(sd['critic_opts'][i])

    # atomic save/load for models+optimizers
    def save_models(self, path):
        os.makedirs(path, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", dir=path)
        try:
            with os.fdopen(tmp_fd, "wb") as f:
                torch.save(self.state_dict(), f)
                f.flush()
                os.fsync(f.fileno())
            final = os.path.join(path, "models_optim.pth")
            os.replace(tmp_path, final)
        except Exception as e:
            print(f"[WARNING] Could not atomic-save models to {path}: {e}")
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    def load_models(self, path):
        p = os.path.join(path, "models_optim.pth")
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        # Try to use safer loading API if supported
        try:
            # PyTorch >=2.5 supports weights_only flag
            sd = torch.load(p, map_location=self.device, weights_only=True)
        except TypeError:
            sd = torch.load(p, map_location=self.device)
        self.load_state_dict(sd)

    def add_transition(self, *args):
        self.replay.add(*args)

    def sample_and_update(self):
        if len(self.replay) < self.hp.batch_size:
            return

        trans = self.replay.sample(self.hp.batch_size)
        # Unzip and convert to tensors on device
        hats_np = np.stack(trans.hats)
        announced_np = np.stack(trans.announced)
        agent_idx_np = np.array(trans.agent_idx)
        joint_actions_np = np.stack(trans.joint_actions)
        action_onehot_np = np.stack(trans.action_onehot)
        reward_np = np.array(trans.reward, dtype=np.float32)
        next_hats_np = np.stack(trans.next_hats)
        next_announced_np = np.stack(trans.next_announced)
        done_np = np.array(trans.done, dtype=np.float32)

        hats = torch.tensor(hats_np, dtype=torch.float32, device=self.device)
        announced = torch.tensor(announced_np, dtype=torch.float32, device=self.device)
        agent_idx = torch.tensor(agent_idx_np, dtype=torch.float32, device=self.device).unsqueeze(-1)
        joint_actions_onehot = torch.tensor(actions_to_onehot_matrix(joint_actions_np, self.N), dtype=torch.float32, device=self.device)
        rewards = torch.tensor(reward_np, dtype=torch.float32, device=self.device)
        next_hats = torch.tensor(next_hats_np, dtype=torch.float32, device=self.device)
        next_announced = torch.tensor(next_announced_np, dtype=torch.float32, device=self.device)
        dones = torch.tensor(done_np, dtype=torch.float32, device=self.device)

        batch_size = hats.shape[0]

        # Build next obs per agent
        input_size = self.N*3 + 1
        per_agent_next_obs = torch.zeros((self.N, batch_size, input_size), device=self.device)
        for j in range(self.N):
            front = torch.zeros((batch_size, self.N), device=self.device)
            if j > 0:
                front[:, :j] = next_hats[:, :j]
            announced_t = next_announced
            announced_mask_t = (announced_t != -1).float()
            agent_col = torch.full((batch_size, 1), float(j), device=self.device)
            per_agent_next_obs[j] = torch.cat([front, announced_t, announced_mask_t, agent_col], dim=-1)

        with torch.no_grad():
            target_next_onehot_parts = []
            for j in range(self.N):
                logits = self.target_actors[j](per_agent_next_obs[j])
                rela = gumbel_softmax_sample(logits, max(self.gumbel_temp, self.hp.gumbel_min))
                rela_hard = hard_onehot_from_relaxed(rela)
                target_next_onehot_parts.append(rela_hard)
            target_joint_next_onehot = torch.cat(target_next_onehot_parts, dim=-1)

        target_Q = torch.zeros(batch_size, device=self.device)
        for b in range(batch_size):
            m = int(agent_idx[b].item())
            next_stepidx = torch.tensor([[max(0.0, float(m-1)) / float(self.N)]], dtype=torch.float32, device=self.device)
            q_next = self.target_critics[m](
                next_hats[b].unsqueeze(0),
                next_announced[b].unsqueeze(0),
                target_joint_next_onehot[b].unsqueeze(0),
                next_stepidx
            )
            target_Q[b] = rewards[b] + (1. - dones[b]) * self.hp.gamma * q_next.squeeze(0)

        # Critic updates
        for m in range(self.N):
            mask = (agent_idx.squeeze(-1) == float(m))
            if mask.sum().item() == 0:
                continue
            idxs = mask.nonzero(as_tuple=False).squeeze(-1)
            hats_b = hats[idxs]
            announced_b = announced[idxs]
            joint_actions_b = joint_actions_onehot[idxs]
            stepidx_b = (agent_idx[idxs] / float(self.N))
            target_b = target_Q[idxs].detach()
            pred = self.critics[m](hats_b, announced_b, joint_actions_b, stepidx_b)
            loss_c = nn.MSELoss()(pred, target_b)
            self.critic_opts[m].zero_grad()
            loss_c.backward()
            torch.nn.utils.clip_grad_norm_(self.critics[m].parameters(), max_norm=1.0)
            self.critic_opts[m].step()

        # Actor updates
        for m in range(self.N):
            mask = (agent_idx.squeeze(-1) == float(m))
            if mask.sum().item() == 0:
                continue
            idxs = mask.nonzero(as_tuple=False).squeeze(-1)
            hats_b = hats[idxs]
            announced_b = announced[idxs]
            stepidx_b = (agent_idx[idxs] / float(self.N))
            joint_actions_b = joint_actions_onehot[idxs].clone()
            k = hats_b.shape[0]
            front = torch.zeros((k, self.N), device=self.device)
            if m > 0:
                front[:, :m] = hats_b[:, :m]
            announced_mask_b = (announced_b != -1).float()
            agent_idx_col = torch.full((k,1), float(m), device=self.device)
            obs_actor = torch.cat([front, announced_b, announced_mask_b, agent_idx_col], dim=-1)
            logits = self.actors[m](obs_actor)
            rela = gumbel_softmax_sample(logits, max(self.gumbel_temp, self.hp.gumbel_min))
            rela_hard = hard_onehot_from_relaxed(rela)
            joint_actions_b[:, m*2 : m*2+2] = rela_hard
            q_val = self.critics[m](hats_b, announced_b, joint_actions_b, stepidx_b)
            actor_loss = -q_val.mean()
            self.actor_opts[m].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[m].parameters(), max_norm=1.0)
            self.actor_opts[m].step()

        # soft-update targets
        for i in range(self.N):
            for param, target_param in zip(self.actors[i].parameters(), self.target_actors[i].parameters()):
                target_param.data.copy_(self.hp.tau * param.data + (1 - self.hp.tau) * target_param.data)
            for param, target_param in zip(self.critics[i].parameters(), self.target_critics[i].parameters()):
                target_param.data.copy_(self.hp.tau * param.data + (1 - self.hp.tau) * target_param.data)

        # anneal temp & increment
        self.gumbel_temp = max(self.hp.gumbel_min, self.gumbel_temp * self.hp.gumbel_anneal)
        self.train_step += 1

    def act(self, obs_vec, agent_idx, deterministic=False):
        obs = torch.tensor(obs_vec[None, :], dtype=torch.float32, device=self.device)
        m = int(agent_idx)
        logits = self.actors[m](obs)
        if deterministic:
            return int(torch.argmax(logits, dim=-1).item())
        else:
            rela = gumbel_softmax_sample(logits, max(self.gumbel_temp, self.hp.gumbel_min))
            rela_hard = hard_onehot_from_relaxed(rela)
            return int(rela_hard.argmax(dim=-1).item())

# ---------- Data collection ----------
def collect_episodes_fill(env_cls, num_eps, N, maddpg: MADDPG, hp: HParams):
    for _ in range(num_eps):
        env = env_cls(N)
        obs = env.reset()
        joint_actions = np.full((N,), -1, dtype=np.int64)
        for t in range(N):
            agent = int(obs['agent_idx'])
            obs_vec = obs_to_vector(obs, N)
            act = maddpg.act(obs_vec, agent, deterministic=False)
            action_onehot = action_to_onehot(act)
            next_obs, reward, done, info = env.step(act)
            next_hats = env.hats.copy()
            next_announced = env.announced.copy()
            joint_snapshot = joint_actions.copy()
            joint_snapshot[agent] = act
            maddpg.add_transition(
                env.hats.copy(),
                env.announced.copy(),
                (env.announced != -1).astype(np.float32),
                float(agent),
                obs_vec,
                joint_snapshot.copy(),
                action_onehot.copy(),
                float(reward),
                next_hats.copy(),
                next_announced.copy(),
                float(done)
            )
            joint_actions[agent] = act
            obs = next_obs
            if done:
                break

# ---------- Evaluation ----------
@torch.no_grad()
def evaluate_policy(maddpg: MADDPG, N, episodes, hp: HParams):
    total_survivors = 0
    for _ in range(episodes):
        env = HatsEnv(N)
        obs = env.reset()
        for t in range(N):
            agent = int(obs['agent_idx'])
            obs_vec = obs_to_vector(obs, N)
            m = int(agent)
            x = torch.tensor(obs_vec[None, :], dtype=torch.float32, device=hp.device)
            logits = maddpg.actors[m](x)
            act = int(torch.argmax(logits, dim=-1).item())
            obs, reward, done, _ = env.step(act)
            if done:
                break
        survivors = int((env.announced == env.hats).sum())
        total_survivors += survivors
    return total_survivors / episodes

# ---------- Checkpoint I/O helpers (atomic + robust resume) ----------
def meta_path(hp): return os.path.join(hp.save_dir, "metadata.json")
def stage_dir(hp, N): return os.path.join(hp.save_dir, f"N{N}")
def models_path(hp, N): return os.path.join(stage_dir(hp, N), "models_optim.pth")
def buffer_path(hp, N): return os.path.join(stage_dir(hp, N), "replay.pkl")

def atomic_write_bytes(path: str, data: bytes):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", dir=os.path.dirname(path) or ".")
    try:
        with os.fdopen(tmp_fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

def atomic_write_json(obj, path: str):
    b = json.dumps(obj).encode("utf-8")
    atomic_write_bytes(path, b)

def save_checkpoint(maddpg: MADDPG, hp: HParams, curriculum_idx: int, in_stage_step: int):
    os.makedirs(stage_dir(hp, maddpg.N), exist_ok=True)
    # atomic save models
    maddpg.save_models(stage_dir(hp, maddpg.N))
    # atomic save replay buffer
    maddpg.replay.save(buffer_path(hp, maddpg.N))
    # atomic save metadata
    meta = {
        "curriculum": hp.curriculum,
        "curriculum_idx": curriculum_idx,
        "N": maddpg.N,
        "in_stage_step": in_stage_step,
        "steps_per_curriculum": hp.steps_per_curriculum,
        "gumbel_temp": maddpg.gumbel_temp,
        "train_step": maddpg.train_step
    }
    atomic_write_json(meta, meta_path(hp))

def try_autoresume(hp: HParams):
    if not os.path.exists(meta_path(hp)):
        return None
    try:
        with open(meta_path(hp), "r") as f:
            meta = json.load(f)
    except Exception as e:
        print(f"[WARNING] Could not read metadata.json: {e}. Ignoring resume.")
        return None
    if meta.get("curriculum") != hp.curriculum:
        print("Checkpoint curriculum differs from current; ignoring resume.")
        return None
    N = meta.get("N")
    idx = meta.get("curriculum_idx", 0)
    in_stage_step = meta.get("in_stage_step", 0)
    # model file must exist; buffer may be missing but we try to load it safely
    if not os.path.exists(models_path(hp, N)):
        print("Model checkpoint missing; ignoring resume.")
        return None
    print(f"Auto-resume found: N={N}, curriculum_idx={idx}, in_stage_step={in_stage_step}")
    trainer = MADDPG(N, hp)
    try:
        trainer.load_models(stage_dir(hp, N))
    except Exception as e:
        print(f"[WARNING] Could not load model checkpoint: {e}. Ignoring resume.")
        return None
    # Try to load replay but don't fail if it's corrupt/missing
    try:
        trainer.replay.load(buffer_path(hp, N))
    except Exception as e:
        print(f"[WARNING] Replay load error: {e}. Will warm up replay from scratch.")
        trainer.replay = ReplayBuffer(hp.buffer_capacity)
    trainer.gumbel_temp = meta.get("gumbel_temp", trainer.gumbel_temp)
    trainer.train_step = meta.get("train_step", trainer.train_step)
    return {"trainer": trainer, "curriculum_idx": idx, "in_stage_step": in_stage_step}

# ---------- Training loop ----------
def train_full(hp: HParams):
    random.seed(hp.seed)
    np.random.seed(hp.seed)
    torch.manual_seed(hp.seed)

    print("Forced device:", hp.device)
    try:
        print("CUDA device name:", torch.cuda.get_device_name(0))
    except Exception:
        pass
    print("CUDA visible:", torch.cuda.is_available())

    survivors_per_stage = []

    # attempt auto-resume
    resume = try_autoresume(hp)
    if resume is None:
        start_idx = 0
        in_stage_start_step = 0
        trainer = None
    else:
        start_idx = resume["curriculum_idx"]
        in_stage_start_step = resume["in_stage_step"]
        trainer = resume["trainer"]

    for ci in range(start_idx, len(hp.curriculum)):
        N = hp.curriculum[ci]
        print(f"\n=== Curriculum stage N = {N} ===")
        if trainer is None or trainer.N != N:
            trainer = MADDPG(N, hp)
            print("Filling replay with initial episodes...")
            collect_episodes_fill(HatsEnv, max(1024, hp.envs_per_fill), N, trainer, hp)
        else:
            if len(trainer.replay) == 0:
                print("Replay buffer empty on resume; warming up...")
                collect_episodes_fill(HatsEnv, max(1024, hp.envs_per_fill), N, trainer, hp)

        steps = hp.steps_per_curriculum
        start_step = in_stage_start_step if ci == start_idx else 0

        pbar = trange(start_step, steps, desc=f"Training N={N}")
        for step in pbar:
            collect_episodes_fill(HatsEnv, max(8, hp.envs_per_fill // 64), N, trainer, hp)
            updates_per_loop = 4
            for _ in range(updates_per_loop):
                trainer.sample_and_update()

            if step % hp.print_every_steps == 0 and step > start_step:
                pbar.set_postfix({
                    'gumbel_temp': f"{trainer.gumbel_temp:.3f}",
                    'buffer_len': len(trainer.replay)
                })
                save_checkpoint(trainer, hp, ci, step)

        mean_surv = evaluate_policy(trainer, N, hp.eval_episodes, hp)
        print(f"After N={N} stage evaluation mean survivors: {mean_surv:.3f}")
        survivors_per_stage.append(mean_surv)
        save_checkpoint(trainer, hp, ci, steps)

        in_stage_start_step = 0
        trainer = None

    # final plot
    plt.figure(figsize=(8,5))
    plt.plot(hp.curriculum, survivors_per_stage, marker='o')
    plt.xlabel('Number of Prisoners (N)')
    plt.ylabel('Average Survivors')
    plt.title('MADDPG Hats: Average Survivors per Curriculum Stage')
    plt.grid(True)
    plt.show()
    return survivors_per_stage

# ---------- Main ----------
if __name__ == "__main__":
    # Prevent Windows from sleeping while training (will revert when process exits)
    try:
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        print("[INFO] SetThreadExecutionState called to prevent sleep.")
    except Exception:
        print("[WARNING] Could not set thread execution state; continue without OS-level sleep prevention.")

    hp = HParams()
    try:
        train_full(hp)
    finally:
        # Clear the sleep-prevention request so OS can sleep normally again
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
        except Exception:
            pass
