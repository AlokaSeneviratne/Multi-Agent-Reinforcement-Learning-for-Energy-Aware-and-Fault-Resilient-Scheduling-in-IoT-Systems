import os
import json
import pickle
import random
import tempfile
import ctypes
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from tqdm import trange
import matplotlib.pyplot as plt

# -------------------- Hyperparams --------------------
class HParams:
    # Force CUDA: fail early if not available
    assert torch.cuda.is_available()
    try:
        torch.set_default_device('cuda')
    except Exception:
        pass
    device = torch.device('cuda')

    # RNG and curriculum
    seed = 42
    curriculum = [5, 7, 10, 15, 20, 50, 100]           # curriculum stages (N)
    steps_per_curriculum = 10000              # "training loops" per stage (similar to your old code)
    envs_per_fill = 512                       # episodes collected per buffer-fill iteration

    # Replay & batches (episode-based replay for RNNs)
    buffer_capacity = 60000                   # number of episodes kept (not transitions)
    batch_episodes = 32                       # how many episodes per gradient update

    # DDQRN hyperparams
    gamma = 0.95                              # <-- requested
    tau = 0.01                                # soft target update rate (α⁻) <-- requested
    lr = 1e-3                                 # learning rate <-- requested
    hidden = 256                              # hidden size for LSTM/FC
    lstm_layers = 1

    # Exploration: epsilon = 1 - 0.5^(1/N) per stage
    # (computed dynamically inside training loop per N)

    # Misc
    eval_episodes = 400
    save_dir = "./maddpg_hats_ckpt"
    print_every_steps = 500   # also periodic checkpoint interval


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


# ---------- Utility functions ----------
def obs_to_vector(obs, N):
    return np.concatenate([
        obs['front'],
        obs['announced'],
        obs['announced_mask'],
        np.array([obs['agent_idx']], dtype=np.float32)
    ])


# ---------- Replay buffer (episode-based, atomic save/load) ----------
class EpisodeReplay:
    """
    Stores full episodes so RNNs can be trained on unrolled sequences.
    Each episode is a dict:
    {
        "obs":  (T, input_dim)   float32
        "act":  (T,)             int64
        "rew":  (T,)             float32
        "done": (T,)             float32  (1.0 at terminal step, else 0.0)
        "next_obs": (T, input_dim) float32 (next obs after each step; zero for terminal if desired)
        "N": int
    }
    """
    def __init__(self, capacity_episodes: int):
        self.buffer = deque(maxlen=capacity_episodes)

    def add_episode(self, ep_dict):
        self.buffer.append(ep_dict)

    def sample_episodes(self, batch_episodes: int):
        batch = random.sample(self.buffer, min(batch_episodes, len(self.buffer)))
        return batch

    def __len__(self):
        return len(self.buffer)

    # Atomic pickle save
    def save(self, path):
        try:
            payload = {
                "capacity": self.buffer.maxlen,
                "episodes": list(self.buffer)
            }
            tmp_fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", dir=os.path.dirname(path) or ".")
            with os.fdopen(tmp_fd, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        except Exception as e:
            print(f"[WARNING] Could not save episode replay to {path}: {e}")

    # Safe load
    def load(self, path):
        if not os.path.exists(path):
            print(f"[INFO] Replay file not found: {path}. Starting with empty replay.")
            return
        try:
            with open(path, "rb") as f:
                payload = pickle.load(f)
            cap = payload.get("capacity", self.buffer.maxlen)
            episodes = payload.get("episodes", [])
            self.buffer = deque(episodes, maxlen=cap)
            print(f"[INFO] Loaded replay: {len(self.buffer)} episodes (cap={cap}).")
        except EOFError:
            print(f"[WARNING] Replay {path} incomplete/corrupt (EOF). Starting empty.")
        except Exception as e:
            print(f"[WARNING] Could not load replay {path}: {e}. Starting empty.")


# ---------- Q-Network with LSTM ----------
class QNetRNN(nn.Module):
    """
    Input per step: [front(N), announced(N), mask(N), agent_idx(1)] => size = N*3 + 1
    Forward expects (seq_len, batch, input_dim)
    Outputs Q-values per action: (seq_len, batch, 2)
    """
    def __init__(self, N, hidden=256, lstm_layers=1):
        super().__init__()
        self.N = N
        self.input_dim = N*3 + 1
        self.hidden = hidden
        self.lstm_layers = lstm_layers

        self.enc = nn.Linear(self.input_dim, hidden)
        self.lstm = nn.LSTM(hidden, hidden, num_layers=lstm_layers)
        self.head = nn.Linear(hidden, 2)

    def forward(self, x, hx=None):
        # x: (T, B, input_dim)
        x = torch.relu(self.enc(x))
        out, hx = self.lstm(x, hx)   # out: (T, B, hidden)
        q = self.head(out)           # (T, B, 2)
        return q, hx

    def init_hx(self, batch_size, device):
        h0 = torch.zeros(self.lstm_layers, batch_size, self.hidden, device=device)
        c0 = torch.zeros(self.lstm_layers, batch_size, self.hidden, device=device)
        return (h0, c0)


# ---------- DDQRN Agent ----------
class DDQRNAgent:
    def __init__(self, N, hp: HParams):
        self.N = N
        self.hp = hp
        self.device = hp.device

        self.online = QNetRNN(N, hidden=hp.hidden, lstm_layers=hp.lstm_layers).to(self.device)
        self.target = QNetRNN(N, hidden=hp.hidden, lstm_layers=hp.lstm_layers).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.opt = optim.Adam(self.online.parameters(), lr=hp.lr)
        self.replay = EpisodeReplay(hp.buffer_capacity)
        self.train_step = 0

    # ------- checkpointing state -------
    def state_dict(self):
        return {
            "N": self.N,
            "train_step": self.train_step,
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "opt": self.opt.state_dict(),
        }

    def load_state_dict(self, sd):
        assert sd['N'] == self.N, f"Checkpoint N={sd['N']} != current N={self.N}"
        self.train_step = sd.get("train_step", 0)
        self.online.load_state_dict(sd["online"])
        self.target.load_state_dict(sd["target"])
        self.opt.load_state_dict(sd["opt"])

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
        try:
            sd = torch.load(p, map_location=self.device)  # weights_only might not support full dicts
        except TypeError:
            sd = torch.load(p, map_location=self.device)
        self.load_state_dict(sd)

    # ------- epsilon-greedy action selection -------
    def act(self, obs_vec, epsilon, hx=None):
        """
        obs_vec: (input_dim,) numpy array
        epsilon: exploration probability
        hx: optional LSTM hidden state (for single-step eval we init fresh)
        returns: action int, hx
        """
        if random.random() < epsilon:
            return random.randint(0, 1), hx
        x = torch.tensor(obs_vec, dtype=torch.float32, device=self.device).view(1, 1, -1)  # (T=1,B=1,D)
        if hx is None:
            hx = self.online.init_hx(batch_size=1, device=self.device)
        with torch.no_grad():
            q, hx = self.online(x, hx)
            a = int(torch.argmax(q[0, 0]).item())
        return a, hx

    # ------- training update (DDQN on episode sequences) -------
    def sample_and_update(self):
        if len(self.replay) < self.hp.batch_episodes:
            return

        batch_eps = self.replay.sample_episodes(self.hp.batch_episodes)
        # Build tensors seq-major (T,B,D)
        # pad not needed since all episodes have length==N for this env
        B = len(batch_eps)
        T = self.N
        D = self.online.input_dim

        obs = np.zeros((T, B, D), dtype=np.float32)
        next_obs = np.zeros((T, B, D), dtype=np.float32)
        acts = np.zeros((T, B), dtype=np.int64)
        rews = np.zeros((T, B), dtype=np.float32)
        dones = np.zeros((T, B), dtype=np.float32)

        for b, ep in enumerate(batch_eps):
            obs[:, b, :] = ep["obs"]
            next_obs[:, b, :] = ep["next_obs"]
            acts[:, b] = ep["act"]
            rews[:, b] = ep["rew"]
            dones[:, b] = ep["done"]

        obs_t = torch.tensor(obs, device=self.device)
        next_obs_t = torch.tensor(next_obs, device=self.device)
        acts_t = torch.tensor(acts, device=self.device)
        rews_t = torch.tensor(rews, device=self.device)
        dones_t = torch.tensor(dones, device=self.device)

        # Q(s, a)
        hx = self.online.init_hx(B, self.device)
        q_online, _ = self.online(obs_t, hx)         # (T,B,2)
        q_sa = q_online.gather(-1, acts_t.unsqueeze(-1)).squeeze(-1)  # (T,B)

        # a* = argmax_a Q_online(s',a)
        with torch.no_grad():
            hx2 = self.online.init_hx(B, self.device)
            q_next_online, _ = self.online(next_obs_t, hx2)  # (T,B,2)
            a_star = torch.argmax(q_next_online, dim=-1)     # (T,B)

            # Q_target(s', a*)
            hx3 = self.target.init_hx(B, self.device)
            q_next_target, _ = self.target(next_obs_t, hx3)  # (T,B,2)
            q_next_star = q_next_target.gather(-1, a_star.unsqueeze(-1)).squeeze(-1)  # (T,B)

            y = rews_t + (1.0 - dones_t) * self.hp.gamma * q_next_star  # DDQN target

        # Loss and optimize
        loss = nn.SmoothL1Loss()(q_sa, y)  # Huber loss is common for DQN
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=1.0)
        self.opt.step()

        # Soft update target
        with torch.no_grad():
            tau = self.hp.tau
            for tp, p in zip(self.target.parameters(), self.online.parameters()):
                tp.data.mul_(1.0 - tau).add_(tau * p.data)

        self.train_step += 1


# ---------- Data collection (episodes) ----------
def collect_episodes_fill(env_cls, num_eps, N, agent: DDQRNAgent, epsilon: float):
    """
    Collect `num_eps` full episodes and put them into replay buffer.
    We use fresh LSTM states per episode (sequential interaction).
    """
    for _ in range(num_eps):
        env = env_cls(N)
        obs = env.reset()

        obs_seq = []
        next_obs_seq = []
        act_seq = []
        rew_seq = []
        done_seq = []

        hx = None
        for t in range(N):
            obs_vec = obs_to_vector(obs, N)
            action, hx = agent.act(obs_vec, epsilon=epsilon, hx=hx)
            next_obs, reward, done, info = env.step(action)
            next_vec = np.zeros_like(obs_vec) if next_obs is None else obs_to_vector(next_obs, N)

            obs_seq.append(obs_vec)
            next_obs_seq.append(next_vec)
            act_seq.append(int(action))
            rew_seq.append(float(reward))
            done_seq.append(1.0 if done else 0.0)

            obs = next_obs
            if done:
                break

        ep = {
            "obs":      np.stack(obs_seq, axis=0).astype(np.float32),      # (T,D)
            "next_obs": np.stack(next_obs_seq, axis=0).astype(np.float32), # (T,D)
            "act":      np.array(act_seq, dtype=np.int64),                 # (T,)
            "rew":      np.array(rew_seq, dtype=np.float32),               # (T,)
            "done":     np.array(done_seq, dtype=np.float32),              # (T,)
            "N":        N
        }
        # for consistency, ensure T == N (env always runs N steps)
        # (In this environment it always should; if not, we could pad.)

        agent.replay.add_episode(ep)


# ---------- Evaluation ----------
@torch.no_grad()
def evaluate_policy(agent: DDQRNAgent, N, episodes):
    total_survivors = 0
    for _ in range(episodes):
        env = HatsEnv(N)
        obs = env.reset()
        hx = agent.online.init_hx(1, agent.device)
        for t in range(N):
            obs_vec = obs_to_vector(obs, N)
            x = torch.tensor(obs_vec, dtype=torch.float32, device=agent.device).view(1, 1, -1)
            q, hx = agent.online(x, hx)
            act = int(torch.argmax(q[0, 0]).item())
            obs, reward, done, _ = env.step(act)
            if done:
                break
        survivors = int((env.announced == env.hats).sum())
        total_survivors += survivors
    return total_survivors / episodes


# ---------- Checkpoint I/O helpers ----------
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

def save_checkpoint(agent: DDQRNAgent, hp: HParams, curriculum_idx: int, in_stage_step: int):
    os.makedirs(stage_dir(hp, agent.N), exist_ok=True)
    agent.save_models(stage_dir(hp, agent.N))
    agent.replay.save(buffer_path(hp, agent.N))
    meta = {
        "kind": "DDQRN",
        "curriculum": hp.curriculum,
        "curriculum_idx": curriculum_idx,
        "N": agent.N,
        "in_stage_step": in_stage_step,
        "steps_per_curriculum": hp.steps_per_curriculum,
        "train_step": agent.train_step,
        "gamma": hp.gamma,
        "tau": hp.tau,
        "lr": hp.lr
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
    if not os.path.exists(models_path(hp, N)):
        print("Model checkpoint missing; ignoring resume.")
        return None

    print(f"Auto-resume found: N={N}, curriculum_idx={idx}, in_stage_step={in_stage_step}")
    agent = DDQRNAgent(N, hp)
    try:
        agent.load_models(stage_dir(hp, N))
    except Exception as e:
        print(f"[WARNING] Could not load models: {e}. Ignoring resume.")
        return None

    # load replay (tolerant)
    try:
        agent.replay.load(buffer_path(hp, N))
    except Exception as e:
        print(f"[WARNING] Replay load error: {e}. Will warm up replay from scratch.")
        agent.replay = EpisodeReplay(hp.buffer_capacity)

    agent.train_step = meta.get("train_step", agent.train_step)
    return {"agent": agent, "curriculum_idx": idx, "in_stage_step": in_stage_step}


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

    survivors_per_stage = []  # mean survivors per N at end of stage
    history = []              # (global_ep, N, success_rate)
    global_ep = 0

    # attempt auto-resume
    resume = try_autoresume(hp)
    if resume is None:
        start_idx = 0
        in_stage_start_step = 0
        agent = None
    else:
        start_idx = resume["curriculum_idx"]
        in_stage_start_step = resume["in_stage_step"]
        agent = resume["agent"]

    for ci in range(start_idx, len(hp.curriculum)):
        N = hp.curriculum[ci]
        # epsilon per stage: 1 - 0.5^(1/N)
        epsilon = 1.0 - (0.5 ** (1.0 / float(N)))
        print(f"\n=== Curriculum stage N = {N} | epsilon = {epsilon:.4f} ===")

        if agent is None or agent.N != N:
            agent = DDQRNAgent(N, hp)
            print("Filling replay with initial episodes...")
            collect_episodes_fill(HatsEnv, max(1024, hp.envs_per_fill), N, agent, epsilon)
        else:
            if len(agent.replay) == 0:
                print("Replay empty on resume; warming up...")
                collect_episodes_fill(HatsEnv, max(1024, hp.envs_per_fill), N, agent, epsilon)

        steps = hp.steps_per_curriculum
        start_step = in_stage_start_step if ci == start_idx else 0

        pbar = trange(start_step, steps, desc=f"Training N={N}")
        for step in pbar:
            # collect episodes
            collect_episodes_fill(HatsEnv, max(8, hp.envs_per_fill // 64), N, agent, epsilon)
            # run several updates
            updates_per_loop = 4
            for _ in range(updates_per_loop):
                agent.sample_and_update()

            # log success rate every 50 steps
            if step % 50 == 0:
                mean_surv = evaluate_policy(agent, N, 50)  # quick eval
                success_rate = mean_surv / N
                history.append((global_ep, N, success_rate))
            global_ep += 1

            # periodic checkpoint
            if step % hp.print_every_steps == 0 and step > start_step:
                pbar.set_postfix({
                    'train_step': agent.train_step,
                    'replay_eps': len(agent.replay)
                })
                save_checkpoint(agent, hp, ci, step)

        # end-of-stage evaluation & checkpoint
        mean_surv = evaluate_policy(agent, N, hp.eval_episodes)
        print(f"After N={N} stage evaluation mean survivors: {mean_surv:.3f}")
        survivors_per_stage.append(mean_surv)
        save_checkpoint(agent, hp, ci, steps)

        in_stage_start_step = 0
        agent = None  # re-init for next N

    # final per-stage plot
    plt.figure(figsize=(8,5))
    plt.plot(hp.curriculum, survivors_per_stage, marker='o')
    plt.xlabel('Number of Prisoners (N)')
    plt.ylabel('Average Survivors')
    plt.title('DDQRN Hats: Average Survivors per Curriculum Stage')
    plt.grid(True)
    plt.show()

    # ---- CHANGED: single graph with smooth curves for all curriculums over episodes ----
    def _smooth(y_vals):
        """Centered moving average with an automatically chosen odd window size."""
        y = np.array(y_vals, dtype=np.float32)
        if len(y) < 5:
            return y
        # window ~ 5% of series length, make it odd and at least 5
        w = max(5, int(round(len(y) * 0.05)))
        if w % 2 == 0:
            w += 1
        kernel = np.ones(w, dtype=np.float32) / w
        # pad edges to keep length
        pad = w // 2
        y_pad = np.pad(y, (pad, pad), mode='edge')
        y_smooth = np.convolve(y_pad, kernel, mode='valid')
        return y_smooth

    plt.figure(figsize=(10,6))
    for N in hp.curriculum:
        x_vals = [ep for ep, n, _ in history if n == N]
        y_vals = [sr for _, n, sr in history if n == N]
        if len(x_vals) == 0:
            continue
        y_smooth = _smooth(y_vals)
        # Keep the same x-axis indexing after smoothing
        plt.plot(x_vals, y_smooth, label=f"N={N}")
    plt.xlabel("Global Episode")
    plt.ylabel("Success Rate")
    plt.title("DDQRN Hats: Smoothed Success Rate over Episodes (All Curriculums)")
    plt.legend()
    plt.grid(True)
    plt.show()
    # ---- END CHANGE ----

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
        print("[WARNING] Could not set thread execution state; continuing.")

    hp = HParams()
    try:
        train_full(hp)
    finally:
        # Clear the sleep-prevention request so OS can sleep normally again
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
        except Exception:
            pass
 