"""Behavior cloning warm-start for MaskablePPO poker policies."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from ppo_setup.env import HeadsUpPokerEnv
from ppo_setup.feature_encoder import ACTION_TO_ID
from ppo_setup.opponents import make_opponent, resolve_opponent_pool


@dataclass
class ExpertDataset:
    observations: np.ndarray
    masks: np.ndarray
    actions: np.ndarray


def collect_expert_examples(
    expert_pool: str,
    opponent_pool: str,
    examples: int,
    rounds: int,
    initial_stack: int,
    small_blind: int,
    ante: int,
    equity_sims: int,
    seed: int,
    progress_every: int = 1000,
) -> ExpertDataset:
    expert_names = resolve_opponent_pool(expert_pool)
    opponent_names = resolve_opponent_pool(opponent_pool)
    if not expert_names:
        raise ValueError("No behavior-cloning experts resolved from %r" % expert_pool)
    if not opponent_names:
        raise ValueError("No behavior-cloning opponents resolved from %r" % opponent_pool)

    rng = random.Random(seed)
    obs_rows: list[np.ndarray] = []
    mask_rows: list[np.ndarray] = []
    action_rows: list[int] = []
    game_idx = 0

    while len(action_rows) < examples:
        expert = make_opponent(rng.choice(expert_names))
        env = HeadsUpPokerEnv(
            opponent_pool=opponent_names,
            max_rounds=rounds,
            initial_stack=initial_stack,
            small_blind=small_blind,
            ante=ante,
            equity_sims=equity_sims,
            randomize_position=bool(game_idx % 2),
            hero_observer=expert,
        )
        obs, _info = env.reset(seed=seed + game_idx)
        terminated = False
        truncated = False
        while not (terminated or truncated) and len(action_rows) < examples:
            ask = env.current_ask
            if ask is None:
                break
            mask = env.action_masks()
            action_name = _expert_action(expert, ask, mask)
            action_id = ACTION_TO_ID[action_name]
            obs_rows.append(obs.astype(np.float32, copy=True))
            mask_rows.append(mask.astype(bool, copy=True))
            action_rows.append(action_id)
            obs, _reward, terminated, truncated, _info = env.step(action_id)
            if progress_every > 0 and len(action_rows) % progress_every == 0:
                print("[bc] collected %d/%d examples" % (len(action_rows), examples), flush=True)
        env.close()
        game_idx += 1

    return ExpertDataset(
        observations=np.stack(obs_rows).astype(np.float32),
        masks=np.stack(mask_rows).astype(bool),
        actions=np.asarray(action_rows, dtype=np.int64),
    )


def behavior_clone_model(
    model,
    dataset: ExpertDataset,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> None:
    import torch
    import torch.nn.functional as functional

    policy = model.policy
    device = policy.device
    obs = torch.as_tensor(dataset.observations, dtype=torch.float32, device=device)
    masks = torch.as_tensor(dataset.masks, dtype=torch.bool, device=device)
    actions = torch.as_tensor(dataset.actions, dtype=torch.long, device=device)

    params = list(policy.mlp_extractor.policy_net.parameters()) + list(policy.action_net.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    n = int(actions.shape[0])
    batch_size = max(1, min(int(batch_size), n))

    policy.train()
    for epoch in range(max(1, int(epochs))):
        order = torch.randperm(n, device=device)
        total_loss = 0.0
        total_correct = 0
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            logits = _policy_logits(policy, obs[idx])
            masked_logits = logits.masked_fill(~masks[idx], -1.0e9)
            loss = functional.cross_entropy(masked_logits, actions[idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach().cpu()) * int(idx.shape[0])
            total_correct += int((masked_logits.argmax(dim=1) == actions[idx]).sum().detach().cpu())
        print(
            "[bc] epoch %d/%d loss=%.5f acc=%.2f%%"
            % (epoch + 1, epochs, total_loss / n, 100.0 * total_correct / n),
            flush=True,
        )
    policy.eval()


def _policy_logits(policy, obs_tensor):
    features = policy.extract_features(obs_tensor)
    if isinstance(features, tuple):
        features = features[0]
    if hasattr(policy.mlp_extractor, "forward_actor"):
        latent_pi = policy.mlp_extractor.forward_actor(features)
    else:
        latent_pi, _latent_vf = policy.mlp_extractor(features)
    return policy.action_net(latent_pi)


def _expert_action(expert, ask: dict, mask: np.ndarray) -> str:
    response = expert.respond_to_ask(ask)
    if isinstance(response, tuple):
        response = response[0]
    legal = {name for name, idx in ACTION_TO_ID.items() if bool(mask[idx])}
    if response in legal:
        return response
    if "call" in legal:
        return "call"
    if "fold" in legal:
        return "fold"
    return next(iter(legal), "fold")
