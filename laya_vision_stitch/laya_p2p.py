"""One neural graph: Laya goal features -> learned bridge -> pretrained P2P policy."""

import json
import shutil
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .p2p_pretrained_policy import OpenP2PPolicy
from .reference_stitch import LAYA_MODELS, load_laya


class GoalBridge(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.mean, self.scale = mx.zeros(width), mx.ones(width)
        self.projection = nn.Linear(width, 768)

    def __call__(self, x):
        return self.projection((x.astype(mx.float32) - self.mean) / self.scale)


class LayaP2P(nn.Module):
    def __init__(self, laya, policy):
        super().__init__()
        self.laya, self.policy = laya, policy
        self.bridge = GoalBridge(laya.encoder.config.hidden_size)
        self.laya.freeze()
        self.policy.freeze()
        self.bridge.freeze(keys=["mean", "scale"], recurse=False)
        self.eval()

    def goal_features(self, ids, mask):
        h = self.laya.encoder(ids, mask)
        h = h + self.laya.type_emb(mx.zeros((ids.shape[0],), mx.int32))[:, None]
        h = self.laya.head(h, mask[:, None, None].astype(mx.bool_))
        weights = mask.astype(mx.float32)[..., None]
        return (h.astype(mx.float32) * weights).sum(1) / weights.sum(1)

    def step(self, pixels, goal_ids, goal_mask, caches=None, position=0, temperature=0.0):
        goal = self.bridge(self.goal_features(goal_ids, goal_mask))
        return self.policy.step(
            pixels, text=goal, caches=caches, position=position, temperature=temperature
        )


class LayaP2PRuntime:
    def __init__(self, model, agent, metadata):
        self.model, self.agent, self.metadata = model, agent, metadata

    @classmethod
    def build(cls, policy_bundle):
        agent = load_laya("english")
        policy = OpenP2PPolicy()
        policy.load_weights(str(Path(policy_bundle) / "model.safetensors"), strict=True)
        policy.freeze()
        model = LayaP2P(agent.model, policy)
        mx.eval(model.parameters())
        return cls(
            model,
            agent,
            {
                "format": "laya-p2p-1",
                "laya_model": LAYA_MODELS["english"][0],
                "laya_revision": LAYA_MODELS["english"][1],
                "encoder_config": agent.encoder_cfg,
                "agent_config": agent.cfg,
                "policy_source": json.loads((Path(policy_bundle) / "config.json").read_text()),
                "trained_goal_bridge": False,
                "deployment_eligible": False,
                "scope": "Laya encoder and decision-head features condition the pretrained P2P image/action policy. No inherited full Qwen reasoning is claimed.",
            },
        )

    def prepare_goal(self, goal):
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError("Provide a nonempty goal")
        ids = self.agent.tok("Goal: " + goal)["input_ids"]
        if len(ids) > self.agent.cfg["max_len"]:
            raise ValueError("Goal exceeds Laya context; no silent truncation")
        return mx.array([ids], mx.int32), mx.ones((1, len(ids)), mx.bool_)

    def save(self, output):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        self.model.save_weights(str(output / "model.safetensors"))
        (output / "config.json").write_text(json.dumps(self.metadata, indent=2) + "\n")
        shutil.copytree(self.agent.model_dir / "tokenizer", output / "tokenizer")

    @classmethod
    def load(cls, directory):
        from laya_mlx.agent import Agent
        from laya_mlx.model import DecisionModel, EncoderConfig
        from laya_mlx.tokenizer import Tokenizer

        directory = Path(directory)
        meta = json.loads((directory / "config.json").read_text())
        if meta["format"] != "laya-p2p-1":
            raise ValueError("Unsupported stitched checkpoint")
        laya = DecisionModel(EncoderConfig.from_dict(meta["encoder_config"]), meta["agent_config"])
        model = LayaP2P(laya, OpenP2PPolicy())
        if meta.get("control_adapter"):
            from .p2p_adaptation import install_control_adapter

            install_control_adapter(model, **meta["control_adapter"])
        if meta.get("visual_adapter"):
            from .p2p_adaptation import install_visual_adapter

            install_visual_adapter(model, **meta["visual_adapter"])
        if meta.get("spatial_adapter"):
            from .spatial_goal_adapter import install_spatial_adapter

            install_spatial_adapter(model, **meta["spatial_adapter"])
        model.load_weights(str(directory / "model.safetensors"), strict=True)
        mx.eval(model.parameters())
        agent = Agent.__new__(Agent)
        agent.model, agent.cfg, agent.encoder_cfg = (
            laya,
            meta["agent_config"],
            meta["encoder_config"],
        )
        agent.tok, agent.model_dir, agent._prefix_cache = (
            Tokenizer(directory / "tokenizer"),
            directory,
            None,
        )
        return cls(model, agent, meta)
