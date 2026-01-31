# src/aoe/config.py
from __future__ import annotations
import argparse, json, os, pathlib, typing as t
from dataclasses import dataclass
import yaml

@dataclass
class AOEConfig:
    raw: dict

    @classmethod
    def load(cls, path: str, cli_overrides: t.Dict[str, t.Any] | None = None) -> "AOEConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        cli_overrides = cli_overrides or {}
        # Shallow override: support --start/--end/--universe/--strategy/--dte etc.
        data.setdefault("data", {})
        data.setdefault("strategy", {})
        data["data"]["start"] = cli_overrides.get("start", data["data"].get("start"))
        data["data"]["end"]   = cli_overrides.get("end",   data["data"].get("end"))
        if "universe" in cli_overrides and cli_overrides["universe"]:
            data["data"]["universe"] = cli_overrides["universe"].split(",")
        if "strategy" in cli_overrides and cli_overrides["strategy"]:
            data["strategy"]["kind"] = cli_overrides["strategy"]
        if "dte" in cli_overrides and cli_overrides["dte"]:
            data["strategy"]["dte"] = int(cli_overrides["dte"])
        if "p_tail" in cli_overrides and cli_overrides["p_tail"]:
            data["strategy"]["p_tail"] = float(cli_overrides["p_tail"])
        if "wing_width" in cli_overrides and cli_overrides["wing_width"]:
            data["strategy"]["wing_width"] = float(cli_overrides["wing_width"])
        return cls(raw=data)

    def dump_to(self, path: str) -> None:
        pathlib.Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.raw, f)

    def __getitem__(self, k): return self.raw[k]
    def get(self, k, d=None): return self.raw.get(k, d)

def add_common_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--start", type=str)
    p.add_argument("--end", type=str)
    p.add_argument("--universe", type=str)   # CSV list
    p.add_argument("--strategy", type=str, choices=["condor","debit","both"])
    p.add_argument("--dte", type=int)
    p.add_argument("--p_tail", type=float)
    p.add_argument("--wing_width", type=float)
    return p
