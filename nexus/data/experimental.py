import os
import json
import torch

CHANNEL_NAMES = ["Nav", "Kir", "K_leak", "Ca", "Cl", "NaKATP", "HKATP", "VATP"]

def load_experimental_records(path: str) -> dict:
    with open(path) as f:
        return json.load(f)

def parse_operation(op: str) -> tuple:
    kind, value = op.split(":", 1)
    if kind == "zero_channel":
        if value not in CHANNEL_NAMES:
            raise ValueError("unknown channel: " + value)
        return ("zero_channel", CHANNEL_NAMES.index(value))
    if kind == "scale_gj":
        return ("scale_gj", float(value))
    raise ValueError("unknown operation kind: " + kind)

def apply_operation(data, op: str):
    kind, value = parse_operation(op)
    out = data.clone()
    if kind == "zero_channel":
        out.x = out.x.clone()
        out.x[:, value] = 0.0
    else:
        out.edge_attr = out.edge_attr.clone() * value
    return out
