from dataclasses import dataclass

@dataclass
class EdgeResult:
    width: float
    credit: float
    edge: float
    used_mid: str   # 'model' | 'raw'
    reason: str     # e.g., 'model_rmse_ok', 'model_unavailable'

def compute_edge(width: float,
                 model_mid_credit: float | None,
                 raw_mid_credit: float | None,
                 model_ok: bool) -> EdgeResult:
    if model_ok and model_mid_credit is not None:
        credit = float(model_mid_credit)
        return EdgeResult(
            width=width,
            credit=credit,
            edge=(credit/width) if width else 0.0,
            used_mid="model",
            reason="model_rmse_ok",
        )
    credit = float(raw_mid_credit or 0.0)
    return EdgeResult(
        width=width,
        credit=credit,
        edge=(credit/width) if width else 0.0,
        used_mid="raw",
        reason="model_unavailable_or_untrusted",
    )
