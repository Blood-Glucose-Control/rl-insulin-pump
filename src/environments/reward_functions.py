from simglucose.analysis.risk import risk_index  # type: ignore


def custom_reward_fn(BG_last_hour):
    """Calculate the reward based on the risk index difference."""
    if len(BG_last_hour) < 2:
        return 0
    _, _, risk_current = risk_index([BG_last_hour[-1]], 1)
    _, _, risk_prev = risk_index([BG_last_hour[-2]], 1)
    return risk_current - risk_prev
