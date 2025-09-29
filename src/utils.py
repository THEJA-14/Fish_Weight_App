def safe_float(x, default=float('nan')):
    try:
        return float(x)
    except Exception:
        return default
