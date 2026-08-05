from typing import Any


def resolve_prior(
    *,
    model_config: dict[str, Any] | None,
    param_name: str,
    default_dist: Any,
    default_kwargs: dict[str, Any],
    dims: str | None = None,
    shape: int | None = None,
) -> Any:
    """
    Create a PyMC prior / distribution named ``param_name`` from ``model_config`` or defaults.

    Config entry shape: ``{param_name: {"dist": DistClass, "kwargs": {...}}}``.
    Missing / ``None`` entries use ``default_dist`` and ``default_kwargs``.
    """
    if (
        not model_config
        or param_name not in model_config
        or model_config[param_name] is None
    ):
        dist = default_dist
        kwargs = dict(default_kwargs)
    else:
        cfg = model_config[param_name]
        if not isinstance(cfg, dict):
            raise ValueError(
                f"model_config[{param_name!r}] must be a dict with keys 'dist' and 'kwargs'"
            )
        if "dist" not in cfg:
            raise ValueError(
                f"model_config[{param_name!r}] is missing required key 'dist'"
            )
        dist = cfg["dist"]
        kwargs = dict(default_kwargs)
        kwargs.update(cfg.get("kwargs", {}))

    call_kwargs: dict[str, Any] = dict(kwargs)
    if dims is not None:
        call_kwargs["dims"] = dims
    if shape is not None:
        call_kwargs["shape"] = shape

    return dist(param_name, **call_kwargs)
