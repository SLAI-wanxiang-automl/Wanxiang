__all__ = ["AsyncEneygyExpert", "EneygyExpert", "EneygyExpertForecaster"]


def __getattr__(name):
    """Keep standalone rolling adapters independent of optional LLM imports."""
    if name in {"AsyncEneygyExpert", "EneygyExpert"}:
        from .agent import AsyncEneygyExpert, EneygyExpert

        return {"AsyncEneygyExpert": AsyncEneygyExpert, "EneygyExpert": EneygyExpert}[name]
    if name == "EneygyExpertForecaster":
        from .forecaster import EneygyExpertForecaster

        return EneygyExpertForecaster
    raise AttributeError(name)
