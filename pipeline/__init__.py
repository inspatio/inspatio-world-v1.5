__all__ = ["CausalInferencePipeline"]


def __getattr__(name):
    if name == "CausalInferencePipeline":
        from .causal_inference import CausalInferencePipeline

        return CausalInferencePipeline
    raise AttributeError(name)
