class PI0Pytorch:
    """Placeholder for the PyTorch implementation of the Pi0 model.

    This stub simply records the config it was initialized with. Any attempt
    to run forward passes should be replaced with the real implementation
    from upstream OpenPI if PyTorch inference is required.
    """

    def __init__(self, config):
        self.config = config

    def __repr__(self) -> str:  # pragma: no cover - trivial helper
        return f"PI0Pytorch(config={self.config!r})"
