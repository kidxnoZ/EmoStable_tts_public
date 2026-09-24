import logging

def get_pylogger(name: str = __name__) -> logging.Logger:
    """Return a standard logger for the retained inference-only subset."""
    return logging.getLogger(name)
