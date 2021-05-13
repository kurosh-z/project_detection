import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


def mkdir_nested(full_path, mode=0o775):
    """creates neste path

    Args:
        full_path (string): path which should be created
        mode (int, optional): Posix permission for created directories. Defaults to 0o775.
    """
    p = Path(full_path)
    p.mkdir(mode=mode, exist_ok=True, parents=True)
