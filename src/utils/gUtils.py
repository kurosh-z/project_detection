from pathlib import Path
import os
import datetime
from torch.utils.tensorboard import SummaryWriter

# logger = logging.getLogger(__name__)


def mkdir_nested(full_path, mode=0o775):
    """creates neste path

    Args:
        full_path (string): path which should be created
        mode (int, optional): Posix permission for created directories. Defaults to 0o775.
    """
    p = Path(full_path)
    p.mkdir(mode=mode, exist_ok=True, parents=True)


class Logger(object):
    def __init__(self, log_dir, log_hist=True):
        """Create a summary writer logging to log_dir."""
        if log_hist:  # Check a new folder for each log should be dreated
            log_dir = os.path.join(log_dir, datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S"))
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        self.writer.add_scalar(tag, value, step)

    def list_of_scalars_summary(self, tag_value_pairs, step):
        """Log scalar variables."""
        for tag, value in tag_value_pairs:
            self.writer.add_scalar(tag, value, step)
