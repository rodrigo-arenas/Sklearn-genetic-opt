import logging
import os
import sys
import time
from pathlib import Path
from stat import S_IMODE, S_ISREG
from tempfile import TemporaryDirectory
from joblib import dump

from tqdm.auto import tqdm

from .base import BaseCallback
from ..parameters import Metrics
from ..evaluation import candidate_records

logger = logging.getLogger(__name__)  # noqa


class ProgressBar(BaseCallback):
    """Displays a tqdm progress bar with the training progress."""

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        kwargs: dict, default = {"file": sys.stdout}
            A dict with valid arguments from tqdm.auto.tqdm
        """
        if not kwargs.get("file"):
            kwargs["file"] = sys.stdout

        self.kwargs = kwargs
        self.progress_bar = None

    def on_start(self, estimator=None):
        """Initializes the progress bar with the kwargs and total generations"""
        self.kwargs["total"] = estimator._n_iterations
        self.progress_bar = tqdm(**self.kwargs)
        self.progress_bar.update(1)

    def on_step(self, record=None, logbook=None, estimator=None):
        """Increases the progress bar by one step"""
        self.progress_bar.update(1)

    def on_end(self, logbook=None, estimator=None):
        """Closes the progress bar"""
        self.progress_bar.close()


class LogbookSaver(BaseCallback):
    """Save candidate records and replace a regular file only after a successful write.

    Regular files also include rows collected during DEAP's final local refinement.
    """

    def __init__(self, checkpoint_path, **dump_options):
        """
        Parameters
        ----------
        checkpoint_path: str, os.PathLike or binary file
            Destination for candidate records. A regular file path needs a
            writable parent directory for atomic replacement.
        dump_options: dict
            Keyword arguments forwarded to joblib :func:`~joblib.dump`.
        """

        self.checkpoint_path = checkpoint_path
        self.dump_options = dump_options

    def on_start(self, estimator=None):
        self._last_path_attempt_rows = None

    def on_step(self, record=None, logbook=None, estimator=None):
        self._save(estimator)
        return False

    def on_end(self, logbook=None, estimator=None):
        if (
            logbook
            and logbook[-1].get("local_refinements")
            and getattr(self, "_last_path_attempt_rows", None) is not None
        ):
            self._save(estimator, only_new_path_rows=True)

    def _save(self, estimator, *, only_new_path_rows=False):
        try:
            records = candidate_records(estimator)
            if only_new_path_rows and len(records) <= self._last_path_attempt_rows:
                return
            if isinstance(self.checkpoint_path, (str, os.PathLike)):
                self._save_path(records, Path(self.checkpoint_path), only_new_path_rows)
            elif not only_new_path_rows:
                dump(records, self.checkpoint_path, **self.dump_options)
        except Exception as error:
            logger.error(
                "Could not save the Logbook in the checkpoint %s: %s",
                self.checkpoint_path,
                error,
            )

    def _save_path(self, records, path, only_new_path_rows):
        destination = path.resolve()
        try:
            mode = destination.stat().st_mode
        except FileNotFoundError:
            mode = None
        if mode is not None and not S_ISREG(mode):
            if not only_new_path_rows:
                dump(records, path, **self.dump_options)
            return

        self._last_path_attempt_rows = len(records)
        with TemporaryDirectory(dir=destination.parent) as directory:
            # Keep the original name so joblib still selects compression by suffix.
            temporary = Path(directory) / path.name
            dump(records, temporary, **self.dump_options)
            if mode is not None:
                temporary.chmod(S_IMODE(mode))
            temporary.replace(destination)


class TensorBoard(BaseCallback):
    """Log all the fitness metrics to Tensorboard into log_dir/run_id folder"""

    def __init__(self, log_dir="./logs", run_id=None):
        """
        Parameters
        ----------
        log_dir: str, default="./logs"
            Path to the main folder where the data will be log
        run_id: str, default=None
            Subfolder where the data will be log, if None it will create a folder
            with the current datetime with format time.strftime("%Y_%m_%d-%H_%M_%S")
        """
        self.log_dir = log_dir

        if run_id is None:
            self.run_id = time.strftime("%Y_%m_%d-%H_%M_%S")
        else:
            self.run_id = run_id

        self.path = os.path.join(log_dir, self.run_id)

    def on_step(self, record=None, logbook=None, estimator=None):
        try:
            import tensorflow as tf
        except ModuleNotFoundError as exc:
            if exc.name != "tensorflow":
                raise
            raise ImportError(
                "The legacy TensorBoard callback requires TensorFlow installed separately. "
                "Use sklearn_genetic.plots.LivePlot for lightweight live charts."
            ) from exc

        # Get the last metric value
        stats = logbook[-1]

        # Create logs files placeholder
        writer = tf.summary.create_file_writer(self.path)

        # Log the metrics
        with writer.as_default():
            for metric in Metrics.list():
                tf.summary.scalar(name=metric, data=stats[metric], step=stats["gen"])
        writer.flush()

        return False
