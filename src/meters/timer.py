import time
from dataclasses import dataclass, field
from typing import Callable, ClassVar, Dict, Optional, Any
from contextlib import ContextDecorator
from torch.utils.tensorboard import SummaryWriter


class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""


@dataclass
class Timer:
    timers: ClassVar[Dict[str, float]] = {}
    tag: Optional[str] = None
    global_step: Optional[int] = None
    writer: Optional[SummaryWriter] = None
    _start_time: Optional[float] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """Initialization: add timer to dict of timers"""
        if self.tag:
            self.timers.setdefault(self.tag, 0)

    def start(self) -> None:
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self) -> float:
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        # Calculate elapsed time
        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None

        # Report elapsed time
        if self.writer:
            assert self.tag is not None, TimerError(
                "Tag needs to given if writer is not None."
            )
            self.writer.add_scalar(
                tag=self.tag, scalar_value=elapsed_time, global_step=self.global_step
            )

        # When tagged the elapsed time is accumulated
        if self.tag:
            self.timers[self.tag] += elapsed_time

        return elapsed_time

    def update_step(self, new_step: int) -> None:
        self.global_step = new_step

    def __enter__(self):
        """Start a new timer as a context manager"""
        self.start()
        return self

    def __exit__(self, *exc_info):
        """Stop the context manager timer"""
        self.stop()
