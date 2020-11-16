# timer.py

from dataclasses import dataclass, field
import time
from typing import Callable, ClassVar, Dict, Optional
import logging

class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""

@dataclass
class Timer:
    timers: ClassVar[Dict[str, float]] = dict()
    name: Optional[str] = None
    text: str = "Elapsed time: {:0.4f} seconds"
    logger: Optional[Callable[[str], None]] = None # print
    _start_time: Optional[float] = field(default=None, init=False, repr=False)
    average: Optional[float] = field(default=0, init=True, repr=False)
    total : Optional[float] = field(default=0, init=True, repr=False)
    _start_count:  Optional[float] = field(default=0, init=True, repr=False)
    time_log: Optional[list] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        """Add timer to dict of timers after initialization"""
        if self.name is not None:
            self.timers.setdefault(self.name, 0)

    def start(self) -> None:
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self) -> None:
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        # Calculate elapsed time
        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        self._start_count += 1

        # Report elapsed time
        if self.logger:
            self.logger(self.text.format(elapsed_time))
        if self.name:
            self.timers[self.name] += elapsed_time

        self.time_log.append(elapsed_time)
        self.total = self.total + elapsed_time
        self.average = self.total / self._start_count

        return elapsed_time


if __name__ == "__main__":
    Timer(logger=logging.INFO)

    # Each Timer is object is singleton
    for j in range(1, 3, 1):
        for i in range(3):
            t = Timer("subtimer {}".format(str(i)))
            t.start()
            time.sleep(0.1 * i)
            t.stop()

    print(Timer.timers)