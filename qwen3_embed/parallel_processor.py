import logging
import os
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum
from multiprocessing import Queue, get_context
from multiprocessing.context import BaseContext
from multiprocessing.process import BaseProcess
from multiprocessing.sharedctypes import Synchronized as BaseValue
from queue import Empty
from typing import Any

from qwen3_embed.common.types import Device

# Single item should be processed in less than:
processing_timeout = 10 * 60  # seconds

max_internal_batch_size = 200


class QueueSignals(StrEnum):
    stop = "stop"
    confirm = "confirm"
    error = "error"


class Worker:
    @classmethod
    def start(cls, *args: Any, **kwargs: Any) -> "Worker":
        raise NotImplementedError()

    def process(self, items: Iterable[tuple[int, Any]]) -> Iterable[tuple[int, Any]]:
        raise NotImplementedError()


def _get_items_from_queue(input_queue: Queue) -> Iterable[Any]:
    while True:
        item = input_queue.get()
        if item == QueueSignals.stop:
            break
        yield item


def _cleanup_worker(
    input_queue: Queue, output_queue: Queue, num_active_workers: BaseValue, worker_id: int
) -> None:
    # It's important that we close and join the queue here before
    # decrementing num_active_workers. Otherwise our parent may join us
    # before the queue's feeder thread has passed all buffered items to
    # the underlying pipe resulting in a deadlock.
    #
    # See:
    # https://docs.python.org/3.6/library/multiprocessing.html?highlight=process#pipes-and-queues
    # https://docs.python.org/3.6/library/multiprocessing.html?highlight=process#programming-guidelines
    try:
        for action, name in [
            (input_queue.close, "close input_queue"),
            (output_queue.close, "close output_queue"),
            (input_queue.join_thread, "join_thread input_queue"),
            (output_queue.join_thread, "join_thread output_queue"),
        ]:
            try:
                action()
            except Exception as e:
                logging.exception(f"Reader worker {worker_id} failed to {name}: {e}")
    finally:
        with num_active_workers.get_lock():
            num_active_workers.value -= 1

    logging.info(f"Reader worker {worker_id} finished")


def _worker(
    worker_class: type[Worker],
    input_queue: Queue,
    output_queue: Queue,
    num_active_workers: BaseValue,
    worker_id: int,
    kwargs: dict[str, Any] | None = None,
) -> None:
    """
    A worker that pulls data pints off the input queue, and places the execution result on the output queue.
    When there are no data pints left on the input queue, it decrements
    num_active_workers to signal completion.
    """

    if kwargs is None:
        kwargs = {}

    logging.info(
        f"Reader worker: {worker_id} PID: {os.getpid()} Device: {kwargs.get('device_id', 'CPU')}"
    )

    try:
        try:
            worker = worker_class.start(**kwargs)
        except Exception as e:  # pylint: disable=broad-except
            logging.exception(f"Worker {worker_id} failed to start: {e}")
            output_queue.put(QueueSignals.error)
            raise

        try:
            for processed_item in worker.process(_get_items_from_queue(input_queue)):
                output_queue.put(processed_item)
        except Exception as e:  # pylint: disable=broad-except
            logging.exception(f"Worker {worker_id} failed during processing: {e}")
            output_queue.put(QueueSignals.error)
            raise
    finally:
        _cleanup_worker(input_queue, output_queue, num_active_workers, worker_id)


@dataclass
class PoolConfig:
    num_workers: int
    start_method: str | None = None
    device_ids: list[int] | None = None
    cuda: bool | Device = Device.AUTO


class ParallelWorkerPool:
    def __init__(
        self,
        worker: type[Worker],
        config: PoolConfig | None = None,
    ):
        if config is None:
            config = PoolConfig(num_workers=1)

        self.worker_class = worker
        self.num_workers = config.num_workers
        self.input_queue: Queue | None = None
        self.output_queue: Queue | None = None
        self.ctx: BaseContext = get_context(config.start_method)
        self.processes: list[BaseProcess] = []
        self.queue_size = self.num_workers * max_internal_batch_size
        self.emergency_shutdown = False
        self.device_ids = config.device_ids
        self.cuda = config.cuda
        self.num_active_workers: BaseValue | None = None

    def start(self, **kwargs: Any) -> None:
        self.input_queue = self.ctx.Queue(self.queue_size)
        self.output_queue = self.ctx.Queue(self.queue_size)

        ctx_value = self.ctx.Value("i", self.num_workers)
        assert isinstance(ctx_value, BaseValue)
        self.num_active_workers = ctx_value

        for worker_id in range(0, self.num_workers):
            worker_kwargs = deepcopy(kwargs)
            if self.device_ids:
                device_id = self.device_ids[worker_id % len(self.device_ids)]
                worker_kwargs["device_id"] = device_id
                worker_kwargs["cuda"] = self.cuda

            assert hasattr(self.ctx, "Process")
            process = self.ctx.Process(  # type: ignore[call-non-callable]
                target=_worker,
                args=(
                    self.worker_class,
                    self.input_queue,
                    self.output_queue,
                    self.num_active_workers,
                    worker_id,
                    worker_kwargs,
                ),
            )
            process.start()
            self.processes.append(process)

    def ordered_map(self, stream: Iterable[Any], *args: Any, **kwargs: Any) -> Iterable[Any]:
        buffer: dict[int, Any] = {}
        next_expected = 0
        sentinel = object()

        for idx, item in self.semi_ordered_map(stream, *args, **kwargs):
            # ⚡ Bolt: Fast path for in-order items and dict.pop to combine lookup and removal
            if idx == next_expected:
                yield item
                next_expected += 1
                while (buffered_item := buffer.pop(next_expected, sentinel)) is not sentinel:
                    yield buffered_item
                    next_expected += 1
            else:
                buffer[idx] = item

    def semi_ordered_map(
        self, stream: Iterable[Any], *args: Any, **kwargs: Any
    ) -> Iterable[tuple[int, Any]]:
        try:
            self.start(**kwargs)
            yield from self._process_stream(stream)
        except BaseException:
            # BaseException, not Exception, because the most common way out of
            # this generator is GeneratorExit -- the consumer stopped reading,
            # by `break` or because its own body raised. GeneratorExit does not
            # derive from Exception, so `except Exception` left
            # emergency_shutdown False and the finally below called the
            # unbounded join(), while the workers were still blocked in
            # output_queue.join_thread() waiting to flush into a pipe nobody
            # drains: the exact deadlock _cleanup_worker's comment warns about.
            # Both sides then waited forever. Every early exit means workers may
            # hold unread output, so every early exit is an emergency shutdown.
            self.emergency_shutdown = True
            raise
        finally:
            if self.emergency_shutdown:
                self.join_or_terminate()
            else:
                self.join()

            if self.input_queue is not None:
                self.input_queue.close()
                if self.emergency_shutdown:
                    self.input_queue.cancel_join_thread()
                else:
                    self.input_queue.join_thread()

            if self.output_queue is not None:
                self.output_queue.close()
                if self.emergency_shutdown:
                    self.output_queue.cancel_join_thread()
                else:
                    self.output_queue.join_thread()

    def _process_stream(self, stream: Iterable[Any]) -> Iterable[tuple[int, Any]]:
        assert self.input_queue is not None, "Input queue was not initialized"
        assert self.output_queue is not None, "Output queue was not initialized"

        self.check_worker_health()

        pushed = 0
        read = 0
        for idx, item in enumerate(stream):
            self.check_worker_health()
            if pushed - read < self.queue_size:
                try:
                    out_item = self.output_queue.get_nowait()
                except Empty:
                    out_item = None
            else:
                try:
                    out_item = self.output_queue.get(timeout=processing_timeout)
                except Empty as e:
                    self.join_or_terminate()
                    raise e

            if out_item is not None:
                if out_item == QueueSignals.error:
                    self.join_or_terminate()
                    raise RuntimeError("Thread unexpectedly terminated")
                yield out_item
                read += 1

            self.input_queue.put((idx, item))
            pushed += 1

        for _ in range(self.num_workers):
            self.input_queue.put(QueueSignals.stop)

        while read < pushed:
            self.check_worker_health()
            out_item = self.output_queue.get(timeout=processing_timeout)
            if out_item == QueueSignals.error:
                self.join_or_terminate()
                raise RuntimeError("Thread unexpectedly terminated")
            yield out_item
            read += 1

        self.check_worker_health()

    def check_worker_health(self) -> None:
        """
        Checks if any worker process has terminated unexpectedly
        """
        for process in self.processes:
            if not process.is_alive() and process.exitcode != 0:
                self.emergency_shutdown = True
                self.join_or_terminate()
                raise RuntimeError(
                    f"Worker PID: {process.pid} terminated unexpectedly with code {process.exitcode}"
                )

    def join_or_terminate(self, timeout: int = 1) -> None:
        """
        Emergency shutdown
        @param timeout:
        @return:
        """
        for process in self.processes:
            process.join(timeout=timeout)
            if process.is_alive():
                process.terminate()
        self.processes.clear()

    def join(self) -> None:
        try:
            for process in self.processes:
                process.join()
            self.check_worker_health()
        finally:
            self.processes.clear()

    def __del__(self) -> None:
        """
        Terminate processes if the user hasn't joined. This is necessary as
        leaving stray processes running can corrupt shared state. In brief,
        we've observed shared memory counters being reused (when the memory was
        free from the perspective of the parent process) while the stray
        workers still held a reference to them.
        For a discussion of using destructors in Python in this manner, see
        https://eli.thegreenplace.net/2009/06/12/safely-using-destructors-in-python/.
        """
        for process in self.processes:
            if process.is_alive():
                process.terminate()
