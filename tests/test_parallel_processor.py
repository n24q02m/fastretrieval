import time
from collections.abc import Iterable
from multiprocessing.sharedctypes import Synchronized as BaseValue
from queue import Empty
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import qwen3_embed.parallel_processor as pp_module
from qwen3_embed.parallel_processor import (
    ParallelWorkerPool,
    PoolConfig,
    QueueSignals,
    Worker,
    _cleanup_worker,
    _get_items_from_queue,
    _worker,
)

# --- Helper Workers ---


class SquareWorker(Worker):
    """
    A simple worker that squares integers.
    """

    @classmethod
    def start(cls, **kwargs: Any) -> "SquareWorker":
        return cls()

    def process(self, items: Iterable[tuple[int, Any]]) -> Iterable[tuple[int, Any]]:
        for idx, item in items:
            yield idx, item * item


class FailingWorker(Worker):
    """
    A worker that raises an exception on a specific input value.
    """

    def __init__(self, failure_val: int):
        self.failure_val = failure_val

    @classmethod
    def start(cls, **kwargs: Any) -> "FailingWorker":
        failure_val = kwargs.get("failure_val", -1)
        return cls(failure_val)

    def process(self, items: Iterable[tuple[int, Any]]) -> Iterable[tuple[int, Any]]:
        for idx, item in items:
            if item == self.failure_val:
                raise ValueError(f"Intentional failure on {item}")
            yield idx, item


class SlowWorker(Worker):
    """
    A worker that sleeps a bit to simulate work.
    """

    @classmethod
    def start(cls, **kwargs: Any) -> "SlowWorker":
        return cls()

    def process(self, items: Iterable[tuple[int, Any]]) -> Iterable[tuple[int, Any]]:
        for idx, item in items:
            time.sleep(0.001)
            yield idx, item


class BulkWorker(Worker):
    """
    A worker whose results are far larger than a pipe buffer.

    Payload size is what makes an abandoned consumer deadlock: small results are
    handed to the OS pipe and the feeder thread finishes regardless of whether
    anyone reads them, so a worker exits cleanly and nothing hangs. Only once the
    pipe is full does the feeder block mid-write, and only then does
    output_queue.join_thread() in the worker wait on a reader that has gone away.
    """

    @classmethod
    def start(cls, **kwargs: Any) -> "BulkWorker":
        return cls()

    def process(self, items: Iterable[tuple[int, Any]]) -> Iterable[tuple[int, Any]]:
        for idx, _item in items:
            yield idx, b"x" * (1 << 18)  # 256 KiB, against a 64 KiB pipe buffer


# --- Tests ---


def test_worker_base_abstract():
    """Test that Worker base class methods raise NotImplementedError."""
    with pytest.raises(NotImplementedError):
        Worker.start()
    worker = Worker()
    with pytest.raises(NotImplementedError):
        list(worker.process([]))


def test_ordered_map_basic():
    """
    Test basic ordered map functionality with multiple workers.
    """
    pool = ParallelWorkerPool(worker=SquareWorker, config=PoolConfig(num_workers=2))
    input_data = list(range(10))
    # Expected output: squares of input
    expected = [x * x for x in input_data]

    results = list(pool.ordered_map(input_data))
    assert results == expected


def test_ordered_map_empty():
    """
    Test ordered map with empty input.
    """
    pool = ParallelWorkerPool(worker=SquareWorker, config=PoolConfig(num_workers=2))
    results = list(pool.ordered_map([]))
    assert results == []
    # Verify processes are joined and list is cleared
    assert len(pool.processes) == 0


def test_ordered_map_generator():
    """
    Test ordered map with a generator input.
    """
    pool = ParallelWorkerPool(worker=SquareWorker, config=PoolConfig(num_workers=2))
    input_gen = (x for x in range(10))
    expected = [x * x for x in range(10)]
    results = list(pool.ordered_map(input_gen))
    assert results == expected


def test_semi_ordered_map_basic():
    """
    Test basic semi-ordered map functionality.
    """
    pool = ParallelWorkerPool(worker=SquareWorker, config=PoolConfig(num_workers=2))
    input_data = list(range(10))
    # Expected: (idx, idx*idx)
    expected = {(idx, idx * idx) for idx in input_data}

    results = set(pool.semi_ordered_map(input_data))
    assert results == expected


def test_worker_failure_handling():
    """
    Test that the pool handles worker failures.
    """
    pool = ParallelWorkerPool(worker=FailingWorker, config=PoolConfig(num_workers=1))
    input_data = [1, 2, 3]

    with pytest.raises(RuntimeError, match="Thread unexpectedly terminated"):
        # failure_val=2 causes an exception in the worker
        list(pool.ordered_map(input_data, failure_val=2))


def test_pool_config_initialization():
    """
    Test PoolConfig default values.
    """
    config = PoolConfig(num_workers=4)
    assert config.num_workers == 4
    assert config.start_method is None
    assert config.device_ids is None


def test_pool_restart():
    """
    Test starting the pool multiple times (implicitly through multiple map calls).
    """
    pool = ParallelWorkerPool(worker=SquareWorker, config=PoolConfig(num_workers=1))
    input_data = [1, 2]

    assert list(pool.ordered_map(input_data)) == [1, 4]
    assert list(pool.ordered_map(input_data)) == [1, 4]


def test_process_stream_with_large_input():
    """
    Test processing more items than the queue size.
    """
    num_workers = 1
    # Queue size is num_workers * max_internal_batch_size = 200
    pool = ParallelWorkerPool(worker=SquareWorker, config=PoolConfig(num_workers=num_workers))
    input_data = list(range(300))
    expected = [x * x for x in input_data]

    results = list(pool.ordered_map(input_data))
    assert results == expected


def test_worker_cleanup_on_stop():
    """
    Verify workers stop when the stop signal is received.
    """
    # This is indirectly tested by the fact that pool.join() finishes.
    pool = ParallelWorkerPool(worker=SquareWorker, config=PoolConfig(num_workers=2))
    list(pool.ordered_map(range(5)))
    pool.join()
    for process in pool.processes:
        assert not process.is_alive()


# --- Tests for check_worker_health (lines 236-246) ---


def test_check_worker_health_healthy_processes():
    """Test check_worker_health does nothing when all processes are alive."""
    pool = ParallelWorkerPool(worker=SquareWorker, config=PoolConfig(num_workers=2))

    mock_process = MagicMock()
    mock_process.is_alive.return_value = True
    mock_process.exitcode = 0
    pool.processes = [mock_process, mock_process]

    # Should not raise
    pool.check_worker_health()
    assert not pool.emergency_shutdown


def test_check_worker_health_exited_zero():
    """Test check_worker_health ignores process that exited with code 0."""
    pool = ParallelWorkerPool(worker=SquareWorker, config=PoolConfig(num_workers=1))

    mock_process = MagicMock()
    mock_process.is_alive.return_value = False
    mock_process.exitcode = 0
    pool.processes = [mock_process]

    # Should not raise (exitcode=0 is normal)
    pool.check_worker_health()
    assert not pool.emergency_shutdown


def test_check_worker_health_dead_process_triggers_emergency():
    """Test check_worker_health raises RuntimeError when process exits with non-zero code."""
    pool = ParallelWorkerPool(worker=SquareWorker, config=PoolConfig(num_workers=1))

    mock_process = MagicMock()
    mock_process.is_alive.return_value = False
    mock_process.exitcode = 1
    mock_process.pid = 99999
    pool.processes = [mock_process]

    with pytest.raises(RuntimeError, match="terminated unexpectedly"):
        pool.check_worker_health()

    assert pool.emergency_shutdown


# --- Tests for join_or_terminate (line 248) ---


def test_join_or_terminate_alive_process_gets_terminated():
    """Test that join_or_terminate terminates a process that is still alive after join."""
    pool = ParallelWorkerPool(worker=SquareWorker, config=PoolConfig(num_workers=1))

    mock_process = MagicMock()
    mock_process.is_alive.return_value = True  # Still alive after join timeout
    pool.processes = [mock_process]

    pool.join_or_terminate(timeout=0)

    mock_process.join.assert_called_once_with(timeout=0)
    mock_process.terminate.assert_called_once()
    assert pool.processes == []


def test_join_or_terminate_dead_process_not_terminated():
    """Test that join_or_terminate does not terminate a process that finishes in time."""
    pool = ParallelWorkerPool(worker=SquareWorker, config=PoolConfig(num_workers=1))

    mock_process = MagicMock()
    mock_process.is_alive.return_value = False  # Finished during join
    pool.processes = [mock_process]

    pool.join_or_terminate(timeout=1)

    mock_process.join.assert_called_once_with(timeout=1)
    mock_process.terminate.assert_not_called()
    assert pool.processes == []


# --- Tests for __del__ (lines 265-277) ---


def test_del_terminates_alive_processes():
    """Test that __del__ terminates any still-alive processes."""
    pool = ParallelWorkerPool(worker=SquareWorker, config=PoolConfig(num_workers=1))

    alive_process = MagicMock()
    alive_process.is_alive.return_value = True

    dead_process = MagicMock()
    dead_process.is_alive.return_value = False

    pool.processes = [alive_process, dead_process]
    pool.__del__()

    alive_process.terminate.assert_called_once()
    dead_process.terminate.assert_not_called()


def test_del_no_processes():
    """Test that __del__ with no processes does not raise."""
    pool = ParallelWorkerPool(worker=SquareWorker, config=PoolConfig(num_workers=1))
    pool.processes = []
    pool.__del__()  # Should not raise


def test_process_stream_timeout_raises_empty():
    """Test that Empty exception from output_queue.get(timeout) is re-raised and join_or_terminate is called."""
    with patch.object(pp_module, "max_internal_batch_size", 1):
        pool = ParallelWorkerPool(worker=SquareWorker, config=PoolConfig(num_workers=1))
        pool.input_queue = MagicMock()
        pool.output_queue = MagicMock()

        # We need pushed - read >= queue_size. With queue_size=1, on the second iteration (idx=1),
        # pushed=1, read=0 -> pushed-read=1 >= 1.
        # So we mock output_queue.get to raise Empty.
        pool.output_queue.get.side_effect = Empty()

        pool.join_or_terminate = MagicMock()  # type: ignore
        pool.check_worker_health = MagicMock()  # type: ignore

        # Processing [1, 2] will cause 2 iterations. First iteration will pass if get_nowait raises Empty
        # (handled gracefully). Second iteration will trigger the else branch and raise the mock Empty.
        pool.output_queue.get_nowait.side_effect = Empty()

        with pytest.raises(Empty):
            list(pool._process_stream([1, 2]))

        pool.join_or_terminate.assert_called_once()  # type: ignore


def test_semi_ordered_map_emergency_shutdown_cancels_join_thread():
    """Test that cancel_join_thread is called in semi_ordered_map finally block if emergency_shutdown is True."""
    pool = ParallelWorkerPool(worker=SquareWorker, config=PoolConfig(num_workers=1))

    # Mock necessary methods to avoid actual processing
    pool.start = MagicMock()  # type: ignore

    # We want emergency_shutdown to be True when we hit the finally block
    def mock_process_stream(*args: Any, **kwargs: Any) -> Any:
        pool.emergency_shutdown = True
        yield from []

    pool._process_stream = mock_process_stream  # type: ignore
    pool.join = MagicMock()  # type: ignore

    # Setup mock queues before calling
    mock_input_queue = MagicMock()
    mock_output_queue = MagicMock()
    pool.input_queue = mock_input_queue
    pool.output_queue = mock_output_queue

    # Need to keep track of the original queues so they aren't overwritten if start() is called,
    # but since we mocked start, they remain intact.

    list(pool.semi_ordered_map([]))

    # Validate cancel_join_thread was called
    mock_input_queue.cancel_join_thread.assert_called_once()
    mock_output_queue.cancel_join_thread.assert_called_once()

    # Ensure standard join_thread was not called
    mock_input_queue.join_thread.assert_not_called()
    mock_output_queue.join_thread.assert_not_called()


def test_worker_multiprocessing_exception_handling():
    """
    Test multiprocessing exception handling where the child worker process
    raises an exception. Verifies that the worker puts QueueSignals.error on the queue
    and the main process subsequently raises a RuntimeError.
    """
    # Using FailingWorker with failure_val=2.
    # When item '2' is processed by the worker subprocess, it raises an exception.
    pool = ParallelWorkerPool(worker=FailingWorker, config=PoolConfig(num_workers=2))

    with pytest.raises(RuntimeError, match="Thread unexpectedly terminated"):
        # The list consumption forces the stream through the pool
        list(pool.ordered_map([1, 2, 3], failure_val=2))


class StartFailingWorker(Worker):
    @classmethod
    def start(cls, **kwargs: Any) -> "StartFailingWorker":
        raise RuntimeError("Initialization failed")

    def process(self, items: Iterable[tuple[int, Any]]) -> Iterable[tuple[int, Any]]:
        yield from items


def test_worker_start_exception_handling():
    """Test that _worker handles exceptions during Worker.start() gracefully."""
    input_queue = MagicMock()
    output_queue = MagicMock()
    num_active_workers = MagicMock()
    num_active_workers.get_lock.return_value.__enter__ = MagicMock()
    num_active_workers.get_lock.return_value.__exit__ = MagicMock()
    num_active_workers.value = 1
    worker_id = 0

    with pytest.raises(RuntimeError):
        _worker(
            worker_class=StartFailingWorker,
            input_queue=input_queue,
            output_queue=output_queue,
            num_active_workers=num_active_workers,
            worker_id=worker_id,
            kwargs={},
        )

    # Verify that QueueSignals.error was put in the output queue
    output_queue.put.assert_called_with(QueueSignals.error)
    # Verify that the worker was cleaned up (num_active_workers decremented)
    assert num_active_workers.value == 0


class ProcessFailingWorker(Worker):
    @classmethod
    def start(cls, **kwargs: Any) -> "ProcessFailingWorker":
        return cls()

    def process(self, items: Iterable[tuple[int, Any]]) -> Iterable[tuple[int, Any]]:
        raise RuntimeError("Processing failed")


def test_worker_processing_exception_handling():
    """Test that _worker handles exceptions during worker.process() gracefully."""
    input_queue = MagicMock()
    output_queue = MagicMock()
    num_active_workers = MagicMock()
    num_active_workers.get_lock.return_value.__enter__ = MagicMock()
    num_active_workers.get_lock.return_value.__exit__ = MagicMock()
    num_active_workers.value = 1
    worker_id = 0

    # Mock _get_items_from_queue to return a single item then stop
    # This triggers worker.process() and then the exception.
    with (
        patch("qwen3_embed.parallel_processor._get_items_from_queue", return_value=[(0, "item")]),
        pytest.raises(RuntimeError),
    ):
        _worker(
            worker_class=ProcessFailingWorker,
            input_queue=input_queue,
            output_queue=output_queue,
            num_active_workers=num_active_workers,
            worker_id=worker_id,
            kwargs={},
        )

    # Verify that QueueSignals.error was put in the output queue
    output_queue.put.assert_called_with(QueueSignals.error)
    # Verify that the worker was cleaned up (num_active_workers decremented)
    assert num_active_workers.value == 0


# --- Coverage Enhancement Tests ---


def test_worker_success_path():
    """Test that _worker successfully processes items in a normal flow."""
    input_queue = MagicMock()
    output_queue = MagicMock()
    num_active_workers = MagicMock()
    num_active_workers.get_lock.return_value.__enter__ = MagicMock()
    num_active_workers.get_lock.return_value.__exit__ = MagicMock()
    num_active_workers.value = 1
    worker_id = 0

    # Mock _get_items_from_queue to return items then stop
    # We yield (0, 10) then stop.
    with patch("qwen3_embed.parallel_processor._get_items_from_queue", return_value=[(0, 10)]):
        _worker(
            worker_class=SquareWorker,
            input_queue=input_queue,
            output_queue=output_queue,
            num_active_workers=num_active_workers,
            worker_id=worker_id,
            kwargs={},
        )

    # SquareWorker should have yielded (0, 100)
    output_queue.put.assert_called_with((0, 100))
    # Verify cleanup happened
    assert num_active_workers.value == 0


def test_worker_none_kwargs():
    """Test _worker handles kwargs=None (line 82)."""
    input_queue = MagicMock()
    output_queue = MagicMock()
    num_active_workers = MagicMock()
    num_active_workers.get_lock.return_value.__enter__ = MagicMock()
    num_active_workers.get_lock.return_value.__exit__ = MagicMock()
    num_active_workers.value = 1
    worker_id = 0

    with patch("qwen3_embed.parallel_processor._get_items_from_queue", return_value=[]):
        _worker(
            worker_class=SquareWorker,
            input_queue=input_queue,
            output_queue=output_queue,
            num_active_workers=num_active_workers,
            worker_id=worker_id,
            kwargs=None,
        )
    assert num_active_workers.value == 0


def test_get_items_from_queue_logic():
    """Test _get_items_from_queue generator and stop signal."""
    mock_queue = MagicMock()
    mock_queue.get.side_effect = [1, 2, QueueSignals.stop]

    items = list(_get_items_from_queue(mock_queue))
    assert items == [1, 2]
    assert mock_queue.get.call_count == 3


def test_cleanup_worker_logic():
    """Test _cleanup_worker functionality."""
    input_queue = MagicMock()
    output_queue = MagicMock()
    num_active_workers = MagicMock()
    num_active_workers.get_lock.return_value.__enter__ = MagicMock()
    num_active_workers.get_lock.return_value.__exit__ = MagicMock()
    num_active_workers.value = 5
    worker_id = 1

    _cleanup_worker(input_queue, output_queue, num_active_workers, worker_id)

    input_queue.close.assert_called_once()
    output_queue.close.assert_called_once()
    input_queue.join_thread.assert_called_once()
    output_queue.join_thread.assert_called_once()
    assert num_active_workers.value == 4


def test_parallel_worker_pool_default_config():
    """Test ParallelWorkerPool handles None config by creating a default one."""
    pool = ParallelWorkerPool(worker=SquareWorker, config=None)
    assert pool.num_workers == 1
    assert pool.worker_class == SquareWorker


def test_pool_with_device_ids():
    """Test ParallelWorkerPool.start with device_ids (lines 146-148)."""
    config = PoolConfig(num_workers=2, device_ids=[0, 1])
    pool = ParallelWorkerPool(worker=SquareWorker, config=config)
    pool.ctx = MagicMock()
    pool.ctx.Value.return_value = MagicMock(spec=BaseValue)
    pool.ctx.Process.return_value = MagicMock()

    pool.start(extra_param="value")

    assert pool.ctx.Process.call_count == 2
    # Check device_id propagation
    args1 = pool.ctx.Process.call_args_list[0][1]["args"]
    args2 = pool.ctx.Process.call_args_list[1][1]["args"]
    assert args1[5]["device_id"] == 0
    assert args2[5]["device_id"] == 1


def test_process_stream_error_signal():
    """Test that QueueSignals.error in _process_stream raises RuntimeError."""
    pool = ParallelWorkerPool(worker=SquareWorker, config=PoolConfig(num_workers=1))
    pool.input_queue = MagicMock()
    pool.output_queue = MagicMock()
    pool.output_queue.get_nowait.return_value = QueueSignals.error
    pool.join_or_terminate = MagicMock()  # type: ignore[assignment]
    pool.check_worker_health = MagicMock()  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="Thread unexpectedly terminated"):
        list(pool._process_stream([1]))

    pool.join_or_terminate.assert_called_once()  # type: ignore[attr-defined]


def test_join_or_terminate_mixed_states():
    """Test join_or_terminate with multiple processes in different states."""
    pool = ParallelWorkerPool(worker=SquareWorker, config=PoolConfig(num_workers=3))

    # P1: Finishes immediately
    p1 = MagicMock()
    p1.is_alive.return_value = False

    # P2: Hangs and needs termination
    p2 = MagicMock()
    p2.is_alive.return_value = True

    # P3: Also hangs
    p3 = MagicMock()
    p3.is_alive.return_value = True

    pool.processes = [p1, p2, p3]

    pool.join_or_terminate(timeout=1)

    # P1 should have been joined but not terminated
    p1.join.assert_called_once_with(timeout=1)
    p1.terminate.assert_not_called()

    # P2 and P3 should have been joined AND terminated
    p2.join.assert_called_once_with(timeout=1)
    p2.terminate.assert_called_once()
    p3.join.assert_called_once_with(timeout=1)
    p3.terminate.assert_called_once()

    # Processes list should be cleared
    assert pool.processes == []


def test_join_or_terminate_empty():
    """Test join_or_terminate with no processes."""
    pool = ParallelWorkerPool(worker=SquareWorker, config=PoolConfig(num_workers=0))
    pool.processes = []
    pool.join_or_terminate()
    assert pool.processes == []


def test_cleanup_worker_error_handling():
    """Test that _cleanup_worker handles exceptions during queue cleanup gracefully."""
    input_queue = MagicMock()
    output_queue = MagicMock()
    num_active_workers = MagicMock()
    num_active_workers.get_lock.return_value.__enter__ = MagicMock()
    num_active_workers.get_lock.return_value.__exit__ = MagicMock()
    num_active_workers.value = 5
    worker_id = 1

    # Mock all cleanup methods to raise exceptions
    input_queue.close.side_effect = Exception("input_queue.close failed")
    output_queue.close.side_effect = Exception("output_queue.close failed")
    input_queue.join_thread.side_effect = Exception("input_queue.join_thread failed")
    output_queue.join_thread.side_effect = Exception("output_queue.join_thread failed")

    with patch("logging.exception") as mock_log_exception:
        _cleanup_worker(input_queue, output_queue, num_active_workers, worker_id)

    # Verify that all exceptions were logged (one for each failed cleanup step)
    assert mock_log_exception.call_count == 4
    log_messages = [call.args[0] for call in mock_log_exception.call_args_list]
    assert any("failed to close input_queue" in msg for msg in log_messages)
    assert any("failed to close output_queue" in msg for msg in log_messages)
    assert any("failed to join_thread input_queue" in msg for msg in log_messages)
    assert any("failed to join_thread output_queue" in msg for msg in log_messages)

    # Verify that all cleanup methods were still called despite previous failures
    input_queue.close.assert_called_once()
    output_queue.close.assert_called_once()
    input_queue.join_thread.assert_called_once()
    output_queue.join_thread.assert_called_once()

    # Verify that num_active_workers was still decremented despite all errors
    assert num_active_workers.value == 4


def test_get_items_from_queue_error_handling():
    """Test that _get_items_from_queue propagates exceptions from queue.get()."""
    mock_queue = MagicMock()
    mock_queue.get.side_effect = Exception("Queue error")

    with pytest.raises(Exception, match="Queue error"):
        list(_get_items_from_queue(mock_queue))


def test_semi_ordered_map_empty():
    """Test semi_ordered_map with an empty stream."""
    pool = ParallelWorkerPool(worker=SquareWorker, config=PoolConfig(num_workers=2))
    stream = []
    result = list(pool.semi_ordered_map(stream))
    assert result == []
    # Verify processes are joined and list is cleared
    assert len(pool.processes) == 0


def test_semi_ordered_map_interrupted_no_hang():
    """Test that semi_ordered_map does not hang when consumption is interrupted early."""
    pool = ParallelWorkerPool(worker=SlowWorker, config=PoolConfig(num_workers=1))

    # We consume only the first item and then break.
    # This should trigger the finally block and cleanup processes without hanging.
    try:
        for _idx, val in pool.semi_ordered_map(range(10)):
            assert val == 0
            break
    except Exception:
        pytest.fail("semi_ordered_map raised an unexpected exception")


@pytest.mark.timeout(90)
def test_semi_ordered_map_abandoned_with_full_pipe_terminates_workers():
    """An abandoned consumer must not deadlock while workers hold unread output.

    The test above looks like it covers this and does not: its results are small
    ints, which fit in the pipe buffer, so the feeder thread drains and every
    worker exits no matter what the pool does. This one uses payloads larger than
    the buffer, which is the condition under which the workers block in
    output_queue.join_thread() -- and then the pool must terminate them rather
    than join() them forever.

    Observed as a real outage before the fix: a TypeError raised by the consumer
    of rerank_pairs(..., parallel=2) closed this generator with GeneratorExit,
    which is not an Exception, so the pool took the non-emergency path and the
    process never exited. CI showed a test suite that produced no result at all
    for 59 minutes.
    """
    pool = ParallelWorkerPool(worker=BulkWorker, config=PoolConfig(num_workers=2))

    stream = pool.semi_ordered_map(range(64))
    for _idx, _val in stream:
        break
    # Close explicitly rather than leaving it to refcounting: this call is the
    # exact point that used to block forever.
    stream.close()

    assert pool.processes == []
    assert pool.emergency_shutdown is True


def test_empty_stream_worker_start_failure():
    """Test that worker start failure is detected even for an empty input stream."""
    pool = ParallelWorkerPool(worker=StartFailingWorker, config=PoolConfig(num_workers=1))

    with pytest.raises(RuntimeError, match="terminated unexpectedly"):
        list(pool.ordered_map([]))


def test_semi_ordered_map_start_failure():
    """Test semi_ordered_map when start() fails."""
    pool = ParallelWorkerPool(worker=SquareWorker, config=PoolConfig(num_workers=1))

    with (
        patch.object(ParallelWorkerPool, "start", side_effect=RuntimeError("Start failed")),
        pytest.raises(RuntimeError, match="Start failed"),
    ):
        list(pool.semi_ordered_map([]))

    # queues should be None or handled safely
    assert pool.input_queue is None
    assert pool.output_queue is None


def test_check_worker_health_healthy():
    """Test check_worker_health does not raise when all processes are alive or exited cleanly."""
    pool = ParallelWorkerPool(worker=SquareWorker, config=PoolConfig(num_workers=2))

    # Process 1: Alive
    p1 = MagicMock()
    p1.is_alive.return_value = True
    p1.exitcode = None

    # Process 2: Exited cleanly
    p2 = MagicMock()
    p2.is_alive.return_value = False
    p2.exitcode = 0

    pool.processes = [p1, p2]

    # Should not raise
    pool.check_worker_health()
    assert pool.emergency_shutdown is False


def test_check_worker_health_unhealthy():
    """Test check_worker_health raises RuntimeError when a process terminates unexpectedly."""
    pool = ParallelWorkerPool(worker=SquareWorker, config=PoolConfig(num_workers=1))

    p1 = MagicMock()
    p1.is_alive.return_value = False
    p1.exitcode = 1
    p1.pid = 1234

    pool.processes = [p1]
    pool.join_or_terminate = MagicMock()  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="Worker PID: 1234 terminated unexpectedly with code 1"):
        pool.check_worker_health()

    assert pool.emergency_shutdown is True
    pool.join_or_terminate.assert_called_once()  # type: ignore[attr-defined]


def test_check_worker_health_called_during_processing():
    """Test that check_worker_health is called during the processing loop."""
    pool = ParallelWorkerPool(worker=SquareWorker, config=PoolConfig(num_workers=1))

    # Mock necessary parts to avoid real multiprocessing
    pool.start = MagicMock()  # type: ignore[assignment]
    pool.join = MagicMock()  # type: ignore[assignment]
    pool.input_queue = MagicMock()  # type: ignore[assignment]
    pool.output_queue = MagicMock()  # type: ignore[assignment]

    # Simulate one item being processed
    # First call to get_nowait returns None, second call returns item
    pool.output_queue.get_nowait.side_effect = [None, (0, 100)]  # type: ignore[attr-defined]
    # Final get call for the remaining items
    pool.output_queue.get.return_value = (0, 100)  # type: ignore[attr-defined]

    with patch.object(ParallelWorkerPool, "check_worker_health") as mock_check_health:
        # We need to exhaust the generator
        list(pool.semi_ordered_map([10]))

        # It should be called at least once
        assert mock_check_health.called
