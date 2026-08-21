## 2023-10-27 - [Batching Multiprocessing Process Health Checks]
**Learning:** Checking `process.is_alive()` or `process.exitcode` via multiprocessing on *every single iteration* of a tight loop can dominate execution time due to Python/OS-level system call overhead (like waitpid).
**Action:** When streaming data to/from process queues, batch worker health checks (e.g. `if idx % 100 == 0:`) or check periodically based on time (`time.monotonic()`) instead of checking synchronously per item.
