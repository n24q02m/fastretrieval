import numpy as np
import time

def bench(batch_size, vocab_size, sparsity):
    np.random.seed(42)
    # create dummy output
    model_output = np.zeros((batch_size, 1, vocab_size), dtype=np.float32)

    # fill with some random values based on sparsity
    for i in range(batch_size):
        num_nonzeros = int(vocab_size * sparsity)
        indices = np.random.choice(vocab_size, num_nonzeros, replace=False)
        model_output[i, 0, indices] = np.random.uniform(0.1, 5.0, num_nonzeros)

    # old way
    start = time.perf_counter()
    for _ in range(100):
        mo = model_output.copy()
        np.log1p(mo[:, 0, :], out=mo[:, 0, :])
        scores_matrix = mo[:, 0, :]
        res = []
        for row_scores in scores_matrix:
            indices = row_scores.nonzero()[0]
            scores = row_scores[indices]
            res.append((scores, indices))
    old_time = time.perf_counter() - start

    # new way
    start = time.perf_counter()
    for _ in range(100):
        mo = model_output.copy()
        scores_matrix = mo[:, 0, :]
        res = []
        for row_scores in scores_matrix:
            indices = row_scores.nonzero()[0]
            scores = row_scores[indices]
            np.log1p(scores, out=scores)
            res.append((scores, indices))
    new_time = time.perf_counter() - start

    print(f"Batch {batch_size}, Sparsity {sparsity:.4f}: Old: {old_time:.4f}s, New: {new_time:.4f}s, Speedup: {old_time/new_time:.2f}x")

bench(64, 30522, 0.01)
bench(256, 30522, 0.01)
