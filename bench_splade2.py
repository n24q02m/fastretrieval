import numpy as np
import time

def bench(batch_size, seq_len, vocab_size, sparsity):
    np.random.seed(42)
    # create dummy output
    model_output = np.zeros((batch_size, seq_len, vocab_size), dtype=np.float32)

    # fill with some random values based on sparsity
    for i in range(batch_size):
        for j in range(seq_len):
            if np.random.rand() > 0.5: # simulate attention mask
                num_nonzeros = int(vocab_size * sparsity)
                indices = np.random.choice(vocab_size, num_nonzeros, replace=False)
                model_output[i, j, indices] = np.random.uniform(-1.0, 5.0, num_nonzeros)

    # old way (current code)
    start = time.perf_counter()
    for _ in range(100):
        mo = model_output.copy()

        # simulate mask
        # mo[attention_mask == 0] = 0 (ignored for simplicity)

        np.max(mo, axis=1, out=mo[:, 0, :])
        np.maximum(mo[:, 0, :], 0, out=mo[:, 0, :])
        np.log1p(mo[:, 0, :], out=mo[:, 0, :])
        scores_matrix = mo[:, 0, :]
        res = []
        for row_scores in scores_matrix:
            indices = row_scores.nonzero()[0]
            scores = row_scores[indices]
            res.append((scores, indices))
    old_time = time.perf_counter() - start

    # new way (optimized)
    start = time.perf_counter()
    for _ in range(100):
        mo = model_output.copy()

        # simulate mask

        # max pool over sequence length
        scores_matrix = np.max(mo, axis=1) # shape: (batch_size, vocab_size)

        # maximum(0) is relu
        np.maximum(scores_matrix, 0, out=scores_matrix)

        res = []
        for row_scores in scores_matrix:
            indices = row_scores.nonzero()[0]
            scores = row_scores[indices]
            # apply log1p only on non-zero elements
            np.log1p(scores, out=scores)
            res.append((scores, indices))
    new_time = time.perf_counter() - start

    print(f"Batch {batch_size}, Seq {seq_len}, Sparsity {sparsity:.4f}: Old: {old_time:.4f}s, New: {new_time:.4f}s, Speedup: {old_time/new_time:.2f}x")

bench(64, 32, 30522, 0.01)
bench(128, 32, 30522, 0.01)
