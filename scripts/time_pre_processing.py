import itertools
import time
import numpy as np

from scorer import VisualAdditionScorer

times = []
for i in range(5):
    all_label_pairs = list(itertools.product(list(range(0, 10)), repeat=2))

    # Start the timer
    start_time = time.time()

    # Augment the label_pairs with the symbolic uncertainty scores
    scorer = VisualAdditionScorer()
    s1_dom = {s1 for s1 in range(10)}
    s2_dom = {s2 for s2 in range(10)}
    y_dom = {y for y in range(19)}
    scores = scorer.score(s1_dom, s2_dom, y_dom)
    train_label_triples = [(a, b, scores[a + b]) for (a, b) in all_label_pairs]

    # Stop the timer
    end_time = time.time()

    # Calculate the execution time
    execution_time = end_time - start_time
    times.append(execution_time)
    print(f"Execution time: {execution_time:.6f} seconds")

print(f"Mean Execution time: {np.mean(times)} +- {np.std(times)}")