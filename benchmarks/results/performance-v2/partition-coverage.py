"""Estimate signal exposure for size-matched proposal policies without fitting models."""

from itertools import combinations
from math import comb
from random import Random


def coverage(width, seed, partitioned):
    rng = Random(seed)
    seen = set()
    singles = set()
    pairs = set()
    while len(seen) < 24:
        columns = rng.sample(range(width), width)
        stops = width - 3 if partitioned else 1
        for start in range(0, stops, 4):
            mask = tuple(sorted(columns[start : start + 4]))
            if mask in seen:
                continue
            seen.add(mask)
            singles.update(mask)
            pairs.update(combinations(mask, 2))
            if len(seen) in (6, 24):
                yield len(singles) / width, len(pairs) / comb(width, 2)
            if len(seen) == 24:
                break


if __name__ == "__main__":
    seeds = range(2048)
    print("columns policy budget single_pct pair_pct")
    for width in (12, 16, 32, 64):
        for partitioned in (False, True):
            runs = [list(coverage(width, seed, partitioned)) for seed in seeds]
            policy = "partition" if partitioned else "random"
            for budget, batch in zip((6, 24), zip(*runs)):
                single, pair = (100 * sum(values) / len(seeds) for values in zip(*batch))
                print(width, policy, budget, f"{single:.2f}", f"{pair:.2f}")
