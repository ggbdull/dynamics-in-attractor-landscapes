import numpy as np

from src.config import *


def phi(x):
    return np.tanh(x)


def generate_ground_truth():
    return np.random.choice([-1, 1], size=(n_o, n_s)).astype(float)


def build_W_from_S(S):
    return (strength / n_s) * (S.T @ S)


def infer(W, start_s):

    s = start_s.copy()

    for _ in range(max_steps):

        s_next = phi(W @ s)

        if np.linalg.norm(s_next - s) < 1e-5:
            break

        s = s_next

    return s


def evaluate_W(W, true_s, n_trials=50, noise_level=0.3):

    success, wrong, fail = 0, 0, 0

    for _ in range(n_trials):

        home_idx = np.random.randint(n_o)

        target = true_s[home_idx]

        start = target + np.random.randn(n_s) * noise_level

        s_final = infer(W, start)

        norm_final = np.linalg.norm(s_final)

        if norm_final < 0.1:
            fail += 1
            continue

        sims = [
            np.dot(s_final, t) /
            (norm_final * np.linalg.norm(t))
            for t in true_s
        ]

        best_idx = np.argmax(sims)

        if sims[best_idx] > 0.9:

            if best_idx == home_idx:
                success += 1
            else:
                wrong += 1

        else:
            fail += 1

    return (
        success / n_trials
        - 0.5 * (wrong / n_trials)
        - 0.5 * (fail / n_trials)
    )


class Region:

    def __init__(self, low, high):

        self.low = low
        self.high = high
        self.score = None


dim = n_o * n_s

S_low = -1.5 * np.ones(dim)

S_high = 1.5 * np.ones(dim)


def sample_S(region):

    return np.random.uniform(
        region.low,
        region.high
    ).reshape(n_o, n_s)


def split_region(region):

    mid = (region.low + region.high) / 2

    regions = []

    for _ in range(100):

        low = region.low.copy()

        high = region.high.copy()

        mask = np.random.randint(0, 2, len(low))

        low[mask == 1] = mid[mask == 1]

        high[mask == 0] = mid[mask == 0]

        regions.append(Region(low, high))

    return regions


def run_search(budget, true_s, top_k):

    regions = [Region(S_low, S_high)]

    best_score = -np.inf

    history = []

    used = 0

    while used < budget:

        for r in regions:

            if r.score is None:

                scores = []

                for _ in range(samples_per_region):

                    if used >= budget:
                        break

                    score = evaluate_W(
                        build_W_from_S(sample_S(r)),
                        true_s
                    )

                    scores.append(score)

                    if score > best_score:
                        best_score = score

                    history.append(best_score)

                    used += 1

                if scores:
                    r.score = np.mean(scores)

        regions.sort(
            key=lambda x: x.score if x.score is not None else -np.inf,
            reverse=True
        )

        regions = regions[:top_k]

        new_regions = []

        for r in regions:

            new_regions.extend(split_region(r))

        regions = new_regions

        if not regions:
            break

    return best_score, history