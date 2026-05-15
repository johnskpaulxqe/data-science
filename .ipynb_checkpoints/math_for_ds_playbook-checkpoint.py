
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Math for Data Science — Hands-on Python Playbook
================================================

This single file is a *runnable, heavily commented* reference that demonstrates
the essential math concepts for data science. Each section has:
- Minimal theory recap (in comments)
- Clean, small functions
- Tiny runnable demos you can tweak

USAGE:
------
$ python math_for_ds_playbook.py

Or open in an editor and run section-by-section.
No external data or non-standard packages required (NumPy and matplotlib optional).
If NumPy is unavailable, many sections fall back to pure-Python implementations.

CONTENTS:
---------
0) Utilities
1) Foundations (Sets, Combinatorics)
2) Probability Basics (Bayes, simulations, distributions)
3) Descriptive Statistics & Inference (mean/var, CLT demo, simple z-test)
4) Linear Algebra (norms, dot, matmul, eig, PCA via SVD)
5) Calculus & Optimization (numeric derivative, gradient descent)
6) Machine Learning Math (linear/logistic regression, regularization, k-means)
7) Markov Chains (simulation, stationary distribution via power method)
8) Information Theory (entropy, KL divergence)
"""

# -----------------------------
# 0) Utilities
# -----------------------------
from typing import List, Tuple, Dict, Callable, Iterable, Optional
import math
import random

try:
    import numpy as np
except Exception:
    np = None  # Script still runs for many parts without NumPy

# Reproducibility for random demos
random.seed(42)
if np is not None:
    np.random.seed(42)


def section(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# -----------------------------
# 1) FOUNDATIONS: Sets & Combinatorics
# -----------------------------
def factorial(n: int) -> int:
    """Compute n! (factorial) iteratively to avoid recursion limits."""
    if n < 0:
        raise ValueError("n must be >= 0")
    result = 1
    for k in range(2, n + 1):
        result *= k
    return result


def permutations(n: int, r: int) -> int:
    """P(n, r) = n! / (n-r)!  (order matters)."""
    if r > n or n < 0 or r < 0:
        return 0
    return factorial(n) // factorial(n - r)


def combinations(n: int, r: int) -> int:
    """C(n, r) = n! / (r! (n-r)!)  (order doesn't matter)."""
    if r > n or n < 0 or r < 0:
        return 0
    return factorial(n) // (factorial(r) * factorial(n - r))


def set_ops_demo():
    """Basic set operations used in probability (events)."""
    A = {1, 2, 3, 4}
    B = {3, 4, 5}
    union = A | B           # A ∪ B
    intersection = A & B    # A ∩ B
    complement_in_universe = set(range(1, 7)) - A  # U \ A (assume U = {1..6})
    return union, intersection, complement_in_universe


# -----------------------------
# 2) PROBABILITY BASICS
# -----------------------------
def conditional_probability(p_a_and_b: float, p_b: float) -> float:
    """P(A|B) = P(A∩B) / P(B)."""
    if p_b == 0:
        raise ZeroDivisionError("P(B) must be > 0")
    return p_a_and_b / p_b


def bayes(p_a: float, p_b_given_a: float, p_b_given_not_a: float) -> float:
    """Bayes' theorem: P(A|B) = P(B|A)P(A) / P(B), with P(B) expanded by total probability."""
    p_not_a = 1 - p_a
    p_b = p_b_given_a * p_a + p_b_given_not_a * p_not_a
    if p_b == 0:
        raise ZeroDivisionError("P(B) must be > 0")
    return (p_b_given_a * p_a) / p_b


def simulate_binomial(n: int, p: float, trials: int = 10_000) -> float:
    """
    Monte Carlo estimate of P(X = k) when X ~ Binomial(n, p).
    Here we just return sample mean as a sanity check (≈ n*p).
    """
    successes = []
    for _ in range(trials):
        x = sum(1 for _ in range(n) if random.random() < p)
        successes.append(x)
    return sum(successes) / len(successes)


def poisson_pmf(lmbda: float, k: int) -> float:
    """Poisson PMF: P(X=k) = e^{-λ} λ^k / k!"""
    return math.exp(-lmbda) * (lmbda ** k) / factorial(k)


def normal_pdf(mu: float, sigma: float, x: float) -> float:
    """Univariate normal PDF."""
    coef = 1.0 / (sigma * math.sqrt(2 * math.pi))
    expo = math.exp(-0.5 * ((x - mu) / sigma) ** 2)
    return coef * expo


# -----------------------------
# 3) DESCRIPTIVE STATS & INFERENCE
# -----------------------------
def mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return sum(xs) / len(xs)


def variance(xs: Iterable[float], ddof: int = 1) -> float:
    """Sample variance by default (ddof=1)."""
    xs = list(xs)
    mu = mean(xs)
    return sum((x - mu) ** 2 for x in xs) / (len(xs) - ddof)


def covariance(xs: Iterable[float], ys: Iterable[float]) -> float:
    xs, ys = list(xs), list(ys)
    if len(xs) != len(ys):
        raise ValueError("xs and ys must be same length")
    mx, my = mean(xs), mean(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (len(xs) - 1)


def correlation(xs: Iterable[float], ys: Iterable[float]) -> float:
    """Pearson correlation."""
    xs, ys = list(xs), list(ys)
    return covariance(xs, ys) / (math.sqrt(variance(xs)) * math.sqrt(variance(ys)))


def clt_demo(sample_size: int = 30, reps: int = 5000) -> Tuple[float, float]:
    """
    Central Limit Theorem demo:
    Draw 'sample_size' values from a skewed distribution (exponential),
    repeat 'reps' times, and return mean and variance of the sample means.
    The distribution of sample means approaches Normal as reps grows.
    """
    means = []
    for _ in range(reps):
        # Exponential via inverse transform: X = -ln(U), U~Uniform(0,1)
        sample = [-math.log(1 - random.random()) for _ in range(sample_size)]
        means.append(mean(sample))
    return mean(means), variance(means, ddof=1)


def z_test_two_means(x1: List[float], x2: List[float], sigma1: float, sigma2: float) -> float:
    """
    Simple two-sample z-test (known population std devs). Returns z statistic.
    For real work, a t-test is better (unknown sigmas), usually via SciPy.
    """
    m1, m2 = mean(x1), mean(x2)
    n1, n2 = len(x1), len(x2)
    se = math.sqrt((sigma1 ** 2) / n1 + (sigma2 ** 2) / n2)
    return (m1 - m2) / se


# -----------------------------
# 4) LINEAR ALGEBRA
# -----------------------------
def dot(x: List[float], y: List[float]) -> float:
    return sum(a * b for a, b in zip(x, y))


def norm2(x: List[float]) -> float:
    return math.sqrt(dot(x, x))


def matmul(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Pure-Python matrix multiplication for clarity."""
    n, k, m = len(A), len(A[0]), len(B[0])
    assert k == len(B), "Inner dims must match"
    C = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            C[i][j] = sum(A[i][t] * B[t][j] for t in range(k))
    return C


def eigen_and_pca_demo(X: List[List[float]]) -> Dict[str, object]:
    """
    PCA via SVD using NumPy if available; otherwise, a tiny covariance + eig demo.
    Returns principal components and explained variance.
    """
    if np is None:
        # Fallback: compute covariance and a crude power iteration for top eigenvector
        # Center the data
        mu = [mean(col) for col in zip(*X)]
        Xc = [[xj - muj for xj, muj in zip(xi, mu)] for xi in X]
        # Covariance matrix (d x d)
        d = len(Xc[0])
        Cov = [[0.0]*d for _ in range(d)]
        n = len(Xc)
        for i in range(d):
            for j in range(d):
                Cov[i][j] = sum(Xc[t][i] * Xc[t][j] for t in range(n)) / (n - 1)
        # Power iteration to find top eigenvector
        v = [1.0] * d
        for _ in range(100):
            v_new = [sum(Cov[i][j]*v[j] for j in range(d)) for i in range(d)]
            norm = math.sqrt(sum(vi*vi for vi in v_new))
            v = [vi / (norm + 1e-12) for vi in v_new]
        # Rayleigh quotient for eigenvalue
        Av = [sum(Cov[i][j]*v[j] for j in range(d)) for i in range(d)]
        eigval = sum(vi*ai for vi, ai in zip(v, Av))
        return {"top_eigenvalue": eigval, "top_eigenvector": v, "note": "NumPy not available; returned only top component."}
    else:
        X = np.asarray(X, dtype=float)
        Xc = X - X.mean(axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        pcs = Vt  # principal axes (rows)
        explained_var = (S**2) / (len(X) - 1)
        explained_ratio = explained_var / explained_var.sum()
        return {"pcs": pcs, "explained_variance": explained_var, "explained_ratio": explained_ratio}


# -----------------------------
# 5) CALCULUS & OPTIMIZATION
# -----------------------------
def numeric_derivative(f: Callable[[float], float], x: float, h: float = 1e-5) -> float:
    """Symmetric difference quotient (good default for smooth functions)."""
    return (f(x + h) - f(x - h)) / (2 * h)


def gradient_descent_1d(f: Callable[[float], float],
                        df: Optional[Callable[[float], float]] = None,
                        x0: float = 0.0,
                        lr: float = 0.1,
                        steps: int = 100) -> Tuple[float, List[float]]:
    """
    Minimize f(x) in 1D using gradient descent.
    If df is None, uses numeric derivative.
    Returns final x and the trajectory.
    """
    x = x0
    traj = [x]
    for _ in range(steps):
        g = df(x) if df is not None else numeric_derivative(f, x)
        x = x - lr * g
        traj.append(x)
    return x, traj


# -----------------------------
# 6) MACHINE LEARNING MATH
# -----------------------------
def linear_regression_closed_form(X: List[List[float]], y: List[float]) -> List[float]:
    """
    Closed-form OLS: beta = (X^T X)^(-1) X^T y.
    Uses NumPy if available; otherwise a naive normal-equation solve using Gaussian elimination.
    """
    if np is not None:
        Xnp = np.asarray(X, dtype=float)
        ynp = np.asarray(y, dtype=float)
        beta = np.linalg.pinv(Xnp.T @ Xnp) @ Xnp.T @ ynp  # pinv for numerical stability
        return beta.tolist()
    else:
        # Pure-Python fallback
        # Build normal equations A = X^T X, b = X^T y, then solve A beta = b
        XT = list(zip(*X))
        A = [[sum(XT[i][t]*XT[j][t] for t in range(len(X))) for j in range(len(XT))] for i in range(len(XT))]
        b = [sum(XT[i][t]*y[t] for t in range(len(X))) for i in range(len(XT))]
        return gaussian_elimination(A, b)


def gaussian_elimination(A: List[List[float]], b: List[float]) -> List[float]:
    """Solve A x = b via basic Gaussian elimination with partial pivoting."""
    n = len(A)
    # Augmented matrix
    M = [Ai[:] + [bi] for Ai, bi in zip(A, b)]
    for col in range(n):
        # Pivot
        pivot = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[pivot] = M[pivot], M[col]
        # Normalize pivot row
        pivot_val = M[col][col]
        if abs(pivot_val) < 1e-12:
            raise ValueError("Matrix is singular or ill-conditioned")
        M[col] = [v / pivot_val for v in M[col]]
        # Eliminate below
        for r in range(col + 1, n):
            factor = M[r][col]
            M[r] = [rv - factor * pv for rv, pv in zip(M[r], M[col])]
    # Back-substitution
    x = [0.0] * n
    for i in reversed(range(n)):
        x[i] = M[i][-1] - sum(M[i][j] * x[j] for j in range(i + 1, n))
    return x


def linear_regression_gd(X: List[List[float]], y: List[float],
                         lr: float = 0.01, steps: int = 1000) -> List[float]:
    """Linear regression via gradient descent on MSE loss."""
    m, d = len(X), len(X[0])
    beta = [0.0] * d
    for _ in range(steps):
        # Compute gradient: (2/m) X^T (X beta - y)
        grad = [0.0] * d
        for i in range(m):
            pred = sum(beta[j] * X[i][j] for j in range(d))
            err = pred - y[i]
            for j in range(d):
                grad[j] += (2.0 / m) * X[i][j] * err
        # Update
        beta = [bj - lr * gj for bj, gj in zip(beta, grad)]
    return beta


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def logistic_regression_gd(X: List[List[float]], y: List[int],
                           lr: float = 0.1, steps: int = 1000, l2: float = 0.0) -> List[float]:
    """
    Logistic regression via gradient descent on log-loss.
    Optional L2 regularization controlled by 'l2' (lambda).
    """
    m, d = len(X), len(X[0])
    w = [0.0] * d
    for _ in range(steps):
        grad = [0.0] * d
        for i in range(m):
            z = sum(w[j] * X[i][j] for j in range(d))
            p = sigmoid(z)
            for j in range(d):
                grad[j] += (p - y[i]) * X[i][j] / m
        # L2 regularization gradient
        for j in range(d):
            grad[j] += (l2 / m) * w[j]
        w = [wj - lr * gj for wj, gj in zip(w, grad)]
    return w


def kmeans(X: List[List[float]], k: int, steps: int = 100, init: Optional[List[List[float]]] = None) -> Tuple[List[int], List[List[float]]]:
    """
    Basic k-means clustering.
    Returns labels and centroids.
    """
    n, d = len(X), len(X[0])
    # Initialize centroids
    if init is None:
        centroids = [X[idx][:] for idx in random.sample(range(n), k)]
    else:
        centroids = [c[:] for c in init]
    labels = [0] * n
    for _ in range(steps):
        # Assignment step
        changed = False
        for i in range(n):
            # Choose closest centroid by Euclidean distance
            dists = [math.sqrt(sum((X[i][j] - c[j]) ** 2 for j in range(d))) for c in centroids]
            new_label = min(range(k), key=lambda a: dists[a])
            if new_label != labels[i]:
                labels[i] = new_label
                changed = True
        # Update step
        new_centroids = [[0.0] * d for _ in range(k)]
        counts = [0] * k
        for xi, li in zip(X, labels):
            counts[li] += 1
            for j in range(d):
                new_centroids[li][j] += xi[j]
        for a in range(k):
            if counts[a] > 0:
                new_centroids[a] = [v / counts[a] for v in new_centroids[a]]
            else:
                new_centroids[a] = centroids[a]  # keep old if empty
        centroids = new_centroids
        if not changed:
            break
    return labels, centroids


# -----------------------------
# 7) MARKOV CHAINS
# -----------------------------
def markov_chain_next(state: int, P: List[List[float]]) -> int:
    """Sample the next state given transition matrix P and current state index."""
    r = random.random()
    cum = 0.0
    for j, p in enumerate(P[state]):
        cum += p
        if r <= cum:
            return j
    return len(P[state]) - 1  # numerical fallback


def stationary_distribution_power(P: List[List[float]], tol: float = 1e-10, steps: int = 10_000) -> List[float]:
    """
    Approximate stationary distribution π via power method:
    Start with uniform π, repeatedly multiply by P until convergence.
    π satisfies π = π P.
    """
    n = len(P)
    pi = [1.0 / n] * n
    for _ in range(steps):
        pi_new = [0.0] * n
        for j in range(n):
            pi_new[j] = sum(pi[i] * P[i][j] for i in range(n))
        # L1 distance check
        if sum(abs(a - b) for a, b in zip(pi_new, pi)) < tol:
            return pi_new
        pi = pi_new
    return pi


# -----------------------------
# 8) INFORMATION THEORY
# -----------------------------
def entropy(p: List[float]) -> float:
    """Shannon entropy H(p) = -Σ p_i log2 p_i. Assumes probabilities sum to 1."""
    eps = 1e-12
    return -sum(pi * math.log(pi + eps, 2) for pi in p if pi > 0)


def kl_divergence(p: List[float], q: List[float]) -> float:
    """KL divergence KL(p||q) = Σ p_i log2(p_i / q_i)."""
    eps = 1e-12
    return sum(pi * math.log((pi + eps) / (qi + eps), 2) for pi, qi in zip(p, q) if pi > 0)


# -----------------------------
# DEMOS (run when executing the script)
# -----------------------------
def demo_foundations():
    section("1) FOUNDATIONS — Sets & Combinatorics")
    print("5! =", factorial(5))
    print("P(10, 3) =", permutations(10, 3))
    print("C(10, 3) =", combinations(10, 3))
    u, inter, comp = set_ops_demo()
    print("Union:", u, "| Intersection:", inter, "| Complement (U\\A):", comp)


def demo_probability():
    section("2) PROBABILITY BASICS — Conditional, Bayes, Distributions")
    p_a_given_b = conditional_probability(0.12, 0.3)
    print("P(A|B) example:", round(p_a_given_b, 4))
    p_a_given_bayes = bayes(p_a=0.01, p_b_given_a=0.9, p_b_given_not_a=0.05)
    print("Bayes P(A|B) medical test example:", round(p_a_given_bayes, 4))
    print("Simulated Binomial mean (n=10, p=0.3):", simulate_binomial(10, 0.3))
    print("Poisson PMF λ=3, k=2:", round(poisson_pmf(3, 2), 4))
    print("Normal PDF μ=0, σ=1 at x=1:", round(normal_pdf(0, 1, 1), 4))


def demo_stats():
    section("3) DESCRIPTIVE STATS & INFERENCE — mean/var/corr, CLT, z-test")
    xs = [1, 2, 3, 4, 5]
    ys = [2, 1, 3, 7, 9]
    print("mean(xs) =", mean(xs), "| var(xs) =", round(variance(xs), 4))
    print("cov(xs, ys) =", round(covariance(xs, ys), 4), "| corr(xs, ys) =", round(correlation(xs, ys), 4))
    mu_means, var_means = clt_demo(sample_size=30, reps=2000)
    print("CLT demo — mean of sample-means:", round(mu_means, 3), " var:", round(var_means, 4))
    # Fake z-test example with known σ
    x1 = [2.1, 2.3, 2.5, 2.2, 2.4]
    x2 = [2.0, 2.1, 2.2, 1.9, 2.0]
    z = z_test_two_means(x1, x2, sigma1=0.2, sigma2=0.2)
    print("Two-sample z statistic:", round(z, 3))


def demo_linear_algebra():
    section("4) LINEAR ALGEBRA — dot/norm/matmul, eig, PCA")
    x = [1, 2, 3]
    y = [4, 5, 6]
    print("dot(x,y) =", dot(x, y), " | norm2(x) =", round(norm2(x), 3))
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    print("A*B =", matmul(A, B))
    X = [[2.5, 2.4],
         [0.5, 0.7],
         [2.2, 2.9],
         [1.9, 2.2],
         [3.1, 3.0],
         [2.3, 2.7],
         [2.0, 1.6],
         [1.0, 1.1],
         [1.5, 1.6],
         [1.1, 0.9]]
    info = eigen_and_pca_demo(X)
    print("PCA / Eigen demo info keys:", list(info.keys()))


def demo_calculus_opt():
    section("5) CALCULUS & OPTIMIZATION — numeric derivative & GD")
    f = lambda x: (x - 3) ** 2 + 4  # convex parabola, min at x=3
    dfx_num = numeric_derivative(f, 1.0)
    print("Numeric derivative at x=1:", round(dfx_num, 3))
    xmin, path = gradient_descent_1d(f, x0=0.0, lr=0.2, steps=50)
    print("GD found minimum near x ≈", round(xmin, 3), "in", len(path), "steps")


def demo_ml():
    section("6) MACHINE LEARNING MATH — linear/logistic regression, k-means")
    # Create a tiny synthetic dataset with bias term
    X = [[1, 0.0], [1, 1.0], [1, 2.0], [1, 3.0]]
    y = [1.0, 2.0, 3.0, 4.0]
    beta_cf = linear_regression_closed_form(X, y)
    beta_gd = linear_regression_gd(X, y, lr=0.1, steps=500)
    print("Linear Regression (closed-form):", [round(b, 3) for b in beta_cf])
    print("Linear Regression (gradient descent):", [round(b, 3) for b in beta_gd])

    # Logistic regression on linearly separable mini dataset
    X2 = [[1, -2], [1, -1], [1, 0], [1, 1], [1, 2]]
    y2 = [0, 0, 0, 1, 1]
    w = logistic_regression_gd(X2, y2, lr=0.5, steps=1000, l2=0.01)
    print("Logistic Regression weights:", [round(a, 3) for a in w])

    # k-means on simple 2D points
    X3 = [[0.1, 0.2], [0.2, 0.1], [0.8, 0.9], [0.9, 0.8], [0.05, 0.0], [1.0, 1.1]]
    labels, cents = kmeans(X3, k=2, steps=100)
    print("k-means labels:", labels)
    print("k-means centroids:", [[round(c, 3) for c in cent] for cent in cents])


def demo_markov_info():
    section("7) MARKOV CHAINS & 8) INFORMATION THEORY")
    # Simple 2-state weather model: 0=Sunny, 1=Rainy
    P = [[0.8, 0.2],
         [0.4, 0.6]]
    pi_star = stationary_distribution_power(P)
    print("Stationary distribution ~", [round(p, 3) for p in pi_star])

    # Entropy & KL
    p = [0.25, 0.25, 0.25, 0.25]
    q = [0.1, 0.2, 0.3, 0.4]
    print("Entropy H(p) bits:", round(entropy(p), 3))
    print("KL(p||q) bits:", round(kl_divergence(p, q), 3))


def main():
    demo_foundations()
    demo_probability()
    demo_stats()
    demo_linear_algebra()
    demo_calculus_opt()
    demo_ml()
    demo_markov_info()


if __name__ == "__main__":
    main()
