import numpy as np
from typing import Dict, Any, List, Tuple
from scipy import stats

def compute_bootstrap_ci(data: np.ndarray, num_bootstrap: int = 10000, alpha: float = 0.05, seed: int = 42) -> Tuple[float, float]:
    """Computes two-sided bootstrap confidence interval for the mean."""
    if len(data) == 0:
        return 0.0, 0.0
    if len(data) == 1:
        return float(data[0]), float(data[0])
    rng = np.random.default_rng(seed)
    boot_means = rng.choice(data, size=(num_bootstrap, len(data)), replace=True).mean(axis=1)
    low = float(np.percentile(boot_means, 100 * (alpha / 2)))
    high = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return low, high

def compute_wilson_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Computes Wilson score interval for a binomial proportion."""
    if total == 0:
        return 0.0, 0.0
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p_hat = successes / total
    denom = 1 + (z**2) / total
    center = (p_hat + (z**2) / (2 * total)) / denom
    margin = (z / denom) * np.sqrt((p_hat * (1 - p_hat) / total) + (z**2) / (4 * total**2))
    return max(0.0, float(center - margin)), min(1.0, float(center + margin))

def compute_cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Computes paired Cohen's d effect size."""
    diff = x - y
    std_diff = np.std(diff, ddof=1)
    if std_diff < 1e-9:
        return 0.0
    return float(np.mean(diff) / std_diff)

def compute_paired_comparison(a_samples: np.ndarray, b_samples: np.ndarray) -> Dict[str, Any]:
    """
    Performs paired Wilcoxon signed-rank test and computes effect sizes.
    """
    if len(a_samples) != len(b_samples) or len(a_samples) < 3:
        return {"p_value": 1.0, "statistic": 0.0, "cohens_d": 0.0}

    diff = a_samples - b_samples
    if np.allclose(diff, 0.0):
        return {"p_value": 1.0, "statistic": 0.0, "cohens_d": 0.0}

    try:
        w_res = stats.wilcoxon(a_samples, b_samples, zero_method="wilcox")
        p_val = float(w_res.pvalue)
        stat = float(w_res.statistic)
    except Exception:
        p_val = 1.0
        stat = 0.0

    d = compute_cohens_d(a_samples, b_samples)
    return {
        "p_value": p_val,
        "statistic": stat,
        "cohens_d": d
    }

def apply_holm_bonferroni(p_values: List[float]) -> List[float]:
    """Adjusts p-values using the step-down Holm-Bonferroni method."""
    m = len(p_values)
    if m == 0:
        return []
    sorted_indices = np.argsort(p_values)
    adjusted = np.zeros(m)
    running_max = 0.0
    for rank, idx in enumerate(sorted_indices):
        p_val = p_values[idx]
        adj = p_val * (m - rank)
        running_max = max(running_max, adj)
        adjusted[idx] = min(1.0, running_max)
    return adjusted.tolist()
