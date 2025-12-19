#!/usr/bin/env python3
"""
Calculate expected value and variance for discrete probability distributions.
"""


def expected_value(distribution):
    """
    Calculate the expected value (mean) of a discrete random variable.

    E[X] = Σ(x_i * p_i)

    Args:
        distribution: Dictionary mapping values to their probabilities

    Returns:
        Expected value (float)
    """
    if abs(sum(distribution.values()) - 1.0) > 1e-6:
        raise ValueError(
            f"Probabilities must sum to 1, got {sum(distribution.values())}"
        )

    return sum(x * p for x, p in distribution.items())


def variance(distribution):
    """
    Calculate the variance of a discrete random variable.

    Var(X) = E[X²] - (E[X])²

    Args:
        distribution: Dictionary mapping values to their probabilities

    Returns:
        Variance (float)
    """
    if abs(sum(distribution.values()) - 1.0) > 1e-6:
        raise ValueError(
            f"Probabilities must sum to 1, got {sum(distribution.values())}"
        )

    # E[X]
    exp_value = expected_value(distribution)

    # E[X²]
    exp_x_squared = sum(x**2 * p for x, p in distribution.items())

    # Var(X) = E[X²] - (E[X])²
    return exp_x_squared - exp_value**2


def standard_deviation(distribution):
    """
    Calculate the standard deviation of a discrete random variable.

    σ = √Var(X)

    Args:
        distribution: Dictionary mapping values to their probabilities

    Returns:
        Standard deviation (float)
    """
    return variance(distribution) ** 0.5


def skewness(distribution):
    """
    Calculate the skewness (third standardized moment) of a discrete random variable.

    Skewness = E[(X - μ)³] / σ³

    Args:
        distribution: Dictionary mapping values to their probabilities

    Returns:
        Skewness (float)
    """
    if abs(sum(distribution.values()) - 1.0) > 1e-6:
        raise ValueError(
            f"Probabilities must sum to 1, got {sum(distribution.values())}"
        )

    mu = expected_value(distribution)
    sigma = standard_deviation(distribution)

    if sigma == 0:
        return 0.0

    # E[(X - μ)³]
    third_moment = sum(((x - mu) ** 3) * p for x, p in distribution.items())

    return third_moment / (sigma**3)


def kurtosis(distribution):
    """
    Calculate the kurtosis (fourth standardized moment) of a discrete random variable.

    Kurtosis = E[(X - μ)⁴] / σ⁴

    Args:
        distribution: Dictionary mapping values to their probabilities

    Returns:
        Kurtosis (float)
    """
    if abs(sum(distribution.values()) - 1.0) > 1e-6:
        raise ValueError(
            f"Probabilities must sum to 1, got {sum(distribution.values())}"
        )

    mu = expected_value(distribution)
    sigma = standard_deviation(distribution)

    if sigma == 0:
        return 0.0

    # E[(X - μ)⁴]
    fourth_moment = sum(((x - mu) ** 4) * p for x, p in distribution.items())

    return fourth_moment / (sigma**4)


def main():
    """Calculate expected value and variance for your distribution"""
    print("Expected Value and Variance Calculator")
    print("=" * 50)

    # Format: {value: probability, value: probability, ...}
    distribution = {1: 1 / 6, 2: 1 / 6, 3: 1 / 6, 4: 1 / 6, 5: 1 / 6, 6: 1 / 6}

    ev = expected_value(distribution)
    var = variance(distribution)
    std = standard_deviation(distribution)
    skew = skewness(distribution)
    kurt = kurtosis(distribution)

    print(f"\nResults:")
    print(f"Distribution: {distribution}")
    print(f"Expected Value: {ev:.4f}")
    print(f"Variance: {var:.4f}")
    print(f"Standard Deviation: {std:.4f}")
    print(f"Skewness (3rd moment): {skew:.4f}")
    print(f"Kurtosis (4th moment): {kurt:.4f}")


if __name__ == "__main__":
    main()
