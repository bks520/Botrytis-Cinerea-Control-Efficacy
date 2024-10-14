import numpy as np
import matplotlib.pyplot as plt


def basis_function(t, i, k, knots):
    if k == 0:
        return 1.0 if knots[i] <= t < knots[i + 1] else 0.0
    else:
        alpha = (t - knots[i]) / (knots[i + k] - knots[i]) if (knots[i + k] - knots[i]) != 0 else 0.0
        beta = (knots[i + k + 1] - t) / (knots[i + k + 1] - knots[i + 1]) if (knots[i + k + 1] - knots[
            i + 1]) != 0 else 0.0
        return alpha * basis_function(t, i, k - 1, knots) + beta * basis_function(t, i + 1, k - 1, knots)


def generate_basis_functions(t, k, knots):
    n = len(knots) - k - 1
    basis_functions = np.zeros((n, len(t)))

    for i in range(n):
        basis_functions[i] = [basis_function(ti, i, k, knots) for ti in t]

    return basis_functions


def hierarchical_truncated_spline(k, knots, t_range):
    t = np.linspace(t_range[0], t_range[1], 1000)

    basis_k = generate_basis_functions(t, k, knots)
    basis_k_plus_one = generate_basis_functions(t, k + 1, knots)

    min_columns = min(basis_k.shape[1], basis_k_plus_one.shape[1])
    active_basis_k = basis_k[:, :min_columns]
    active_basis_k_plus_one = basis_k_plus_one[:, :min_columns]

    hierarchical_basis = np.vstack([active_basis_k, active_basis_k_plus_one])

    return t, hierarchical_basis

k = 3

#节点向量
knots = [0, 0, 0, 0, 1, 2, 3, 4, 5, 5, 5, 5]

t_range = (0, 5)

t, hierarchical_basis = hierarchical_truncated_spline(k, knots, t_range)

plt.figure(figsize=(10, 6))
plt.plot(t, hierarchical_basis.T)
plt.title(f'Hierarchical Truncated Spline (degree={k})')
plt.xlabel('t')
plt.ylabel('Basis Functions')
plt.show()
