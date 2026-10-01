"""Van der Pol drift and coordinate transformation, in original time t.

dZ = v(Z) dt + sqrt(2*sigma) dW, with independent noise in both coordinates.
"""
import numpy as np


def validate_coordinates(coordinates):
    """Reject misspellings instead of silently using a different model."""
    if coordinates not in ('original', 'lienard'):
        raise ValueError("coordinates must be 'original' or 'lienard'.")


def drift(z, mu, coordinates='original'):
    """Vector drift; also accepts arrays whose last dimension is two."""
    validate_coordinates(coordinates)
    x, y = np.asarray(z)[..., 0], np.asarray(z)[..., 1]
    if coordinates == 'lienard':
        return np.stack((mu * (y - cubic(x)), -x / mu), axis=-1)
    return np.stack((y, mu*(1.0-x*x)*y-x), axis=-1)


def jacobian(z, mu, coordinates='original'):
    """Jacobian of the deterministic drift in the chosen coordinates."""
    validate_coordinates(coordinates)
    x, y = z
    if coordinates == 'lienard':
        return np.array([[mu*(1.-x*x), mu], [-1./mu, 0.]])
    return np.array([[0., 1.], [-2.*mu*x*y-1., mu*(1.-x*x)]])


def hessians(z, mu, coordinates='original'):
    """H[k,i,j] = second derivative of v_k in coordinates i,j."""
    validate_coordinates(coordinates)
    x, y = z
    if coordinates == 'lienard':
        return np.array([[[-2.*mu*x, 0.], [0., 0.]],
                         [[0., 0.], [0., 0.]]])
    return np.array([[[0., 0.], [0., 0.]],
                     [[-2.*mu*y, -2.*mu*x], [-2.*mu*x, 0.]]])


def cubic(x):
    """Critical curve w = x^3/3 - x in Lienard coordinates."""
    return x**3 / 3 - x


def to_lienard(z, mu):
    """Map points from (x,y) to (x,w), with epsilon = 1/mu."""
    z = np.asarray(z)
    return np.stack((z[..., 0], z[..., 1] / mu + cubic(z[..., 0])), axis=-1)


def gamma(z, gradient, hessian, mu, coordinates='original'):
    """Apply the noise generator to E0, including the transformed Ito drift.

    Original additive noise has generator sigma*Laplacian. In (x,w),
    B=[[1,0],[x*x-1,1/mu]] and the extra drift is sigma*(0,2*x).
    """
    validate_coordinates(coordinates)
    if coordinates == 'original':
        return np.trace(hessian)
    x = z[0]
    B = np.array([[1., 0.], [x*x - 1., 1./mu]])
    return np.sum((B @ B.T) * hessian) + 2*x*gradient[1]
