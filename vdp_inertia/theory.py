"""Compute E0, Gamma = L1(E0), and D = integral Gamma along the flow.

L1 is the original-coordinate Laplacian, or its full Ito-transformed operator
when the absorbing rectangle is defined in Lienard coordinates.

First solve the trajectory forward to its first box exit. Then transport the
exit-time gradient and Hessian backward along that stored trajectory. This is
the whole-trajectory form of the manuscript's four-step differentiation.
"""
import numpy as np
from scipy.integrate import solve_ivp
from .model import drift, jacobian, hessians, gamma, validate_coordinates

NORMALS = np.array([[-1., 0.], [1., 0.], [0., -1.], [0., 1.]])


def exit_trajectory(initial, mu, box, *, rtol=1e-9, atol=1e-11,
                    max_step=.1, max_time=100., coordinates='original'):
    """Find the earliest outward crossing of any side of the open rectangle."""
    validate_coordinates(coordinates)
    if not np.isfinite(mu) or mu <= 0:
        raise ValueError('mu must be finite and positive.')
    xmin, xmax, ymin, ymax = box
    x, y = initial
    if not (xmin < x < xmax and ymin < y < ymax):
        raise ValueError("The initial point must lie strictly inside the box.")
    events = []
    for normal, level in zip(NORMALS, (-xmin, xmax, -ymin, ymax)):
        def event(t, z, normal=normal, level=level):
            return normal @ z - level
        event.terminal = True
        event.direction = 1
        events.append(event)
    path = solve_ivp(lambda t, z: drift(z, mu, coordinates),
                     (0., max_time), initial, events=events, dense_output=True,
                     method="DOP853", rtol=rtol, atol=atol, max_step=max_step)
    if not path.success:
        raise RuntimeError(path.message)
    hits = [(ts[0], side) for side, ts in enumerate(path.t_events) if len(ts)]
    if not hits:
        raise RuntimeError(f"No deterministic exit before t={max_time}.")
    T, side = min(hits)
    theta = path.sol(T)
    n = NORMALS[side]
    corner_distance = (min(theta[1]-ymin, ymax-theta[1]) if side < 2
                       else min(theta[0]-xmin, xmax-theta[0]))
    smooth_exit = corner_distance > 1e-6 and n @ drift(theta, mu, coordinates) > 1e-9
    return path, float(T), theta, side, smooth_exit


def boundary_derivatives(theta, normal, mu, coordinates='original'):
    """Gradient g and Hessian H of the exit time on the exiting planar face."""
    b = drift(theta, mu, coordinates)
    d = normal @ b
    if d <= 0:
        raise ValueError('The terminal face must be crossed transversely outward.')
    a = jacobian(theta, mu, coordinates).T @ normal
    g = -normal / d
    H = ((np.outer(normal, a) + np.outer(a, normal)) / d**2
         - (a @ b) * np.outer(normal, normal) / d**3)
    return g, H


def response(initial, mu, box, *, samples=0, rtol=1e-9, atol=1e-11,
             max_step=.1, max_time=100., coordinates='original'):
    """Return E0, pointwise Gamma, and integrated response D at one point.

    samples > 0 also returns Gamma and the remaining integral along the path.
    Corner/tangent exits get NaN derivatives; no smooth-face formula is used.
    """
    path, T, theta, side, smooth = exit_trajectory(
        initial, mu, box, rtol=rtol, atol=atol, max_step=max_step, max_time=max_time,
        coordinates=coordinates)
    result = {"E0": T, "Gamma": np.nan, "D": np.nan, "face": side, "smooth": smooth}
    if not smooth:
        return result

    gT, HT = boundary_derivatives(theta, NORMALS[side], mu, coordinates)
    terminal = np.r_[gT, HT.ravel(), 0.]  # [gradient(2), Hessian(4), integral(1)]

    def transport(t, state):
        z = path.sol(t)  # Evaluate the stored forward path; never reverse its ODE.
        g, H = state[:2], state[2:6].reshape(2, 2)
        A = jacobian(z, mu, coordinates)
        dg = -A.T @ g
        dH = -A.T @ H - H @ A - np.einsum("k,kij->ij", g, hessians(z, mu, coordinates))
        dJ = -gamma(z, g, H, mu, coordinates)  # Backward from J(T)=0.
        return np.r_[dg, dH.ravel(), dJ]

    backward = solve_ivp(transport, (T, 0.), terminal, method="DOP853",
                         dense_output=samples > 0, rtol=rtol, atol=atol, max_step=max_step)
    if not backward.success:
        raise RuntimeError(backward.message)
    start = backward.y[:, -1]
    H0 = start[2:6].reshape(2, 2)
    result.update(Gamma=float(gamma(initial, start[:2], H0, mu, coordinates)), D=float(start[6]),
                  gradient=start[:2], hessian=H0)
    if samples:
        t = np.linspace(0., T, samples)
        states = backward.sol(t)
        points = path.sol(t).T
        gamma_path = [gamma(z, q[:2], q[2:6].reshape(2, 2), mu, coordinates)
                      for z, q in zip(points, states.T)]
        result.update(t=t, path=points, gamma_path=np.array(gamma_path),
                      remaining_integral=states[6])
    return result
