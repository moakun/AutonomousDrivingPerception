"""Constant-velocity Kalman filter over an arbitrary observation vector.

Used with dim=4 (cx, cy, w, h) for image-space box tracking in M2, and reused
with dim=2 (x, y in BEV meters) for metric tracking in M3. This filter is the
ONLY place velocity is ever estimated in ADP — downstream code must read
smoothed velocity from here, never difference raw positions.

State layout: [z_0..z_{d-1}, vz_0..vz_{d-1}] — observed values then their rates.
Supports variable dt per step (nuScenes camera timestamps are not perfectly
uniform).
"""

from __future__ import annotations

import numpy as np


class ConstantVelocityKalman:
    def __init__(
        self,
        z0: np.ndarray,
        pos_std: float,
        vel_std: float,
        meas_std: float,
        process_std: float,
    ):
        """
        z0: initial observation (d,)
        pos_std / vel_std: initial state uncertainty
        meas_std: measurement noise std
        process_std: process (acceleration-like) noise std per unit time
        """
        z0 = np.asarray(z0, dtype=float)
        self.d = len(z0)
        self.x = np.concatenate([z0, np.zeros(self.d)])
        self.P = np.diag([pos_std**2] * self.d + [vel_std**2] * self.d)
        self.meas_var = meas_std**2
        self.process_var = process_std**2

    @property
    def position(self) -> np.ndarray:
        return self.x[: self.d]

    @property
    def velocity(self) -> np.ndarray:
        return self.x[self.d :]

    def position_std(self) -> np.ndarray:
        return np.sqrt(np.diag(self.P)[: self.d])

    def velocity_std(self) -> np.ndarray:
        return np.sqrt(np.diag(self.P)[self.d :])

    def predict(self, dt: float) -> np.ndarray:
        """Advance state by dt seconds; returns predicted observation."""
        d = self.d
        F = np.eye(2 * d)
        F[:d, d:] = np.eye(d) * dt
        # White-noise acceleration model.
        q = self.process_var
        Q = np.zeros((2 * d, 2 * d))
        Q[:d, :d] = np.eye(d) * (0.25 * dt**4 * q)
        Q[:d, d:] = Q[d:, :d] = np.eye(d) * (0.5 * dt**3 * q)
        Q[d:, d:] = np.eye(d) * (dt**2 * q)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        return self.position.copy()

    def mahalanobis_sq(self, z: np.ndarray, meas_var: float | None = None) -> float:
        """Squared Mahalanobis distance of a measurement from the prediction —
        the innovation gate statistic (chi-square with d dof under the model)."""
        d = self.d
        R = np.eye(d) * (self.meas_var if meas_var is None else meas_var)
        y = np.asarray(z, dtype=float) - self.position
        S = self.P[:d, :d] + R
        return float(y @ np.linalg.solve(S, y))

    def update(self, z: np.ndarray, meas_var: float | None = None) -> None:
        d = self.d
        H = np.zeros((d, 2 * d))
        H[:, :d] = np.eye(d)
        R = np.eye(d) * (self.meas_var if meas_var is None else meas_var)
        y = np.asarray(z, dtype=float) - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(2 * d) - K @ H) @ self.P
