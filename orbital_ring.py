import tensorflow as tf
import numpy as np
import pandas as pd
import pint
from dataclasses import dataclass, field
import dataclasses as dc
from typing import List, Tuple
import os
from datetime import datetime
import numba

#############################################################################
# Computational context
#############################################################################


@dataclass(frozen=True)
class Context:
    """Context enables tensors with units.
  
  TensorFlow does not know about units, and the `pint` package for units does not have TenserFlow support, so we have to remove/replace the units when calling TensorFlow functions.
  """
    ureg: pint.UnitRegistry
    use_units: bool = True

    def without_units(self):
        return dc.replace(self, use_units=False)

    def Q(self, *args, **kwargs):
        res = self.ureg.Quantity(*args, **kwargs)
        if not self.use_units:
            return res.to_base_units().magnitude
        else:
            return res

    def resolve(self, input):
        if isinstance(input, self.ureg.Quantity) and not self.use_units:
            return input.to_base_units().magnitude
        else:
            return input

    def strip_units(self, input):
        if self.use_units:
            return input.magnitude, input.units
        else:
            return input, None

    def add_units(self, input, units):
        if self.use_units:
            return self.ureg.Quantity(input, units)
        else:
            return input

    def m_as(self, input, units):
        if self.use_units:
            if isinstance(input, self.ureg.Quantity):
                return input.m_as(units)
            return input
        else:
            return input

    def constant(self, value, units, name='Const'):
        return self.Q(tf.constant(value, dtype=tf.float64, name=name), units)

    def norm(self, input, axis=None):
        """Returns the L2-norm
    """
        input, units = self.strip_units(input)
        res = tf.sqrt(tf.reduce_sum(input * input, axis=axis))
        return self.add_units(res, units)

    def sum(self, input, axis=None):
        input, units = self.strip_units(input)
        res = tf.reduce_sum(input, axis=axis)
        return self.add_units(res, units)

    def mean(self, input, axis=None):
        input, units = self.strip_units(input)
        res = tf.reduce_mean(input, axis=axis)
        return self.add_units(res, units)

    def min(self, input, axis=None):
        input, units = self.strip_units(input)
        res = tf.reduce_min(input, axis=axis)
        return self.add_units(res, units)

    def max(self, input, axis=None):
        input, units = self.strip_units(input)
        res = tf.reduce_max(input, axis=axis)
        return self.add_units(res, units)

    def clip(self, input, value_min, value_max):
        input, units = self.strip_units(input)
        value_min = self.m_as(value_min, units)
        value_max = self.m_as(value_max, units)
        res = tf.clip_by_value(input, value_min, value_max)
        return self.add_units(res, units)

    def expand_dims(self, input, axis):
        input, units = self.strip_units(input)
        res = tf.expand_dims(input, axis=axis)
        return self.add_units(res, units)

    def linspace(self, start, stop, num, name=None, axis=0):
        start, start_u = self.strip_units(start)
        stop = self.m_as(stop, start_u)
        res = tf.linspace(start, stop, num, name, axis)
        return self.add_units(res, start_u)

    def cos(self, input, name=None):
        input = self.m_as(input, self.ureg.rad)
        return self.add_units(tf.cos(input, name), None)

    def sin(self, input, name=None):
        input = self.m_as(input, self.ureg.rad)
        return self.add_units(tf.sin(input, name), None)

    def floor(self, input, name=None):
        input = self.m_as(input, self.ureg.rad)
        return self.add_units(tf.floor(input, name), None)

    def zeros_like(self, input, dtype=None, name=None):
        input, units = self.strip_units(input)
        res = tf.zeros_like(input, dtype, name)
        return self.add_units(res, units)

    def sqrt(self, input, name=None):
        input, units = self.strip_units(input)
        res = tf.sqrt(input, name)
        return self.add_units(res, units**0.5)

    def stack(self, values, axis=0, name='stack', units=None):
        stripped_values = [self.strip_units(v) for v in values]
        if units is None:
            units = stripped_values[0][1]
        values = [self.m_as(v, units) for v in values]
        res = tf.stack(values, axis, name)
        return self.add_units(res, units)

    def reshape(self, input, shape, name=None):
        input, units = self.strip_units(input)
        res = tf.reshape(input, shape, name)
        return self.add_units(res, units)

    def roll(self, input, shift, axis, name=None):
        input, units = self.strip_units(input)
        res = tf.roll(input, shift, axis, name)
        return self.add_units(res, units)

    def reverse(self, input, axis, name=None):
        input, units = self.strip_units(input)
        res = tf.reverse(input, axis, name)
        return self.add_units(res, units)

    def maximum(self, x, y, name=None):
        units = None
        if isinstance(x, self.ureg.Quantity):
            x, units = self.strip_units(x)
            if isinstance(y, self.ureg.Quantity):
                y = self.m_as(y, units)
        elif isinstance(x, self.ureg.Quantity):
            y, units = self.strip_units(y)
        res = tf.maximum(x, y, name)
        return self.add_units(res, units)

    def minimum(self, x, y, name=None):
        units = None
        if isinstance(x, self.ureg.Quantity):
            x, units = self.strip_units(x)
            if isinstance(y, self.ureg.Quantity):
                y = self.m_as(y, units)
        elif isinstance(x, self.ureg.Quantity):
            y, units = self.strip_units(y)
        res = tf.minimum(x, y, name)
        return self.add_units(res, units)

    def concat(self, values, axis, name='concat', units=None):
        stripped_values = [self.strip_units(v) for v in values]
        if units is None:
            units = stripped_values[0][1]
        values = [self.m_as(v, units) for v in values]
        res = tf.concat(values, axis, name)
        return self.add_units(res, units)

    def argmin(self, input, axis=None, output_type=tf.int64, name=None):
        input, units = self.strip_units(input)
        return tf.argmin(input, axis, output_type, name)

    def argmax(self, input, axis=None, output_type=tf.int64, name=None):
        input, units = self.strip_units(input)
        return tf.argmax(input, axis, output_type, name)

    def gather(self,
               params,
               indices,
               validate_indices=None,
               axis=None,
               batch_dims=0,
               name=None):
        params, units = self.strip_units(params)
        res = tf.gather(params, indices, validate_indices, axis, batch_dims,
                        name)
        return self.add_units(res, units)

    def tensor_scatter_nd_add(self, tensor, indices, updates, name=None):
        tensor, units = self.strip_units(tensor)
        updates = self.m_as(updates, units)
        res = tf.tensor_scatter_nd_add(tensor, indices, updates, name)
        return self.add_units(res, units)


def point_to_segment(ctx, p, s0, s1, axis=0):
    """Find the closest point to `p` on the segment `(s0,s1)`.
  """
    a = s1 - s0
    b = p - s0
    t = ctx.sum(a * b, axis=axis) / ctx.sum(a * a, axis=axis)
    t = ctx.clip(t, 0, 1)

    return t, (s0 * (1 - t) + s1 * t)


ctx = Context(pint.UnitRegistry())
Q_ = ctx.ureg.Quantity

#############################################################################
# Save/load simulation state
#############################################################################


def save_state(t, p, q, file):
    np.savez(file, t=t.m_as('s'), p=p.m_as('m'), q=q.m_as('kg m/s'))


def load_state(file):
    arrays = np.load(file)
    t = Q_(arrays['t'], 's')
    p = Q_(arrays['p'], 'm')
    q = Q_(arrays['q'], 'kg m/s')
    return t, p, q


#############################################################################
# Constants
#############################################################################

G = ctx.constant(6.67430e-11, 'm^3/kg/s^2', name='G')

#############################################################################
# Material properties
#############################################################################


@dataclass(frozen=True)
class Material:
    density: Q_
    elasticity: Q_
    tensile_strength: Q_

    def __post_init__(self):
        object.__setattr__(self, 'density', self.density.to('kg / m^3'))
        object.__setattr__(self, 'elasticity',
                           self.elasticity.to('kg / (m s^2)'))
        object.__setattr__(self, 'tensile_strength',
                           self.tensile_strength.to('kg / (m s^2)'))

    @property
    def maximum_stretch_ratio(self):
        """Returns the maximum ratio by which the material can be stretched before
    the elastic force exceeds the tensile strength.
    """
        return 1.0 + self.tensile_strength / self.elasticity

    @property
    def spring_coef(self, cross_section, rest_length):
        """Returns the spring coefficient

    This is valid for a solid block of material with a given cross section
    (orthogonal to the stretching direction) and a given rest length (parallel
    to the stretching direction).
    """
        return self.elasticity * cross_section / rest_length


steel = Material(
    density=ctx.constant(8000, 'kg/m^3'),
    elasticity=ctx.constant(2e11, 'Pa'),
    tensile_strength=ctx.constant(5e8, 'Pa'),
)

carbon_fiber = Material(
    density=ctx.constant(2000, 'kg/m^3'),
    elasticity=ctx.constant(1.5e11, 'Pa'),
    tensile_strength=ctx.constant(2e9, 'Pa'),
)

#############################################################################
# Bodies
#############################################################################


@dataclass(frozen=True)
class Body:
    mass: 'Q'
    radius: 'Q'
    angular_velocity: 'Q'

    def __post_init__(self):
        object.__setattr__(self, 'mass', self.mass.to('kg'))
        object.__setattr__(self, 'radius', self.radius.to('m'))
        object.__setattr__(self, 'angular_velocity',
                           self.angular_velocity.to('rad / s'))

    @property
    def surface_gravity(self):
        return G * self.mass / (self.radius * self.radius)


earth = Body(
    mass=ctx.constant(5.972e24, 'kg'),
    radius=ctx.constant(6.371e6, 'm'),
    angular_velocity=ctx.constant(2 * np.pi, 'rad / day'),
)

#############################################################################
# Energy
#############################################################################


def gravitational_energy(ctx: Context, position0: Q_, mass0: Q_, position1: Q_,
                         mass1: Q_) -> Q_:
    """Returns the gravitational potential energy between 2 sets of particles

  Note: this does not include any gravitational energy among particles in the
  same set.

  >>> gravitational_energy(ctx,
                           ctx.constant([0,0,0],'m'),
                           earth.mass,
                           ctx.constant([[1, 0], 
                                         [0, 2],
                                         [0,0]], '') * earth.radius,
                           ctx.constant([3, 5], 'kg'))
  -344096778.8416261 kilogram meter2/second2
  """
    if len(position0.shape) < len(position1.shape):
        position0 = ctx.expand_dims(position0, axis=1)
    if len(position1.shape) < len(position0.shape):
        position1 = ctx.expand_dims(position1, axis=1)
    r = ctx.norm(position1 - position0, axis=0)
    return -ctx.resolve(G) * ctx.sum(mass0 * mass1 / r)


def kinetic_energy(ctx: Context, momentum: Q_, mass: Q_) -> Q_:
    """Returns the kinetic energy of a set of particles

  >>> kinetic_energy(ctx,
                     ctx.constant([[0, 7e3], 
                                   [-7e3, 0],
                                   [0, 0]], 'kg m/s'),
                     ctx.constant([1,2], 'kg'))
  36750000.0 kilogram meter2/second2
  """
    return 0.5 * ctx.sum(momentum * momentum / mass)


def elastic_energy(ctx: Context,
                   position: Q_,
                   rest_length: Q_,
                   spring: Q_,
                   cycle: bool = False):
    """Returns the potential energy in a chain of springs

  Note: this assumes each spring exerts a force when stretched *longer* than
  rest_length, but does *not* exert any force when compressed *shorter* than
  rest_length.

  >>> elastic_energy(ctx,
                     position=ctx.constant([[1, 0, 0.5],
                                            [0, 1, 0.5]], 'm'),
                     rest_length=ctx.constant(0.2, 'm'),
                     spring=ctx.constant(0.7, 'kg/s^2'),
                     cycle=True)
  0.6960202025355334 kilogram meter2/second2

  >>> elastic_energy(ctx,
                     position=ctx.constant([[1, 0, 0.5],
                                            [0, 1, 0.5]], 'm'),
                     rest_length=ctx.constant(0.2, 'm'),
                     spring=ctx.constant(0.7, 'kg/s^2'),
                     cycle=False)
  0.6060151519016501 kilogram meter2/second2
  """
    if cycle:
        d = ctx.roll(position, shift=-1, axis=1) - position
    else:
        d = position[:, 1:] - position[:, :-1]
    length = ctx.norm(d, axis=0)
    stretch = ctx.maximum(length - rest_length, 0.0)
    return 0.5 * ctx.sum(spring * stretch * stretch)


#############################################################################
# Mechanics
#############################################################################
def gradient(ctx, H, t, p, q):
    """Computes the gradient of the Hamiltonian with respect to position and momentum

  By Hamilton's equations, these give the time derivatives of momentum and
  position, respectively.
  """
    with tf.GradientTape() as tape:
        p_, _ = ctx.strip_units(p)
        q_, _ = ctx.strip_units(q)
        tape.watch(p_)
        tape.watch(q_)
        h = H(ctx, t, p, q)
        h_, _ = ctx.strip_units(h)
    grad_p, grad_q = tape.gradient(h_, [p_, q_])
    if grad_p is not None:
        grad_p = ctx.Q(grad_p, 'kg m / s^2')
    if grad_q is not None:
        grad_q = ctx.Q(grad_q, 'm / s')
    return grad_p, grad_q


def step_linear(ctx, H, t, p, q, dt):
    """Computes new position/momentum vectors using a simple linear approximation

  Note: step_rk4 is much more accurate
  """
    dq_dt, dp_dt = gradient(ctx, H, t, p, q)
    p1 = p - dp_dt * dt
    q1 = q + dq_dt * dt
    return p1, q1


def step_rk4(ctx, H, t, p, q, dt):
    """Computes new position/momentum vectors using the Runge-Kutta method (RK4)
  """
    p1, q1, t1 = p, q, t
    dq1, dp1 = gradient(ctx, H, t1, p1, q1)

    p2 = p1 + 0.5 * dt * dp1
    q2 = q1 - 0.5 * dt * dq1
    t2 = t1 + 0.5 * dt
    dq2, dp2 = gradient(ctx, H, t2, p2, q2)

    p3 = p1 + 0.5 * dt * dp2
    q3 = q1 - 0.5 * dt * dq2
    t3 = t1 + 0.5 * dt
    dq3, dp3 = gradient(ctx, H, t3, p3, q3)

    p4 = p1 + dt * dp3
    q4 = q1 - dt * dq3
    t4 = t + dt
    dq4, dp4 = gradient(ctx, H, t4, p4, q4)

    t = t + dt
    p = p1 + (dp1 + 2 * dp2 + 2 * dp3 + dp4) * dt / 6
    q = q1 - (dq1 + 2 * dq2 + 2 * dq3 + dq4) * dt / 6

    return p, q


def simulate(H,
             t0,
             p0,
             q0,
             dt,
             steps,
             checkpoint_steps,
             post_update=None,
             use_tensorflow=False):
    """Simulate multiple time steps, returning the time/position/momentum
  """
    checkpoint_count = (steps + checkpoint_steps - 1) // checkpoint_steps
    t_checkpoints = ctx.Q(
        np.zeros(checkpoint_count, dtype=t0.dtype.as_numpy_dtype()), 's')
    p_checkpoints = ctx.Q(
        np.zeros((checkpoint_count,) + tuple(p0.shape),
                 dtype=p0.dtype.as_numpy_dtype()), 'm')
    q_checkpoints = ctx.Q(
        np.zeros((checkpoint_count,) + tuple(q0.shape),
                 dtype=q0.dtype.as_numpy_dtype()), 'kg m / s')
    t_var = tf.Variable(t0.m_as('s'))
    p_var = tf.Variable(p0.m_as('m'))
    q_var = tf.Variable(q0.m_as('kg m / s'))
    t = ctx.Q(t_var, 's')
    p = ctx.Q(p_var, 'm')
    q = ctx.Q(q_var, 'kg m / s')
    checkpoint_index = 0

    # @ctx.ureg.wraps('J', ('s', 'm', 'kg m/s', 's'))
    # @tf.function
    def _sim(ctx, H, t, p, q, dt):
        for i in range(checkpoint_steps):
            p1, q1 = step_rk4(ctx, H, t, p, q, dt)
            if post_update is not None:
                p2, q2 = post_update(ctx, t + dt, p1, q1, dt)
            else:
                p2, q2 = p1, q1
            t_var.assign(ctx.m_as(t + dt, 's'))
            p_var.assign(ctx.m_as(p2, 'm'))
            q_var.assign(ctx.m_as(q2, 'kg m/s'))

    @ctx.ureg.wraps(None, (None, None, 's', 'm', 'kg m/s', 's'))
    @tf.function
    def _sim_tf(ctx, H, t, p, q, dt):
        return _sim(ctx.without_units(), H, t, p, q, dt)

    for i in range(steps // checkpoint_steps):
        if use_tensorflow:
            _sim_tf(ctx, H, t, p, q, dt)
        else:
            _sim(ctx, H, t, p, q, dt)
        print(f'checkpoint {checkpoint_index} ({H(ctx,t,p,q)})')
        t_checkpoints[checkpoint_index] = ctx.Q(t.m_as('s').numpy(), 's')
        p_checkpoints[checkpoint_index, ...] = ctx.Q(p.m_as('m').numpy(), 'm')
        q_checkpoints[checkpoint_index,
                      ...] = ctx.Q(q.m_as('kg m/s').numpy(), 'kg m/s')
        checkpoint_index += 1
    return t_checkpoints, p_checkpoints, q_checkpoints


#############################################################################
# Orbits
#############################################################################
@dataclass(frozen=True)
class Orbit:
    """An elliptic orbit
  """
    body: Body
    semi_major_axis: Q_
    eccentricity: Q_

    def __post_init__(self):
        object.__setattr__(self, 'semi_major_axis',
                           self.semi_major_axis.to('m'))
        if isinstance(self.eccentricity, ctx.ureg.Quantity):
            object.__setattr__(self, 'eccentricity', self.eccentricity.to(''))
        else:
            object.__setattr__(self, 'eccentricity',
                               ctx.constant(self.eccentricity, ''))

    @property
    def period(self):
        """The time to complete one orbit
    """
        a = self.semi_major_axis
        return 2 * np.pi * ctx.sqrt(a * a * a / (G * self.body.mass))

    def points(self, count, theta0=None, theta1=None):
        """Computes points along a section of the orbit, equally spaced in time
    """
        dt_init = 0.25 * self.period / count

        mu = G * self.body.mass
        r_periapsis = (1 - self.eccentricity) * self.semi_major_axis
        p0 = ctx.constant([1, 0], '') * r_periapsis
        v0 = ctx.constant([0, 1], '') * np.sqrt(
            mu * (2 / r_periapsis - 1 / self.semi_major_axis))

        if theta0 is not None:
            t0, p1, v1 = _angle_to_time(theta0, mu, p0, v0, dt_init)
        else:
            t0 = ctx.constant(0, 's')
            p1, v1 = p0, v0

        if theta1 is not None:
            t1, _, _ = _angle_to_time(theta1, mu, p1, v1, dt_init)
        else:
            t1 = self.period

        t = ctx.linspace(t0, t0 + t1, count + 1)
        dt = t1 / count

        p, v = _build_orbit(mu, p1, v1, dt, count)
        return t, p, v


@numba.jit(nopython=True)
def _gravitational_acceleration_jit(mu, p):
    r_squared = np.sum(p * p)
    r = np.sqrt(r_squared)
    return -mu * p / (r * r_squared)


@numba.jit(nopython=True)
def _orbit_rk4_jit(mu, p, v, dt):
    p1 = p
    v1 = v
    a1 = _gravitational_acceleration_jit(mu, p1)

    p2 = p + 0.5 * dt * v1
    v2 = v + 0.5 * dt * a1
    a2 = _gravitational_acceleration_jit(mu, p2)

    p3 = p + 0.5 * dt * v2
    v3 = v + 0.5 * dt * a2
    a3 = _gravitational_acceleration_jit(mu, p3)

    p4 = p + dt * v3
    v4 = v + dt * a3
    a4 = _gravitational_acceleration_jit(mu, p4)

    p_next = p1 + (v1 + 2 * v2 + 2 * v3 + v4) * dt / 6
    v_next = v1 + (a1 + 2 * a2 + 2 * a3 + a4) * dt / 6
    return p_next, v_next


@numba.jit(nopython=True)
def _intersect_segments_jit(a1, a2, b1, b2):
    s = np.vstack((a1, a2, b1, b2))
    h = np.hstack((s, np.ones((4, 1))))
    l1 = np.cross(h[0], h[1])
    l2 = np.cross(h[2], h[3])
    p = np.cross(l1, l2)
    return p[0:2] / p[2]


@numba.jit(nopython=True)
def _angle_to_time_jit(theta, mu, p0, v0, dt):
    p_cur = p0
    v_cur = v0
    theta0 = np.arctan2(p0[1], p0[0])
    theta = np.remainder(theta - theta0, 2 * np.pi)
    theta_cur = np.float64(0.0)
    t_cur = np.float64(0.0)

    while True:
        p_next, v_next = _orbit_rk4_jit(mu, p_cur, v_cur, dt)
        theta_next = np.remainder(
            np.arctan2(p_next[1], p_next[0]) - theta0, 2 * np.pi)
        if theta_next > theta or theta_next < theta_cur:
            break
        p_cur, v_cur, theta_cur = p_next, v_next, theta_next
        t_cur += dt

    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]],
                   dtype=np.float64)
    dir = np.sum(rot * np.expand_dims(p0, axis=0), axis=1)

    intersect = _intersect_segments_jit(np.zeros_like(dir), dir, p_cur, p_next)
    dp = p_next - p_cur
    dp_intersect = intersect - p_cur
    t_intersect = dt * np.sqrt(np.sum(dp_intersect * dp_intersect)) / np.sqrt(
        np.sum(dp * dp))
    t_final = t_cur + t_intersect
    p_final, v_final = _orbit_rk4_jit(mu, p_cur, v_cur, t_intersect)
    t_final = np.array((t_final,))
    return np.concatenate((t_final, p_final, v_final), axis=0)


@ctx.ureg.wraps(('s', 'm', 'm/s'), ('rad', 'm^3 / s^2', 'm', 'm / s', 's'))
def _angle_to_time(theta, mu, p0, v0, dt):
    arr = _angle_to_time_jit(theta.numpy(), mu.numpy(), p0.numpy(), v0.numpy(),
                             dt.numpy())
    t = tf.constant(arr[0])
    p = tf.constant(arr[1:3])
    v = tf.constant(arr[3:5])
    return t, p, v


@numba.jit(nopython=True)
def _build_orbit_jit(mu, p0, v0, dt, count):
    p = np.zeros((2, count + 1), dtype=np.float64)
    v = np.zeros((2, count + 1), dtype=np.float64)

    p[:, 0] = p0
    v[:, 0] = v0

    for i in range(1, count + 1):
        pi, vi = _orbit_rk4_jit(mu, p[:, i - 1], v[:, i - 1], dt)
        p[:, i] = pi
        v[:, i] = vi

    return np.concatenate((p, v), axis=0)


@ctx.ureg.wraps(('m', 'm/s'), ('m^3 / s^2', 'm', 'm/s', 's', None))
def _build_orbit(mu, p0, v0, dt, count):
    arr = _build_orbit_jit(mu.numpy(), p0.numpy(), v0.numpy(), dt.numpy(),
                           count)
    p = tf.constant(arr[0:2])
    v = tf.constant(arr[2:4])
    return p, v


#############################################################################
# Rings
#############################################################################
@dataclass(frozen=True)
class Ring:
    """A generic orbital ring of unspecified geometry
  """
    body: Body
    material: Material
    cross_section: Q_
    segment_count: int

    @property
    def segment_rest_length(self):
        raise NotImplementedError()

    @property
    def segment_mass(self):
        return self.material.density * self.cross_section * self.segment_rest_length

    @property
    def total_mass(self):
        return self.segment_mass * self.segment_count

    @property
    def segment_spring(self):
        return self.material.elasticity * self.cross_section / self.segment_rest_length

    @property
    def initial_position(self):
        raise NotImplementedError()

    @property
    def initial_momentum(self):
        raise NotImplementedError()

    def H(self, ctx_, t_, p_, q_):
        ke = kinetic_energy(ctx_, q_, ctx_.resolve(self.segment_mass))
        ge = gravitational_energy(ctx_, p_, ctx_.resolve(self.segment_mass),
                                  ctx_.constant([0, 0, 0], 'm'),
                                  ctx_.resolve(self.body.mass))
        se = elastic_energy(ctx_,
                            p_,
                            ctx_.resolve(self.segment_rest_length),
                            ctx_.resolve(self.segment_spring),
                            cycle=True)
        return ke + se + ge


@dataclass(frozen=True)
class CircularRing(Ring):
    """An orbital ring in a perfectly circular orbit
  """
    altitude: Q_
    excess_speed: Q_
    stretch_ratio: Q_

    @property
    def radius(self):
        return self.body.radius + self.altitude

    @property
    def orbital_speed(self):
        return ctx.sqrt((G * self.body.mass / self.radius))

    @property
    def speed(self):
        return self.orbital_speed + self.excess_speed

    @property
    def segment_rest_length(self):
        t = ctx.linspace(ctx.constant(0, 'deg'),
                         ctx.constant(360, 'deg') / self.segment_count, 2)
        x = ctx.cos(t) * self.radius
        y = ctx.zeros_like(x)
        z = ctx.sin(t) * self.radius
        p = ctx.stack([x, y, z], axis=0)
        init_length = ctx.norm(p[:, 1] - p[:, 0], axis=0)
        return init_length / self.stretch_ratio

    @property
    def initial_position(self):
        t = ctx.linspace(ctx.constant(0, 'deg'), ctx.constant(360, 'deg'),
                         self.segment_count + 1)[:-1]
        x = ctx.cos(t) * self.radius
        y = ctx.zeros_like(x)
        z = ctx.sin(t) * self.radius
        return ctx.stack([x, y, z], axis=0)

    @property
    def initial_momentum(self):
        t = ctx.linspace(ctx.constant(0, 'deg'), ctx.constant(360, 'deg'),
                         self.segment_count + 1)[:-1]
        vx = -ctx.sin(t) * self.speed
        vy = ctx.zeros_like(vx)
        vz = ctx.cos(t) * self.speed
        v = ctx.stack([vx, vy, vz], axis=0)
        return self.segment_mass * v


@dataclass(frozen=True)
class EllipticSectionRing(Ring):
    """An orbital ring following two eccentric orbits meeting at cusps at the poles

  The change in momentum at the cusp can balance out a load force
  """
    eccentricity: Q_
    periapsis: Q_
    _period: Q_ = None
    _segment_rest_length: Q_ = None
    _initial_position: Q_ = None
    _initial_momentum: Q_ = None
    _load_delta_v: Q_ = None

    def init(self):
        o = Orbit(self.body, self.semi_major_axis, self.eccentricity)
        t0, p0, v0 = o.points(count=self.segment_count // 2,
                              theta0=ctx.constant(-np.pi / 2, 'rad'),
                              theta1=ctx.constant(np.pi / 2, 'rad'))
        period = 2 * (t0[-1] - t0[0])

        # Drop cusp point from one end, or it would end up duplicated when we mirror
        p0 = p0[:, :-1]
        v0 = v0[:, :-1]

        # Nominal load needs to negate the y-component of the velocity at the cusp
        load_delta_v = 2 * v0[1, 0]

        # zero y component of velocity at the cusp, as the anchor would already have
        # applied half of the delta-v
        v00 = ctx.constant([[1], [0]], '') * v0[:, 0:1]
        v0 = ctx.concat([v00, v0[:, 1:]], axis=1)

        p = ctx.concat([p0, -p0], axis=1)
        v = ctx.concat([v0, -v0], axis=1)

        dp = ctx.roll(p, shift=-1, axis=1) - p
        max_length = ctx.max(ctx.norm(dp, axis=0))
        segment_mass = self.material.density * self.cross_section * max_length

        p = ctx.stack([p[0, :], ctx.zeros_like(p[0, :]), p[1, :]], axis=0)
        v = ctx.stack([v[0, :], ctx.zeros_like(v[0, :]), v[1, :]], axis=0)

        return dc.replace(self,
                          segment_count=np.shape(p)[1],
                          _period=period,
                          _segment_rest_length=max_length,
                          _initial_position=p,
                          _initial_momentum=v * segment_mass,
                          _load_delta_v=load_delta_v)

    @property
    def semi_major_axis(self):
        return self.periapsis / (1 - self.eccentricity)

    @property
    def period(self):
        return self._period

    @property
    def segment_rest_length(self):
        return self._segment_rest_length

    @property
    def initial_position(self):
        return self._initial_position

    @property
    def initial_momentum(self):
        return self._initial_momentum

    @property
    def load_delta_v(self):
        return self._load_delta_v

    @property
    def load_force(self):
        """Returns the hypothetical load force that could be supported at each of
    the poles
    """
        v = self._initial_momentum[:, 0] / self.segment_mass
        speed = ctx.norm(v, axis=0)
        linear_density = self.material.density * self.cross_section
        mass_flow_rate = linear_density * speed
        return self.load_delta_v * mass_flow_rate

    @property
    def load_position(self):
        p0 = self._initial_position[:, 0]
        return ctx.stack([p0, -p0], axis=1)

    @property
    def load_altitude(self):
        p0 = self._initial_position[:, 0]
        return ctx.norm(p0, axis=0) - self.body.radius

    @property
    def load_spring(self):
        return self.load_force / (self.load_altitude - self.periapsis)


#############################################################################
# Anchors
#############################################################################
def find_closest_points(ctx, anchors, segments):
    # anchors: (xyz, anchor) -> (xyz, anchor, 1)
    anchors = ctx.expand_dims(anchors, axis=2)

    # segments: (xyz, segment) -> (xyz, 1, segment)
    segments = ctx.expand_dims(segments, axis=1)

    return point_to_segment(ctx, anchors, segments,
                            ctx.roll(segments, shift=-1, axis=2))


@dataclass(frozen=True)
class FixedAnchor:
    position: Q_

    def anchor_energy(self, ctx_: Context, t_: Q_, dist: Q_) -> Q_:
        raise NotImplementedError()

    @property
    def H(self):

        def _H(ctx_, t_, p_, q_):
            anchor_pos = ctx_.resolve(self.position)
            #print(f'anchor_pos: {anchor_pos}')
            closest_coef, closest_point = find_closest_points(
                ctx_, anchor_pos, p_)
            #print(f'closest_point: {closest_point}')
            dist = ctx_.norm(closest_point -
                             ctx_.expand_dims(anchor_pos, axis=2),
                             axis=0)
            #print(f'dist: {dist}')
            min_dist = ctx_.min(dist, axis=1)
            return self.anchor_energy(ctx_, t_, min_dist)

        return _H


@dataclass(frozen=True)
class FixedSpringAnchor(FixedAnchor):
    rest_length: Q_
    spring: Q_

    def anchor_energy(self, ctx_: Context, t_: Q_, dist: Q_) -> Q_:
        rest_length = ctx_.resolve(self.rest_length)
        spring = ctx_.resolve(self.spring)
        stretch = ctx_.maximum(dist - rest_length, 0)
        return 0.5 * ctx_.sum(spring * stretch * stretch)
