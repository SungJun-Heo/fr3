"""Reusable object library.

Each helper adds an object to an ``mjSpec`` (a model being edited) and returns
the created body. Objects are free-floating by default (they get a freejoint),
so a manipulator can pick them up. Pass ``free=False`` for a static fixture.

Tasks describe objects as plain dicts (see ``tasks.py``); ``add_object`` is the
single entry point that dispatches on the ``kind`` field.
"""

import mujoco

_PRIMITIVES = {
    "box": mujoco.mjtGeom.mjGEOM_BOX,
    "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
    "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
    "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE,
    "ellipsoid": mujoco.mjtGeom.mjGEOM_ELLIPSOID,
}


def _size3(size):
    """MuJoCo geom size is always a 3-vector; pad shorter specs with zeros."""
    s = list(size) if hasattr(size, "__iter__") else [size]
    return (s + [0.0, 0.0, 0.0])[:3]


def add_primitive(spec, kind, name, pos, size=(0.02, 0.02, 0.02),
                  rgba=(0.6, 0.6, 0.6, 1), quat=(1, 0, 0, 0), free=True):
    """Add a single-geom primitive body (box/sphere/cylinder/capsule/ellipsoid)."""
    body = spec.worldbody.add_body(name=name, pos=list(pos), quat=list(quat))
    if free:
        body.add_freejoint()
    body.add_geom(name=name, type=_PRIMITIVES[kind], size=_size3(size), rgba=list(rgba))
    return body


def add_bin(spec, name, pos, inner=(0.08, 0.08), height=0.05, wall=0.005,
            rgba=(0.5, 0.4, 0.3, 1), quat=(1, 0, 0, 0), free=False):
    """A composite object: an open-top box (floor + 4 walls), e.g. for bin picking."""
    ix, iy = inner
    t, h = wall, height
    body = spec.worldbody.add_body(name=name, pos=list(pos), quat=list(quat))
    if free:
        body.add_freejoint()

    def wall_geom(gname, size, gpos):
        body.add_geom(name=gname, type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=list(size), pos=list(gpos), rgba=list(rgba))

    wall_geom(f"{name}_floor", (ix + t, iy + t, t), (0, 0, t))
    wall_geom(f"{name}_wx+", (t, iy + t, h), (ix + t, 0, h))
    wall_geom(f"{name}_wx-", (t, iy + t, h), (-(ix + t), 0, h))
    wall_geom(f"{name}_wy+", (ix + t, t, h), (0, iy + t, h))
    wall_geom(f"{name}_wy-", (ix + t, t, h), (0, -(iy + t), h))
    return body


# Registry: object "kind" -> builder. Add new object types here.
_BUILDERS = {k: (lambda spec, kind=k, **kw: add_primitive(spec, kind, **kw))
             for k in _PRIMITIVES}
_BUILDERS["bin"] = lambda spec, **kw: add_bin(spec, **kw)


def add_object(spec, kind, **params):
    """Dispatch a task's object spec (``dict(kind=..., name=..., pos=..., ...)``)."""
    if kind not in _BUILDERS:
        raise ValueError(f"unknown object kind '{kind}'. "
                         f"available: {', '.join(sorted(_BUILDERS))}")
    return _BUILDERS[kind](spec, **params)
