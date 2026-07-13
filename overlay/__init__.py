"""Viewer overlay drawing, shared by ``teleop`` and the examples.

A thin overlay layer: it draws markers and pose frames into a passive MuJoCo
viewer's ``user_scn`` and nothing more. It sits above every package (it imports
none of them) and below none, so both a library package like ``teleop`` and a
downstream example can use it without a layering inversion.
"""

from overlay.draw import add_marker, add_frame

__all__ = ["add_marker", "add_frame"]
