import mujoco
import mujoco.viewer
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "models/fr3/scene.xml"

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)

mujoco.mj_resetDataKeyframe(model, data, 0)  # "home" keyframe

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
