"""
RF pick-outcome classifier used by eva_controller_relative_motion.py during 'release'.

Aligns flex / pressure / force / ToF with an ApproximateTimeSynchronizer, keeps a sliding
window per sensor (same features as training), and predicts a label:
1 = success, 2 or 3 = failure.
"""

from collections import deque
from pathlib import Path

import joblib
import numpy as np
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import WrenchStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.qos import QoSProfile
from std_msgs.msg import Float32MultiArray, Int32, Float32

DEFAULT_FEATURE_ORDER = ["Flex", "Pressure", "Force", "TOF"]


class RFPickClassifier:
    def __init__(self, node, period=0.009):
        self.node = node
        self.log = node.get_logger()
        self.period = period       # min seconds between predictions
        self.last_time = 0.0

        self.model = None
        self.scaler = None
        self.loaded = False
        self.window_size = 5
        self.feature_order = list(DEFAULT_FEATURE_ORDER)
        self._force_raw_buffer = deque(maxlen=21)  # pre-filter buffer
        self._last_built_X = None                  # last aligned window built by _sync_cb
        self._sync = None

        model_path = Path(get_package_share_directory("harvest_control")) / "resource" / "rf_pick_classifier.joblib"
        try:
            data = joblib.load(model_path)

            # model is either a bare estimator or a bundle dict with metadata
            if isinstance(data, dict) and "model" in data:
                self.model = data["model"]
                self.window_size = int(data.get("window_size", 5))
                self.feature_order = list(data.get("feature_order", DEFAULT_FEATURE_ORDER))
                self.scaler = data.get("scaler", data.get("scalar"))  # accept the old misspelled key too
            elif hasattr(data, "predict"):
                self.model = data
            else:
                raise RuntimeError(f"Unrecognized model format: {type(data)}")

            self.loaded = True
            self.log.info(f"RF loaded. Window: {self.window_size}, Features: {self.feature_order}")
        except Exception as e:
            self.log.error(f"Failed to load RF model. RF Control DISABLED. Error: {e}")

        self._hist = {k: deque(maxlen=self.window_size) for k in self.feature_order}

        if self.loaded:
            try:
                qos = QoSProfile(depth=10)
                # message_filters Subscriber requires node and qos_profile kwarg in ROS2 wrapper
                subs = [
                    Subscriber(node, Float32MultiArray, "/flex_sensor_data", qos_profile=qos),
                    Subscriber(node, Float32, "/vacuum_pressure", qos_profile=qos),
                    Subscriber(node, WrenchStamped, "/force_torque_sensor_broadcaster/wrench", qos_profile=qos),
                    Subscriber(node, Int32, "/tof_sensor_data", qos_profile=qos),
                ]
                # tune slop to match sensor skew; 0.05 (50 ms) is a good starting point
                self._sync = ApproximateTimeSynchronizer(subs, queue_size=10, slop=0.05, allow_headerless=True)
                self._sync.registerCallback(self._sync_cb)
                self._subs = subs  # keep references alive
                self.log.info("RF ApproximateTimeSynchronizer registered (slop=0.05s).")
            except Exception as e:
                self.log.warn(f"Could not create RF ApproximateTimeSynchronizer: {e}")
                self._sync = None

    def reset(self):
        """Clear the sliding windows (called when release starts)."""
        for hist in self._hist.values():
            hist.clear()

    def predict(self, now):
        """Return the predicted label, or None if not due yet / not enough data."""
        if not self.loaded or now - self.last_time <= self.period:
            return None
        self.last_time = now
        X = self._last_built_X
        if X is None:
            return None
        assert X.shape[1] == self.model.n_features_in_
        return int(self.model.predict(X)[0])

    def _sync_cb(self, flex_msg, pressure_msg, force_msg, tof_msg):
        """
        Called when flex, pressure, force, tof messages are approximately time-aligned.
        Builds the same sliding-window flattened feature vector used at training time.
        """
        # 1) force magnitude, then 21-sample mean as proxy for training's filter_force(...,21)
        try:
            f = force_msg.wrench.force
            force_mag = float((f.x**2 + f.y**2 + f.z**2)**0.5)
        except Exception as e:
            self.log.debug(f"rf _sync_cb: bad force_msg: {e}")
            return
        self._force_raw_buffer.append(force_mag)
        filtered_force = float(np.mean(self._force_raw_buffer))

        # 2) flex norm (training used a single flex_norm per timestep)
        try:
            flex_norm = float(np.linalg.norm(np.ravel(np.asarray(flex_msg.data, dtype=float)) / 4.0))
        except Exception as e:
            self.log.debug(f"rf _sync_cb: cannot parse flex_msg: {e}")
            return

        # 3) pressure and tof scalars
        pressure = float(pressure_msg.data)
        tof = float(tof_msg.data)

        # 4) append to sliding windows (order must match training bundle)
        for k in self.feature_order:
            if k not in self._hist:
                self._hist[k] = deque(maxlen=self.window_size)
        self._hist["Flex"].append(flex_norm)
        self._hist["Pressure"].append(pressure)
        self._hist["Force"].append(filtered_force)
        self._hist["TOF"].append(tof)

        # 5) once full, flatten oldest->newest per sensor
        W = int(self.window_size)
        if not all(len(self._hist[k]) >= W for k in self.feature_order):
            self._last_built_X = None  # not enough history yet
            return
        feat_list = []
        for k in self.feature_order:
            feat_list.extend(list(self._hist[k])[-W:])
        X = np.array(feat_list, dtype=float).reshape(1, -1)

        # optional scaler
        if self.scaler is not None and hasattr(self.scaler, "transform"):
            try:
                X = self.scaler.transform(X)
            except Exception as e:
                self.log.warn(f"rf _sync_cb: scaler.transform failed: {e}")
                X = None

        # sanity-check vs model
        if X is not None and hasattr(self.model, "n_features_in_") and X.shape[1] != self.model.n_features_in_:
            self.log.warn(f"rf _sync_cb: built {X.shape[1]} features but model expects {self.model.n_features_in_}")
            X = None

        self._last_built_X = X
