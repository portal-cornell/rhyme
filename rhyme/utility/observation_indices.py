import numpy as np

ACTION_INDICES = {
    "bottom burner": np.array([11, 12]),
    "top burner": np.array([15, 16]),
    "light switch": np.array([17, 18]),
    "slide cabinet": np.array([19]),
    "full slide cabinet": np.array([19]),
    "hinge cabinet": np.array([20, 21]),
    "full hinge cabinet": np.array([20, 21]),
    "microwave": np.array([22]),
    "full microwave": np.array([22]),
    "kettle": np.array([23, 24, 25, 26, 27, 28, 29]),
    "lift kettle": np.array([23, 24, 25, 26, 27, 28, 29]),
}

ACTION_GOALS = {
    "bottom burner": [np.array([-0.88, 0])],
    "top burner": [np.array([-0.92, 0])],
    "light switch": [np.array([-0.69, -0.05])],
    "slide cabinet": [np.array([0.37])],
    "full slide cabinet": [np.array([0.5])],
    "hinge cabinet": [np.array([0.0, 1.45])],
    "full hinge cabinet": [np.array([0.0, 1.7])],
    "microwave": [np.array([-0.75])],
    "full microwave": [np.array([-1.5])],
    "kettle": [np.array([-0.23, 0.75, 1.62, 0.99, 0.0, 0.0, -0.06])],
    "lift kettle": [
        np.array([-0.26, 0.3, 1.9, 0.99, 0.0, 0.0, -0.06]),
        np.array([-0.26, 0.65, 1.8, 0.99, 0.0, 0.0, -0.06]),
        np.array([-0.23, 0.75, 1.62, 0.99, 0.0, 0.0, -0.06]),
    ],
}

KETTLE_INIT = np.array([-0.269, 0.35, 1.62, 0.99, 0.0, 0.0, 0.0])