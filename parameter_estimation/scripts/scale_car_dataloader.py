import os
import numpy as np

from torch.utils.data import Dataset
from load_from_json import (
    load_car_from_json,
    NN,
    STATE,
    DEFAULT_ANGULAR_SMOOTHING,
    DEFAULT_LINEAR_SMOOTHING,
    DEFAULT_ROLL_SMOOTHING,
)


class ScaleCarDynamicsData(Dataset):
    def __init__(
        self,
        file_name,
        past_timesteps,
        future_timesteps,
        angular_smoothing_factor=DEFAULT_ANGULAR_SMOOTHING,
        linear_smoothing_factor=DEFAULT_LINEAR_SMOOTHING,
    ):
        """
        Args:
            file_name (string): Path to the json or npz file with vehicle data.
                                If the filename ends in .npz, it will be loaded directly instead of interpolating from a json.
            angular_smoothing_factor (double): Smoothing factor for angular rate spline.
            linear_smoothing_factor (double): Smoothing factor for linear velocity spline.
        """
        file_base, file_extension = os.path.splitext(file_name)
        if file_extension == ".json":
            # Extract car data from the json file, including splining
            self.controls, self.states, self.time, self.nn_input, self.nn_target = load_car_from_json(
                file_name, angular_smoothing_factor, linear_smoothing_factor
            )
            # Cut states down to only those used in the car.  From:
            # pos_x, pos_y, theta, roll, vel_x, vel_y, measured_theta_rate, acc_x, acc_y, theta_accel_rate_spline
            self.states = self.states[0:10, :].astype("float32")
            self.controls = self.controls.astype("float32")
            self.nn_input = self.nn_input.astype("float32")
            self.nn_target = self.nn_target.astype("float32")
            # Save the extracted data.
            print(f'Saving {file_base + ".npz"}')
            np.savez(
                file_base + ".npz",
                states=self.states,
                controls=self.controls,
                nn_input=self.nn_input,
                nn_target=self.nn_target,
                time=self.time,
            )
        elif file_extension == ".npz":
            # Load cached data from an npz file
            print(f"Loading cache from {file_name}")
            cached_data = np.load(file_name, allow_pickle=True)
            self.states = cached_data["states"]
            self.controls = cached_data["controls"]
            self.nn_input = cached_data["nn_input"]
            self.nn_target = cached_data["nn_target"]
            self.time = cached_data["time"]
        else:
            print(f"Unsupported file type {file_extension} for file {file_name}.")
            exit()

        self.past_timesteps = past_timesteps
        self.future_timesteps = future_timesteps

    def __len__(self):
        # We cannot get closer than [self.future_timesteps, self.past_timesteps] to the end of the data
        return len(self.time) - self.future_timesteps - self.past_timesteps

    def __getitem__(self, idx):
        end_idx = idx + self.past_timesteps + self.future_timesteps
        states = self.states[:, idx:end_idx]
        nn_input = self.nn_input[:, idx:end_idx]
        nn_target = self.nn_target[:, idx:end_idx]

        return {"nn_input": nn_input.transpose(), "nn_target": nn_target.transpose(), "states": states.transpose()}


if __name__ == "__main__":
    # This file is designed as a library, and should not be run directly except for developer testing.
    import copy
    import time
    import matplotlib.pyplot as plt

    # dynamics_file = "racecar_3_2022-03-22-10-34-20.json"
    dynamics_file = "racecar_3_2022-03-22-11-11-48.json"

    d = ScaleCarDynamicsData("parameter_estimation/data/" + dynamics_file, 0, 1000)
    for i in range(5):
        data = d[10000]
        plt.figure()
        plt.plot(data["nn_input"][:, i])
    plt.show()
    pass