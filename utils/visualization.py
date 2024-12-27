import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


class PoseVisualizer:
    def __init__(self):
        self.fig = plt.figure(figsize=(10, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Define boxing pose connections
        self.connections = [
            # Torso
            (11, 12), (12, 24), (24, 23), (23, 11),
            # Right arm
            (12, 14), (14, 16),
            # Left arm
            (11, 13), (13, 15),
            # Right leg
            (24, 26), (26, 28),
            # Left leg
            (23, 25), (25, 27)
        ]

    def plot_3d_pose(self, points_3d: np.ndarray):
        self.ax.clear()

        # Plot points
        self.ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='b', marker='o')

        # Draw connections
        for start_idx, end_idx in self.connections:
            self.ax.plot([points_3d[start_idx, 0], points_3d[end_idx, 0]],
                         [points_3d[start_idx, 1], points_3d[end_idx, 1]],
                         [points_3d[start_idx, 2], points_3d[end_idx, 2]], 'r-')

        # Set labels and title
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('3D Boxing Pose')

        # Set limits with some padding
        padding = 0.2
        max_range = np.array([points_3d[:, 0].max() - points_3d[:, 0].min(),
                              points_3d[:, 1].max() - points_3d[:, 1].min(),
                              points_3d[:, 2].max() - points_3d[:, 2].min()]).max() / 2.0

        mid_x = (points_3d[:, 0].max() + points_3d[:, 0].min()) * 0.5
        mid_y = (points_3d[:, 1].max() + points_3d[:, 1].min()) * 0.5
        mid_z = (points_3d[:, 2].max() + points_3d[:, 2].min()) * 0.5

        self.ax.set_xlim(mid_x - max_range * (1 + padding), mid_x + max_range * (1 + padding))
        self.ax.set_ylim(mid_y - max_range * (1 + padding), mid_y + max_range * (1 + padding))
        self.ax.set_zlim(mid_z - max_range * (1 + padding), mid_z + max_range * (1 + padding))

        plt.draw()
        plt.pause(0.001)