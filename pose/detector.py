import cv2 as cv
import mediapipe as mp
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from threading import Thread
from queue import Queue
import matplotlib.animation as animation


class StereoPoseDetector3D:
    def __init__(self, calibration_file='stereo_calibration.dat'):
        # Load calibration data
        with open(calibration_file, 'rb') as f:
            self.calib_data = pickle.load(f)

        self.mtx1 = self.calib_data['mtx1']
        self.dist1 = self.calib_data['dist1']
        self.mtx2 = self.calib_data['mtx2']
        self.dist2 = self.calib_data['dist2']
        self.R = self.calib_data['R']
        self.T = self.calib_data['T'].reshape(3, 1)  # Ensure T is 3x1

        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

        # Create pose detectors
        self.pose_left = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.pose_right = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Calculate essential matrix
        tx = self.T[0, 0]
        ty = self.T[1, 0]
        tz = self.T[2, 0]

        # Create skew-symmetric matrix
        T_cross = np.array([
            [0, -tz, ty],
            [tz, 0, -tx],
            [-ty, tx, 0]
        ])

        # Calculate Essential and Fundamental matrices
        self.E = T_cross @ self.R
        self.F = np.linalg.inv(self.mtx2).T @ self.E @ np.linalg.inv(self.mtx1)

        # Calculate projection matrices
        self.P1 = np.concatenate([np.eye(3), np.zeros((3, 1))], axis=1)
        self.P2 = np.concatenate([self.R, self.T], axis=1)

        # 3D visualization setup
        plt.ion()
        self.fig = plt.figure(figsize=(10, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.points_3d_queue = Queue(maxsize=1)
        self.setup_3d_plot()

    def setup_3d_plot(self):
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('3D Pose')
        self.ax.grid(True)
        self.ax.set_box_aspect([1, 1, 1])
        self.ax.view_init(elev=15, azim=45)
        plt.tight_layout()

    def undistort_points(self, points, camera_matrix, dist_coeffs):
        """Undistort points using camera parameters"""
        points = points.reshape(-1, 1, 2).astype(np.float32)
        undistorted = cv.undistortPoints(points, camera_matrix, dist_coeffs, None, camera_matrix)
        return undistorted.reshape(-1, 2)

    def triangulate_point(self, point1, point2):
        """Triangulate 3D point from stereo observations with improved accuracy

        Args:
            point1: 2D point from left camera (x,y)
            point2: 2D point from right camera (x,y)

        Returns:
            np.ndarray: 3D point coordinates
        """
        # Undistort points
        point1_undist = self.undistort_points(np.array([point1]), self.mtx1, self.dist1)[0]
        point2_undist = self.undistort_points(np.array([point2]), self.mtx2, self.dist2)[0]

        # Convert to homogeneous coordinates
        point1_h = np.append(point1_undist, 1)
        point2_h = np.append(point2_undist, 1)

        # Build the DLT matrix with improved conditioning
        A = np.zeros((4, 4))
        A[0] = point1_h[1] * self.P1[2] - self.P1[1]
        A[1] = self.P1[0] - point1_h[0] * self.P1[2]
        A[2] = point2_h[1] * self.P2[2] - self.P2[1]
        A[3] = self.P2[0] - point2_h[0] * self.P2[2]

        # Solve using SVD with additional numerical stability
        U, S, Vh = np.linalg.svd(A)
        point_3d_h = Vh[-1]
        point_3d = point_3d_h[:3] / point_3d_h[3]

        if np.any(np.isnan(point_3d)) or np.any(np.abs(point_3d) > 1000):
            print(f"Warning: Invalid 3D point detected: {point_3d}")
            return np.zeros(3)

        return point_3d

    def process_frame_parallel(self, frame_left, frame_right):
        """Process frames in parallel using threads"""
        left_queue, right_queue = Queue(), Queue()

        def process_frame(frame, detector, queue):
            result = detector.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
            queue.put(result)

        Thread(target=process_frame, args=(frame_left, self.pose_left, left_queue)).start()
        Thread(target=process_frame, args=(frame_right, self.pose_right, right_queue)).start()

        return left_queue.get(), right_queue.get()

    def update_3d_plot(self, frame_num):
        """Update the 3D matplotlib visualization with enhanced coordinate handling."""
        if not self.points_3d_queue.empty():
            points_3d = self.points_3d_queue.get()

            # Clear previous plot content
            for collection in self.ax.collections[:]:
                collection.remove()
            for line in self.ax.lines[:]:
                line.remove()

            if points_3d.size > 0:
                # Apply initial rotation to correct coordinate system alignment
                initial_rotation = np.array([
                    [0, 0, 1],  # Map camera Z to world X
                    [1, 0, 0],  # Map camera X to world Y
                    [0, 1, 0]  # Map camera Y to world Z
                ])
                points_3d = np.dot(points_3d, initial_rotation)

                # Center the pose
                centroid = np.mean(points_3d, axis=0)
                centered_points = points_3d - centroid

                # Scale based on the pose dimensions while preserving proportions
                scale_factors = np.ptp(centered_points, axis=0)
                overall_scale = np.mean(scale_factors[scale_factors > 0])
                if overall_scale > 0:
                    normalized_points = centered_points / overall_scale

                    # Set fixed scale limits for consistency
                    self.ax.set_xlim([-2, 2])
                    self.ax.set_ylim([-2, 2])
                    self.ax.set_zlim([-2, 2])

                    # Draw landmarks with depth-based coloring
                    depths = normalized_points[:, 2]
                    scatter = self.ax.scatter(normalized_points[:, 0],
                                              normalized_points[:, 1],
                                              normalized_points[:, 2],
                                              c=depths,
                                              cmap='viridis',
                                              s=50,
                                              alpha=0.8)

                    # Draw skeleton connections with anatomically-based grouping
                    for connection in self.mp_pose.POSE_CONNECTIONS:
                        start_idx = connection[0]
                        end_idx = connection[1]

                        if start_idx < len(normalized_points) and end_idx < len(normalized_points):
                            start_point = normalized_points[start_idx]
                            end_point = normalized_points[end_idx]

                            # Color different body parts distinctly
                            if start_idx < 11 and end_idx < 11:  # Upper body
                                color = 'red'
                            elif start_idx > 22 or end_idx > 22:  # Legs
                                color = 'blue'
                            else:  # Arms
                                color = 'green'

                            self.ax.plot([start_point[0], end_point[0]],
                                         [start_point[1], end_point[1]],
                                         [start_point[2], end_point[2]],
                                         color=color,
                                         linewidth=2,
                                         alpha=0.6)

            # Update view settings for optimal visualization
            self.ax.view_init(elev=20, azim=45)
            self.ax.set_box_aspect([1, 1, 1])
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def process_stereo_stream(self, left_video_path, right_video_path):
        """Process synchronized stereo video streams"""
        cap_left = cv.VideoCapture(left_video_path)
        cap_right = cv.VideoCapture(right_video_path)

        if not (cap_left.isOpened() and cap_right.isOpened()):
            raise ValueError("Could not open video streams")

        anim = animation.FuncAnimation(
            self.fig,
            self.update_3d_plot,
            interval=30,
            cache_frame_data=False
        )
        plt.show(block=False)

        frame_times = []
        while True:
            start_time = time.time()

            ret_left, frame_left = cap_left.read()
            ret_right, frame_right = cap_right.read()

            if not (ret_left and ret_right):
                break

            left_results, right_results = self.process_frame_parallel(frame_left, frame_right)

            if left_results.pose_landmarks and right_results.pose_landmarks:
                points_3d = []
                h_left, w_left = frame_left.shape[:2]
                h_right, w_right = frame_right.shape[:2]

                for left_lm, right_lm in zip(left_results.pose_landmarks.landmark,
                                             right_results.pose_landmarks.landmark):
                    # Convert normalized coordinates to pixel coordinates
                    left_point = np.array([
                        left_lm.x * w_left,
                        left_lm.y * h_left
                    ], dtype=np.float32)

                    right_point = np.array([
                        right_lm.x * w_right,
                        right_lm.y * h_right
                    ], dtype=np.float32)

                    point_3d = self.triangulate_point(left_point, right_point)
                    points_3d.append(point_3d)

                if not self.points_3d_queue.full():
                    self.points_3d_queue.put(np.array(points_3d))

            # Draw 2D poses
            if left_results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame_left, left_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            if right_results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame_right, right_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

            # Update visualization and handle FPS display
            plt.pause(0.01)

            frame_time = time.time() - start_time
            frame_times.append(frame_time)
            if len(frame_times) > 30:
                frame_times.pop(0)
            fps = 1.0 / (sum(frame_times) / len(frame_times))

            cv.putText(frame_left, f'FPS: {int(fps)}', (10, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv.putText(frame_right, f'FPS: {int(fps)}', (10, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv.imshow('Left Camera', frame_left)
            cv.imshow('Right Camera', frame_right)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        cap_left.release()
        cap_right.release()
        cv.destroyAllWindows()
        plt.close()

def main():
    left_video = '../camera_calibration/Data/video1.mp4'
    right_video = '../camera_calibration/Data/video2.mp4'

    detector = StereoPoseDetector3D('../camera_calibration/stereo_calibration.dat')
    detector.process_stereo_stream(left_video, right_video)


if __name__ == "__main__":
    main()