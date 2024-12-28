from GFLocalization import *
from EKF import *
from DR_3DOFDifferentialDrive import *
from DifferentialDriveSimulatedRobot import *

class EKF_3DOFDifferentialDriveCtVelocity(GFLocalization, DR_3DOFDifferentialDrive, EKF):

    def __init__(self, kSteps, robot, *args):

        self.x0 = np.zeros((6, 1))  # initial state x0=[x y z psi u v w r]^T
        self.P0 = np.zeros((6, 6))  # initial covariance

        # this is required for plotting
        self.index = [IndexStruct("x", 0, None), IndexStruct("y", 1, None), IndexStruct("yaw", 2, 1),
                 IndexStruct("u", 3, 2), IndexStruct("v", 4, 3), IndexStruct("yaw_dot", 5, None)]

        # TODO: To be completed by the student
        super().__init__(self.index, kSteps, robot, self.x0, self.P0, *args)


    def f(self, xk_1, uk):
        # TODO: To be completed by the student
        # Extract state components
        pose = xk_1[:3]  # [x, y, yaw]
        velocities = xk_1[3:]  # [u, v, yaw_dot]
        
        dt = self.dt
        # Use pose composition
        new_pose = Pose3D(pose).oplus(Pose3D(velocities * dt))
        
        # Combine with unchanged velocities
        xk_bar = np.vstack([new_pose, velocities])
        
        return xk_bar

    def Jfx(self, xk_1):
        """
        Computes the Jacobian of the motion model with respect to the state vector `xk_1`.

        :param xk_1: The state vector at the previous time step
        :return: Jacobian matrix of the motion model
        """
        # Extract pose and velocity from the state vector
        pose = xk_1[:3]  
        velocities = xk_1[3:] 
        dt = self.dt  # Time step

        # Compute Jacobians for the pose composition
        J1 = Pose3D(pose).J_1oplus(Pose3D(velocities * dt))  # Jacobian wrt pose
        J2 = Pose3D(pose).J_2oplus() * dt  # Jacobian wrt velocity scaled by dt

        # Initialize a 6x6 Jacobian matrix
        J = np.zeros((6, 6))

        # Fill the top-left block with J1 (wrt pose)
        J[:3, :3] = J1

        # Fill the top-right block with J2 (wrt velocity)
        J[:3, 3:] = J2

        # Fill the bottom-right block with the identity matrix (velocity remains unchanged)
        J[3:, 3:] = np.eye(3)

        return J


    def Jfw(self, xk_1):
        # TODO: To be completed by the student
        pose = xk_1[:3]
        dt = self.dt
        
        # Get J2 from pose composition
        J2 = Pose3D(pose).J_2oplus()  # Jacobian wrt velocity scaled by dt
        
        # Scale by dt for velocity integration
        J = np.zeros((6, 3))
        J[:3, :] = (J2 * (dt**2))/2
        J[3:, :] = np.eye(3)*dt
        return J
    
    def h(self, xk):  #:hm(self, xk):
        # TODO: To be completed by the student
        Hk = np.array([]).reshape((0, 6))
        if self.robot.k % self.robot.yaw_reading_frequency == 0:
            Hk = np.vstack([Hk,
                           [0, 0, 1, 0, 0, 0],])
        if self.robot.k % self.robot.encoder_reading_frequency == 0:
            Hk = np.vstack([Hk,
                           [0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 1, 0],])
        h = Hk @ xk
        return h

    def GetInput(self):
        """
        Computes the control input vector `uk` and the process noise covariance `Qk`.

        :return: uk (empty, as there is no direct input), Qk (uncertainty in x, y, theta)
        """
        # No direct input in constant velocity model
        uk = np.array([])
        Qk = self.robot.Qsk

        return uk, Qk

    
    
    def GetMeasurements(self):
        """
        Retrieve sensor measurements and construct the observation vector (zk),
        noise covariance matrix (Rk), observation Jacobian (Hk), and noise Jacobian (Vk).

        :return: zk, Rk, Hk, Vk
        """
        zsk, Re = self.robot.ReadEncoders()
        yaw, Ryaw = self.robot.ReadCompass()

        # Initialize default values
        zk = np.array([]).reshape((0, 1))
        Rk = np.array([]).reshape((0, 0))
        Hk = np.array([]).reshape((0, 6))
        Vk = np.array([]).reshape((0, 0))   

        # Handle compass reading
        if yaw.size > 0:
            yaw = np.array([[yaw]]) if yaw.ndim == 1 else yaw.reshape((1, 1))
            zk = yaw
            Rk = Ryaw
            Hk = np.zeros((1, 6))
            Hk[0, 2] = 1.0  # yaw measurement
            Vk = np.eye(1)

        # Handle encoder readings
        if zsk.size > 0:
            left_wheel_pulses, right_wheel_pulses = zsk
            wl = (left_wheel_pulses / self.robot.pulse_x_wheelTurns) * (2 * np.pi / self.dt)
            wr = (right_wheel_pulses / self.robot.pulse_x_wheelTurns) * (2 * np.pi / self.dt)
            u = self.wheelRadius/2 * (wl + wr)
            yaw_dot = (self.wheelRadius * (wr - wl)) / self.wheelBase

            encoder_z = np.array([[u], [yaw_dot]])
            J_zsk = np.array([[self.wheelRadius/(2*self.dt*self.robot.pulse_x_wheelTurns)*(2*np.pi), self.wheelRadius/(2*self.dt*self.robot.pulse_x_wheelTurns)*(2*np.pi)],
                             [-self.wheelRadius/(self.wheelBase*self.dt*self.robot.pulse_x_wheelTurns)*(2*np.pi), self.wheelRadius/(self.wheelBase*self.dt*self.robot.pulse_x_wheelTurns)*(2*np.pi)]])
            encoder_R = (J_zsk @ Re) @ J_zsk.T

            # Combine with existing measurements if any
            zk = np.vstack([zk, encoder_z]) if zk.size > 0 else encoder_z
            Rk = scipy.linalg.block_diag(Rk, encoder_R) if Rk.size > 0 else encoder_R
            encoder_H = np.zeros((2, 6))
            encoder_H[0, 3] = 1.0  # u
            encoder_H[1, 4] = 1.0  # v
            Hk = np.vstack([Hk, encoder_H]) if Hk.size > 0 else encoder_H
            Vk = np.eye(zk.shape[0])

        return zk, Rk, Hk, Vk

                


if __name__ == '__main__':

    M = [CartesianFeature(np.array([[-40, 5]]).T),
           CartesianFeature(np.array([[-5, 40]]).T),
           CartesianFeature(np.array([[-5, 25]]).T),
           CartesianFeature(np.array([[-3, 50]]).T),
           CartesianFeature(np.array([[-20, 3]]).T),
           CartesianFeature(np.array([[40,-40]]).T)]  # feature map. Position of 2 point features in the world frame.

    xs0 = np.zeros((6,1))  # initial simulated robot pose

    robot = DifferentialDriveSimulatedRobot(xs0, M)  # instantiate the simulated robot object
    kSteps = 5000

    xs0 = np.zeros((6, 1))  # initial simulated robot pose
    index = [IndexStruct("x", 0, None), IndexStruct("y", 1, None), IndexStruct("yaw", 2, 1)]

    x0 = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).T
    P0 = np.diag(np.array([0.0, 0.0, 0.0, 0.5 ** 2, 0 ** 2, 0.05 ** 2]))

    dd_robot = EKF_3DOFDifferentialDriveCtVelocity(kSteps, robot)  # initialize robot and KF
    dd_robot.LocalizationLoop(x0, P0, np.array([[0.5, 0.03]]).T)  # run localization loop

    exit(0)