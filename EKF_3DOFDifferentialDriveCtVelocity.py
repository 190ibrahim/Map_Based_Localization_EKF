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
        super().__init__(index, kSteps, robot, self.x0, self.P0, *args)


    def f(self, xk_1, uk):
        # TODO: To be completed by the student
        # Extract state components
        pose = xk_1[:3]  # [x, y, yaw]
        velocities = xk_1[3:]  # [u, v, r]
        
        # Create displacement pose from velocities
        dt = self.dt
        displacement = np.array([[velocities[0, 0] * dt],  # x displacement
                                [velocities[1, 0] * dt],  # y displacement
                                [velocities[2, 0] * dt]])  # angle displacement
        
        # Use pose composition
        new_pose = Pose3D(pose).oplus(Pose3D(displacement))
        
        # Combine with unchanged velocities
        xk_bar = np.vstack([new_pose, velocities])
        
        return xk_bar

    def Jfx(self, xk_1):
        # TODO: To be completed by the student
        pose = xk_1[:3]
        velocities = xk_1[3:]
        dt = self.dt
        
        # Create displacement pose
        displacement = np.array([[velocities[0, 0] * dt],
                                [velocities[1, 0] * dt],
                                [velocities[2, 0] * dt]])
            
        # Get J1 from pose composition
        J1 = Pose3D(pose).J_1oplus(Pose3D(displacement))
        
        # Create full Jacobian
        J = np.zeros((6, 6))
        J[:3, :3] = J1
        J[3:, 3:] = np.eye(3)  # velocities remain unchanged
        
        return J

    def Jfw(self, xk_1):
        # TODO: To be completed by the student
        pose = xk_1[:3]
        dt = self.dt
        
        # Get J2 from pose composition
        J2 = Pose3D(pose).J_2oplus()
        
        # Scale by dt for velocity integration
        J = np.zeros((6, 6))
        J[:3, :3] = J2 * dt
        J[3:, 3:] = np.eye(3)
        
        return J
    def h(self, xk):  #:hm(self, xk):
        # TODO: To be completed by the student
        # Measurements: compass, u, v
        h = np.array([[xk[2,0]],   # yaw from compass
                    [xk[3,0]],   # u velocity
                    [xk[4,0]]])  # v velocity
        return h

    def GetInput(self):
        """

        :return: uk,Qk:
        """
        # TODO: To be completed by the student
        # No direct input in constant velocity model
        uk = np.zeros((6,1))
        # Process noise covariance
        Qk = np.diag([0.1**2, 0.1**2, (np.pi/180)**2,  # pose noise
                      0.5**2, 0.1**2, (np.pi/60)**2])   # velocity noise
        return uk, Qk
    
    
    def GetMeasurements(self):  # override the observation model
        """
        Retrieve sensor measurements and construct the observation vector (zk),
        noise covariance matrix (Rk), observation Jacobian (Hk), and noise Jacobian (Vk).
        
        :return: zk, Rk, Hk, Vk
        """
        zsk, Re = self.robot.ReadEncoders()    # e.g., shape (2,1) if it returns [u_enc, v_enc]
        yaw, Ryaw = self.robot.ReadCompass()   # shape (1,1)

        # Combine them into a single measurement vector
        if zsk.size > 0 and yaw.size > 0:

            # Make sure yaw is (1,1):
            if not isinstance(yaw, np.ndarray):
                yaw = np.array([[yaw]])           # shape (1,1)
            elif yaw.ndim == 1:
                yaw = yaw.reshape((1,1))

            # Make sure zsk is (2,1) if it holds two scalar readings:
            if zsk.ndim == 1:
                zsk = zsk.reshape((2,1))          # shape (2,1)
            
            # Now stack them
            zk = np.vstack([yaw, zsk])            # shape (3,1)

            # Build Rk as a block diagonal of Ryaw (1×1) and Re (2×2)
            Rk = scipy.linalg.block_diag(Ryaw, Re)  # shape (3,3)
        else:
            # If no measurements are available this time, return empties or None
            zk = np.array([])
            Rk = np.array([])

        # Hk: This is how your 6D state maps to [theta, u, v].
        # => Hk shape: (3×6)
        Hk = np.zeros((3, 6))

        #   row 0 measures theta => depends on state[2]
        Hk[0, 2] = 1.0

        #   row 1 measures u => depends on state[3]
        Hk[1, 3] = 1.0

        #   row 2 measures v => depends on state[4]
        Hk[2, 4] = 1.0
        print(Hk)
        # Vk: If noise is uncorrelated and each measurement has 1 noise dimension,
        #     simply an identity of size 3×3.
        if zk.size > 0:
            Vk = np.eye(3)  # shape (3×3)
        else:
            Vk = np.array([])

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