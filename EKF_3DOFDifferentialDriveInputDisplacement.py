from GFLocalization import *
from EKF import *
from DR_3DOFDifferentialDrive import *
from DifferentialDriveSimulatedRobot import *
from MapFeature import *
from Pose import Pose3D
class EKF_3DOFDifferentialDriveInputDisplacement(GFLocalization, DR_3DOFDifferentialDrive, EKF):
    """
    This class implements an EKF localization filter for a 3 DOF Diffenteial Drive using an input displacement motion model incorporating
    yaw measurements from the compass sensor.
    It inherits from :class:`GFLocalization.GFLocalization` to implement a localization filter, from the :class:`DR_3DOFDifferentialDrive.DR_3DOFDifferentialDrive` class and, finally, it inherits from
    :class:`EKF.EKF` to use the EKF Gaussian filter implementation for the localization.
    """
    def __init__(self, kSteps, robot, *args):
        """
        Constructor. Creates the list of  :class:`IndexStruct.IndexStruct` instances which is required for the automated plotting of the results.
        Then it defines the inital stawe vecto mean and covariance matrix and initializes the ancestor classes.

        :param kSteps: number of iterations of the localization loop
        :param robot: simulated robot object
        :param args: arguments to be passed to the base class constructor
        """

        self.dt = 0.1  # dt is the sampling time at which we iterate the KF
        x0 = np.zeros((3, 1))  # initial state x0=[x y z psi u v w r]^T
        P0 = np.zeros((3, 3))  # initial covariance

        # this is required for plotting
        index = [IndexStruct("x", 0, None), IndexStruct("y", 1, None), IndexStruct("z", 2, 0), IndexStruct("yaw", 3, 1)]

        self.t_1 = 0
        self.t = 0
        self.Dt = self.t - self.t_1
        super().__init__(index, kSteps, robot, x0, P0, *args)

    def f(self, xk_1, uk):
        """
        Motion model of the EKF.

        :param xk_1: previous mean state vector (Pose3D object)
        :param uk: input vector [dL, dR]
        :return: xk_bar (Pose3D object)
        """

        xk_bar = Pose3D(xk_1).oplus(Pose3D(uk))
        return xk_bar


    def Jfx(self, xk_1):
        """
        Jacobian of the motion model with respect to the state vector. **Method to be overwritten by the child class**.

        :param xk_1: Linearization point. By default the linearization point is the previous state vector taken from a class attribute.
        :return: Jacobian matrix
        """       
        # TODO: To be completed by the student

        return Pose3D(xk_1).J_1oplus(Pose3D(self.uk))

    def Jfw(self, xk_1):
        """
        Jacobian of the motion model with respect to the noise vector. **Method to be overwritten by the child class**.

        :param xk_1: Linearization point. By default the linearization point is the previous state vector taken from a class attribute.
        :return: Jacobian matrix
        """
        # TODO: To be completed by the student

        J = Pose3D.J_2oplus(xk_1)

        return J

    def h(self, xk):  #:hm(self, xk):  # observation model
        """
        The observation model of the EKF is given by:
        :return: expected observation vector
        """
        # TODO: To be completed by the student
        if self.robot.yaw_reading_frequency !=0 and self.robot.k % self.robot.yaw_reading_frequency == 0:
            yaw  = xk[2,0]  # Heading from the state vector
            h= np.array([[yaw]])
        else:
            h=np.array([[]])

        return h  # return the expected observations

    def GetInput(self):
        """

        :return: uk,Qk
        """
        # TODO: To be completed by the student
        uk, Re = self.robot.ReadEncoders()  # get the wheel encoder readings
        left_wheel_pulses, right_wheel_pulses = uk  # unpack the input encoder readings

        # Calculate the distances traveled by each wheel
        dL = (left_wheel_pulses / self.robot.pulse_x_wheelTurns) * (2 * np.pi * self.wheelRadius)  # distance left wheel
        dR = (right_wheel_pulses / self.robot.pulse_x_wheelTurns) * (2 * np.pi * self.wheelRadius)  # distance right wheel
        # Calculate the change in pose
        delta_d = (dR + dL) / 2  # average distance traveled
        d_theta = np.arctan2((dR - dL), self.wheelBase)  # change in orientation
        
        uk = np.array([delta_d,0, d_theta])

        # Calculate the Jacobian of the input
        J_uk = np.array([
            [ (np.pi * self.wheelRadius) / self.robot.pulse_x_wheelTurns, (np.pi * self.wheelRadius) / self.robot.pulse_x_wheelTurns],
            [0, 0],
            [(-2 * np.pi * self.wheelRadius) / self.robot.pulse_x_wheelTurns, (2 * np.pi * self.wheelRadius) / self.robot.pulse_x_wheelTurns]
        ])

        # the covariance matrix Qk for the input noise
        Qk = (J_uk @ Re ) @ J_uk.T  # propagate the encoder noise through the Jacobian
        return uk.reshape(3, 1), Qk

    def GetMeasurements(self):  # override the observation model
        """
        Get the measurements from the robot. Corresponds to the observation model:
        :return: zk, Rk, Hk, Vk: observation vector and covariance of the observation noise. Hk is the Observation matrix and Vk is the noise observation matrix.
        """
        # TODO: To be completed by the student

        zk, Rk = self.robot.ReadCompass()  # get the compass measurement
        if zk.size == 0:
            return zk, Rk, None, None
        Hk = np.array([[0, 0, 1]])  # observation matrix
        Vk = np.array([[1]])  # noise observation matrix

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

    x0 = np.zeros((3, 1))
    P0 = np.zeros((3, 3))

    dd_robot = EKF_3DOFDifferentialDriveInputDisplacement(kSteps,robot)  # initialize robot and KF
    dd_robot.LocalizationLoop(x0, P0, np.array([[0.5, 0.03]]).T)


    exit(0)