from Localization import *
import numpy as np
from DifferentialDriveSimulatedRobot import *
from MapFeature import *
from MapFeature import *
class DR_3DOFDifferentialDrive(Localization):
    """
    Dead Reckoning Localization for a Differential Drive Mobile Robot.
    """
    def __init__(self, index, kSteps, robot, x0, *args):
        """
        Constructor of the :class:`prlab.DR_3DOFDifferentialDrive` class.

        :param args: Rest of arguments to be passed to the parent constructor
        """

        super().__init__(index, kSteps, robot, x0, *args)  # call parent constructor

        self.dt = 0.1  # dt is the sampling time at which we iterate the DR
        self.t_1 = 0.0  # t_1 is the previous time at which we iterated the DR
        self.wheelRadius = 0.1  # wheel radius
        self.wheelBase = 0.5  # wheel base
        self.robot.pulse_x_wheelTurns = 4096  # number of pulses per wheel turn

    def Localize(self, xk_1, uk):  # motion model
        """
        Motion model for the 3DOF (:math:`x_k=[x_{k}~y_{k}~\psi_{k}]^T`) Differential Drive Mobile robot using as input the readings of the wheel encoders (:math:`u_k=[n_L~n_R]^T`).

        :parameter xk_1: previous robot pose estimate (:math:`x_{k-1}=[x_{k-1}~y_{k-1}~\psi_{k-1}]^T`)
        :parameter uk: input vector (:math:`u_k=[u_{k}~v_{k}~w_{k}~r_{k}]^T`)
        :return xk: current robot pose estimate (:math:`x_k=[x_{k}~y_{k}~\psi_{k}]^T`)
        """

        # TODO: to be completed by the student


        # TODO: IDK Store previous state and input for Logging purposes
        self.etak_1 = xk_1  # store previous state
        self.uk = uk  # store input


        dL, dR = uk
        # Calculate the change in pose
        delta_d = (dR + dL) / 2  # average distance traveled
        delta_theta = (dR - dL) / self.wheelBase  # change in orientation
        
        # Update the pose
        x_k = xk_1[0] + delta_d * np.cos(xk_1[2] + delta_theta / 2)  
        y_k = xk_1[1] + delta_d * np.sin(xk_1[2] + delta_theta / 2)  
        psi_k = xk_1[2] + delta_theta  # new orientation
        xk = np.array([x_k, y_k, psi_k])

        return xk

    def GetInput(self):
        """
        Get the input for the motion model. In this case, the input is the readings from both wheel encoders.

        :return: uk:  input vector (:math:`u_k=[n_L~n_R]^T`)
        """

        # TODO: to be completed by the student
        uk, _ = self.robot.ReadEncoders()  # get the wheel encoder readings
        left_wheel_pulses, right_wheel_pulses = uk  # unpack the input encoder readings

        # Calculate the distances traveled by each wheel
        dL = (left_wheel_pulses / self.robot.pulse_x_wheelTurns) * (2 * np.pi * self.wheelRadius)  # distance left wheel
        dR = (right_wheel_pulses / self.robot.pulse_x_wheelTurns) * (2 * np.pi * self.wheelRadius)  # distance right wheel
        uk = np.array([dL, dR])

        return uk
        

if __name__ == "__main__":

    # feature map. Position of 2 point features in the world frame.
    M = [CartesianFeature(np.array([[-40, 5]]).T),
           CartesianFeature(np.array([[-5, 40]]).T),
           CartesianFeature(np.array([[-5, 25]]).T),
           CartesianFeature(np.array([[-3, 50]]).T),
           CartesianFeature(np.array([[-20, 3]]).T),
           CartesianFeature(np.array([[40,-40]]).T)]  # feature map. Position of 2 point features in the world frame.

    xs0=np.zeros((6,1))   # initial simulated robot pose
    robot = DifferentialDriveSimulatedRobot(xs0, M) # instantiate the simulated robot object

    kSteps = 5000 # number of simulation steps
    xsk_1 = xs0 = np.zeros((6, 1))  # initial simulated robot pose
    index = [IndexStruct("x", 0, None), IndexStruct("y", 1, None), IndexStruct("yaw", 2, 1)] # index of the state vector used for plotting

    x0=Pose3D(np.zeros((3,1)))
    dr_robot=DR_3DOFDifferentialDrive(index,kSteps,robot,x0)
    dr_robot.LocalizationLoop(x0, np.array([[0.5, 0.03]]).T)

    exit(0)