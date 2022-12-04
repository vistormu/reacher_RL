import numpy as np

from ...repository import ManipulatorRepository

from ...entities import ManipulatorData
from ....core.entities import OrientedPoint



class MockManipulatorRepository(ManipulatorRepository):
    def __init__(self) -> None:
        self.manipulator_data: ManipulatorData = None   # type: ignore
    
    def send_manipulator_data(self, manipulator_data: ManipulatorData) -> None:
        # Copy values
        a: list[float] = manipulator_data.dh_parameters.a.copy()
        d: list[float] = manipulator_data.dh_parameters.d.copy()
        alpha: list[float] = manipulator_data.dh_parameters.alpha.copy()
        theta: list[float] = manipulator_data.angles.copy()
        
        systems: list[OrientedPoint] = []
        
        # Get absolute systems
        system: np.ndarray = np.eye(4)
        for i in range(len(systems)):
            step_matrix: np.ndarray = np.array([[np.cos(theta[i]), -np.cos(alpha[i])*np.sin(theta[i]), np.sin(alpha[i])*np.sin(theta[i]), a[i]*np.cos(theta[i])],
                                                [np.sin(theta[i]), np.cos(alpha[i])*np.cos(theta[i]), -np.sin(alpha[i])*np.cos(theta[i]), a[i]*np.sin(theta[i])],
                                                [0, np.sin(alpha[i]), np.cos(alpha[i]), d[i]],
                                                [0, 0, 0, 1]])
            system = system @ step_matrix
            
            systems.append(OrientedPoint.from_htm(system))
            
        self.manipulator_data.systems = systems
        
    
    def get_manipulator_data(self) -> ManipulatorData:
        return self.manipulator_data
    