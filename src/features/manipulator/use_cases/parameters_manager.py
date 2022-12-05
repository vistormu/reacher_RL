import yaml
import numpy as np
from copy import deepcopy

from ..entities import DHParameters, IKParameters
from ...core.entities import OrientedPoint

ASSETS_PATH: str = 'src/features/manipulator/assets/'


class ParametersManager:
    @staticmethod
    def get_dh_parameters(robot: str) -> DHParameters:
        path: str = f'{ASSETS_PATH}{robot}/dh.yaml'

        with open(path, "r") as file:
            dh_table: dict = dict()
            try:
                dh_table = yaml.safe_load(file)
            except yaml.YAMLError as exception:
                print(exception)

        a: list[float] = dh_table[robot]['a']
        d: list[float] = dh_table[robot]['d']
        alpha: list[float] = dh_table[robot]['alpha']

        return DHParameters(a, d, alpha)

    @staticmethod
    def extend_dh_parameters(dh_parameters: DHParameters, base: OrientedPoint) -> DHParameters:
        # Create object
        extended_dh_parameters: DHParameters = deepcopy(dh_parameters)

        # Calculate base values
        d_0: float = base.position.z
        alpha_0: float = np.arctan2(base.axes.y.w, base.axes.z.w)
        theta_0 = np.arctan2(base.axes.x.v, base.axes.x.u)
        a_0: float = base.position.x
        if theta_0:  # TMP
            a_0: float = base.position.x/np.cos(theta_0)
            a_0_bis: float = base.position.y/np.sin(theta_0)

        # Insert values
        extended_dh_parameters.a.insert(0, a_0)
        extended_dh_parameters.d.insert(0, d_0)
        extended_dh_parameters.alpha.insert(0, alpha_0)

        return extended_dh_parameters

    @staticmethod
    def get_ik_parameters(dh_parameters: DHParameters) -> IKParameters:
        phi: list[int] = [0]*len(dh_parameters.a)
        mu: list[int] = [0]*len(dh_parameters.a)
        for i, (a, d, alpha) in enumerate(zip(dh_parameters.a, dh_parameters.d, dh_parameters.alpha)):
            phi[i] = 0 if d == 0 else 1
            mu[i] = np.sign(-a-d*np.sin(alpha))

        return IKParameters(phi, mu, 0)
