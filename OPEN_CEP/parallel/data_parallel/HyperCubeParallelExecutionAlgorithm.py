"""
 Data parallel HyperCube algorithms
"""
from abc import ABC
from OPEN_CEP.parallel.data_parallel.DataParallelExecutionAlgorithm import DataParallelExecutionAlgorithm
import math
from OPEN_CEP.base.Pattern import Pattern
from OPEN_CEP.evaluation.EvaluationMechanismFactory import \
    EvaluationMechanismParameters
from OPEN_CEP.base.DataFormatter import DataFormatter
from OPEN_CEP.base.PatternMatch import *
from OPEN_CEP.stream.Stream import *


class HyperCubeParallelExecutionAlgorithm(DataParallelExecutionAlgorithm, ABC):
    """
    Implements the HyperCube algorithm.
    """
    def __init__(self, units_number, patterns: Pattern or List[Pattern],
                 eval_mechanism_params: EvaluationMechanismParameters, platform, attributes_dict: dict):
        super().__init__(units_number - 1, patterns, eval_mechanism_params, platform)
        self.__attributes_dict = attributes_dict

    def eval(self, events: InputStream, matches: OutputStream, data_formatter: DataFormatter):
        raise NotImplementedError()
