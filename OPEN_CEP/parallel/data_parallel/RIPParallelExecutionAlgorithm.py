from abc import ABC
from OPEN_CEP.parallel.data_parallel.DataParallelExecutionAlgorithm import DataParallelExecutionAlgorithm
import math
from OPEN_CEP.base.Pattern import Pattern
from OPEN_CEP.evaluation.EvaluationMechanismFactory import \
    EvaluationMechanismParameters, EvaluationMechanismFactory
from OPEN_CEP.base.DataFormatter import DataFormatter
from OPEN_CEP.base.PatternMatch import *
from OPEN_CEP.stream.Stream import *


class RIPParallelExecutionAlgorithm(DataParallelExecutionAlgorithm, ABC):
    """
    Implements the RIP algorithm.
    """
    def __init__(self, units_number, patterns: Pattern or List[Pattern],
                 eval_mechanism_params: EvaluationMechanismParameters,
                 platform, multiple):
        super().__init__(units_number - 1, patterns, eval_mechanism_params, platform)
        self.__eval_mechanism_params = eval_mechanism_params

    def eval(self, events: InputStream, matches: OutputStream, data_formatter: DataFormatter):
        raise NotImplementedError()
