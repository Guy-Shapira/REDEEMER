from abc import ABC
from OPEN_CEP.parallel.data_parallel.DataParallelExecutionAlgorithm import DataParallelExecutionAlgorithm

from OPEN_CEP.base.Pattern import Pattern
from OPEN_CEP.evaluation.EvaluationMechanismFactory import \
    EvaluationMechanismParameters
from OPEN_CEP.base.DataFormatter import DataFormatter
from OPEN_CEP.base.PatternMatch import *
from OPEN_CEP.stream.Stream import *


class GroupByKeyParallelExecutionAlgorithm(DataParallelExecutionAlgorithm, ABC):
    """
    Implements the key-based partitioning algorithm.
    """
    def __init__(self, units_number, patterns: Pattern or List[Pattern],
                 eval_mechanism_params: EvaluationMechanismParameters,
                 platform, key: str):
        super().__init__(units_number, patterns, eval_mechanism_params, platform)
        self.__key = key

    def eval(self, events: InputStream, matches: OutputStream,
             data_formatter: DataFormatter):
        raise NotImplementedError()
