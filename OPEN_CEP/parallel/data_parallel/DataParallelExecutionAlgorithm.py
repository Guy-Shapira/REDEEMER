from abc import ABC
from OPEN_CEP.base.Pattern import Pattern
from OPEN_CEP.evaluation.EvaluationMechanismFactory import \
    EvaluationMechanismParameters, EvaluationMechanismFactory
from OPEN_CEP.base.DataFormatter import DataFormatter
from OPEN_CEP.base.PatternMatch import *
from OPEN_CEP.parallel.platform.ParallelExecutionPlatform import ParallelExecutionPlatform
from OPEN_CEP.stream.Stream import *


class DataParallelExecutionAlgorithm(ABC):
    """
    An abstract base class for all data parallel evaluation algorithms.
    """
    def __init__(self, units_number, patterns: Pattern or List[Pattern],
                 eval_mechanism_params: EvaluationMechanismParameters, platform: ParallelExecutionPlatform):
        raise NotImplementedError()

    def eval(self, events: InputStream, matches: OutputStream, data_formatter: DataFormatter):
        """
        Activates the actual parallel algorithm.
        """
        raise NotImplementedError()

    def get_structure_summary(self):
        raise NotImplementedError()
