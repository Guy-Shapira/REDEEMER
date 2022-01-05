from datetime import timedelta
from typing import List, Set

from OPEN_CEP.base.Event import Event
from OPEN_CEP.base.Formula import Formula, RelopTypes, EquationSides
from OPEN_CEP.base.PatternStructure import PrimitiveEventStructure
from OPEN_CEP.tree.Node import Node, PrimitiveEventDefinition
from OPEN_CEP.tree.PatternMatchStorage import TreeStorageParameters, SortedPatternMatchStorage


class LeafNode(Node):
    """
    A leaf node is responsible for a single event type of the pattern.
    """

    def __init__(
        self,
        sliding_window: timedelta,
        leaf_index: int,
        leaf_event: PrimitiveEventStructure,
        parents: List[Node],
        pattern_ids: int or Set[int] = None,
    ):
        super().__init__(sliding_window, parents, pattern_ids)
        self.__leaf_index = leaf_index
        self.__event_name = leaf_event.name
        self.__event_type = leaf_event.type

    def create_parent_to_info_dict(self):
        """
        Creates the dictionary that maps parent to event type, event name and index.
        This dictionary helps to pass the parents a partial match with the right definitions.
        """
        if len(self._parents) == 0:
            return
        # we call this method before we share nodes so each node has at most one parent
        if len(self._parents) > 1:
            raise Exception(
                "This method should not be called when there is more than one parent."
            )
        self._parent_to_info_dict[self._parents[0]] = [
            PrimitiveEventDefinition(
                self.__event_type, self.__event_name, self.__leaf_index
            )
        ]

    def get_leaves(self):
        return [self]

    def apply_formula(self, formula: Formula):
        condition = formula.get_formula_of(self.__event_name)
        if condition is not None:
            self._condition = condition

    def get_event_definitions(self):
        return [
            PrimitiveEventDefinition(
                self.__event_type, self.__event_name, self.__leaf_index
            )
        ]

    def get_event_type(self):
        """
        Returns the type of events processed by this leaf.
        """
        return self.__event_type

    def get_event_name(self):
        """
        Returns the name of events processed by this leaf.
        """
        return self.__event_name

    def get_leaf_index(self):
        """
        Returns the index of this leaf.
        """
        return self.__leaf_index

    def set_leaf_index(self, index: int):
        """
        Sets the index of this leaf.
        """
        self.__leaf_index = index

    def handle_event(self, event: Event):
        """
        Inserts the given event to this leaf.
        """
        self.clean_expired_partial_matches(event.timestamp)
        self._validate_and_propagate_partial_match([event])

    def _validate_new_match(self, events_for_new_match: List[Event]):
        """
        Validates the condition stored in this node on the given set of events.
        """
        if not super()._validate_new_match(events_for_new_match):
            return False
        binding = {self.__event_name: events_for_new_match[0].payload}
        return self._condition.eval(binding)

    def create_storage_unit(
        self,
        storage_params: TreeStorageParameters,
        sorting_key: callable = None,
        rel_op: RelopTypes = None,
        equation_side: EquationSides = None,
        sort_by_first_timestamp: bool = False,
    ):
        """
        For leaf nodes, we always want to create a sorted storage, since the events arrive in their natural order
        of occurrence anyway. Hence, a sorted storage is initialized either according to a user-specified key, or an
        arrival order if no storage parameters were explicitly specified.
        """
        should_use_default_storage_mode = (
            not storage_params.sort_storage or sorting_key is None
        )
        actual_sorting_key = (
            (lambda pm: pm.events[0].timestamp)
            if should_use_default_storage_mode
            else sorting_key
        )
        actual_sort_by_first_timestamp = (
            should_use_default_storage_mode or sort_by_first_timestamp
        )
        self._partial_matches = SortedPatternMatchStorage(
            actual_sorting_key,
            rel_op,
            equation_side,
            storage_params.clean_up_interval,
            actual_sort_by_first_timestamp,
            True,
        )

    def get_structure_summary(self):
        return self.__event_name

    def is_structure_equivalent(self, other):
        """
        Checks if the type of both of the nodes is the same and then checks if the nodes have the same event_type.
        """
        if type(other) != LeafNode:
            return False
        return self.__event_type == other.get_event_type()

    def propagate_sliding_window(self, sliding_window: timedelta):
        """
        Updates the sliding window of this leaf node.
        """
        self.set_sliding_window(sliding_window)

    def propagate_pattern_id(self, pattern_id: int):
        self.add_pattern_ids({pattern_id})
