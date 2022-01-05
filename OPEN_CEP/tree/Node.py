from abc import ABC
from datetime import timedelta, datetime
from queue import Queue
from typing import List, Set

from OPEN_CEP.base.Event import Event
from OPEN_CEP.base.Formula import TrueFormula, Formula, RelopTypes, EquationSides
from OPEN_CEP.base.PatternMatch import PatternMatch
from OPEN_CEP.tree.PatternMatchStorage import TreeStorageParameters


class PrimitiveEventDefinition:
    """
    An internal class for capturing the information regarding a single primitive event appearing in a pattern.
    """

    def __init__(self, event_type: str, event_name: str, event_index: int):
        self.type = event_type
        self.name = event_name
        self.index = event_index


class Node(ABC):
    """
    This class represents a single node of an evaluation tree.
    """

    # A static variable specifying whether the system is allowed to delete expired matches.
    # In several very special cases, it is required to switch off this functionality.
    __enable_partial_match_expiration = True

    @staticmethod
    def _toggle_enable_partial_match_expiration(enable):
        """
        Sets the value of the __enable_partial_match_expiration flag.
        """
        Node.__enable_partial_match_expiration = enable

    @staticmethod
    def _is_partial_match_expiration_enabled():
        """
        Returns the static variable specifying whether match expiration is enabled.
        """
        return Node.__enable_partial_match_expiration

    def __init__(
        self, sliding_window: timedelta, parents, pattern_ids: int or Set[int] = None
    ):
        self._parents = []
        self._sliding_window = sliding_window
        self._partial_matches = None
        self._condition = TrueFormula()

        # Full pattern matches that were not yet reported. Only relevant for an output node, that is, for a node
        # corresponding to a full pattern definition.
        self._unreported_matches = Queue()
        self._is_output_node = False

        # set of event types that will only appear in a single full match
        self._single_event_types = set()
        # events that were added to a partial match and cannot be added again
        self._filtered_events = set()

        # set of pattern IDs with which this node is associated
        if pattern_ids is None:
            pattern_ids = set()
        elif isinstance(pattern_ids, int):
            pattern_ids = {pattern_ids}
        self._pattern_ids = pattern_ids

        # Maps parent to event definition. This field helps to pass the parents a partial match with
        # the right event definitions.
        self._parent_to_info_dict = {}
        # matches that were not yet pushed to the parents for further processing
        self._parent_to_unhandled_queue_dict = {}

        self.set_parents(parents, on_init=True)

    def get_next_unreported_match(self):
        """
        Removes and returns an unreported match buffered at this node.
        Used in an output node to collect full pattern matches.
        """
        ret = self._unreported_matches.get()
        return ret

    def has_unreported_matches(self):
        """
        Returns True if this node contains any matches we did not report yet and False otherwise.
        """
        return self._unreported_matches.qsize() > 0

    def get_last_unhandled_partial_match_by_parent(self, parent):
        """
        Returns the last partial match buffered at this node and not yet transferred to parent.
        """
        return self._parent_to_unhandled_queue_dict[parent].get(block=False)

    def set_parents(self, parents, on_init: bool = False):
        """
        Sets the parents of this node to the given list of nodes. Providing None as the parameter will render
        this node parentless.
        """
        if parents is None:
            parents = []
        elif isinstance(parents, Node):
            # a single parent was specified
            parents = [parents]
        self._parents = []
        self._parent_to_unhandled_queue_dict = {}
        self._parent_to_info_dict = {}
        for parent in parents:
            self.add_parent(parent, on_init)

    def set_parent(self, parent):
        """
        A more intuitive API for setting the parent list of a node to a single parent.
        Simply invokes set_parents as the latter already supports the case of a single node instead of a list.
        """
        self.set_parents(parent)

    def add_parent(self, parent, on_init: bool = False):
        """
        Adds a parent to this node.
        """
        if parent in self._parents:
            return
        self._parents.append(parent)
        self._parent_to_unhandled_queue_dict[parent] = Queue()
        if not on_init:
            self._parent_to_info_dict[parent] = self.get_event_definitions()

    def get_parents(self):
        """
        Returns the parents of this node.
        """
        return self._parents

    def get_event_definitions_by_parent(self, parent):
        """
        Returns the event definitions according to the parent.
        """
        if parent not in self._parent_to_info_dict.keys():
            raise Exception("parent is not in the dictionary.")
        return self._parent_to_info_dict[parent]

    def get_sliding_window(self):
        """
        Returns the sliding window of this node.
        """
        return self._sliding_window

    def set_sliding_window(self, new_sliding_window: timedelta):
        """
        Sets the sliding window of this node.
        """
        self._sliding_window = new_sliding_window

    def get_pattern_ids(self):
        """
        Returns the pattern ids of this node.
        """
        return self._pattern_ids

    def get_condition(self):
        """
        Returns the condition of this node.
        """
        return self._condition

    def add_pattern_ids(self, ids: Set[int]):
        """
        Adds a set of Ds of patterns with which this node is associated.
        """
        self._pattern_ids |= ids

    def set_is_output_node(self, is_output_node: bool):
        """
        Sets this node to be defined as an output node according to the given parameter.
        """
        self._is_output_node = is_output_node

    def is_output_node(self):
        """
        Returns whether this node is an output node.
        """
        return self._is_output_node

    def clean_expired_partial_matches(self, last_timestamp: datetime):
        """
        Removes partial matches whose earliest timestamp violates the time window constraint.
        Also removes the expired filtered events if the "single" consumption policy is enabled.
        """
        if not Node._is_partial_match_expiration_enabled():
            return
        self._partial_matches.try_clean_expired_partial_matches(
            last_timestamp - self._sliding_window
        )
        if len(self._single_event_types) == 0:
            # "single" consumption policy is disabled or no event types under the policy reach this node
            return
        self._filtered_events = set(
            [
                event
                for event in self._filtered_events
                if event.timestamp >= last_timestamp - self._sliding_window
            ]
        )

    def register_single_event_type(self, event_type: str):
        """
        Add the event type to the internal set of event types for which "single" consumption policy is enabled.
        Recursively updates the ancestors of the node.
        """
        self._single_event_types.add(event_type)
        for parent in self._parents:
            parent.register_single_event_type(event_type)

    def _add_partial_match(self, pm: PatternMatch):
        """
        Registers a new partial match at this node.
        In case of SortedPatternMatchStorage the insertion is by timestamp or condition, O(log n).
        In case of UnsortedPatternMatchStorage the insertion is directly at the end, O(1).
        """
        self._partial_matches.add(pm)
        for parent in self._parents:
            self._parent_to_unhandled_queue_dict[parent].put(pm)
            parent.handle_new_partial_match(self)
        if self.is_output_node():
            self._unreported_matches.put(pm)

    def __can_add_partial_match(self, pm: PatternMatch) -> bool:
        """
        Returns True if the given partial match can be passed up the tree and False otherwise.
        As of now, only the activation of the "single" consumption policy might prevent this method from returning True.
        In addition, this method updates the filtered events set.
        """
        if len(self._single_event_types) == 0:
            return True
        new_filtered_events = set()
        for event in pm.events:
            if event.type not in self._single_event_types:
                continue
            if event in self._filtered_events:
                # this event was already passed
                return False
            else:
                # this event was not yet passed but should only be passed once - remember it
                new_filtered_events.add(event)
        self._filtered_events |= new_filtered_events
        return True

    def _validate_and_propagate_partial_match(self, events: List[Event]):
        """
        Creates a new partial match from the list of events, validates it, and propagates it up the tree.
        """
        if not self._validate_new_match(events):
            return
        self._propagate_partial_match(events)

    def _propagate_partial_match(self, events: List[Event]):
        """
        Receives an already verified list of events for new partial match and propagates it up the tree.
        """
        new_partial_match = PatternMatch(events)
        if self.__can_add_partial_match(new_partial_match):
            self._add_partial_match(new_partial_match)

    def get_partial_matches(self, filter_value: int or float = None):
        """
        Returns only partial matches that can be a good fit the partial match identified by the given filter value.
        """
        return (
            self._partial_matches.get(filter_value)
            if filter_value is not None
            else self._partial_matches.get_internal_buffer()
        )

    def get_storage_unit(self):
        """
        Returns the internal partial match storage of this node.
        """
        return self._partial_matches

    def _validate_new_match(self, events_for_new_match: List[Event]):
        """
        Validates the condition stored in this node on the given set of events.
        """
        min_timestamp = min([event.timestamp for event in events_for_new_match])
        max_timestamp = max([event.timestamp for event in events_for_new_match])
        return max_timestamp - min_timestamp <= self._sliding_window

    def is_equivalent(self, other):
        """
        Returns True if the given node is equivalent to this one and False otherwise.
        Two nodes are considered equivalent if they possess equivalent structures and verify equivalent conditions.
        """
        # TODO: after the conditions branch is merged, the condition equivalence will no longer work and will need to be fixed ASAP
        return (
            self.is_structure_equivalent(other)
            and self._condition == other.get_condition()
        )

    def propagate_sliding_window(self, sliding_window: timedelta):
        """
        Propagates the given sliding window down the subtree of this node.
        """
        raise NotImplementedError()

    def propagate_pattern_id(self, pattern_id: int):
        """
        Propagates the given pattern ID down the subtree of this node.
        """
        raise NotImplementedError()

    def create_parent_to_info_dict(self):
        """
        Traverses the subtree of this node and initializes the internal dictionaries mapping each parent node to the
        corresponding event definitions.
        To be implemented by subclasses.
        """
        raise NotImplementedError()

    def get_leaves(self):
        """
        Returns all leaves in this tree - to be implemented by subclasses.
        """
        raise NotImplementedError()

    def apply_formula(self, formula: Formula):
        """
        Applies a given formula on all nodes in this tree - to be implemented by subclasses.
        """
        raise NotImplementedError()

    def get_event_definitions(self) -> List[PrimitiveEventDefinition]:
        """
        Returns the specifications of all events collected by this tree - to be implemented by subclasses.
        """
        raise NotImplementedError()

    def get_structure_summary(self):
        """
        Returns the summary of the subtree rooted at this node - to be implemented by subclasses.
        """
        raise NotImplementedError()

    def is_structure_equivalent(self, other):
        """
        Returns True if the structure of the subtree of this node is equivalent to the one of the given node and
        False otherwise.
        To be implemented by subclasses.
        """
        raise NotImplementedError()

    def create_storage_unit(
        self,
        storage_params: TreeStorageParameters,
        sorting_key: callable = None,
        rel_op: RelopTypes = None,
        equation_side: EquationSides = None,
        sort_by_first_timestamp: bool = False,
    ):
        """
        An abstract method for recursive partial match storage initialization.
        """
        raise NotImplementedError()
