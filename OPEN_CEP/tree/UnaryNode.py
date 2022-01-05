from abc import ABC
from datetime import timedelta
from typing import List, Set

from OPEN_CEP.base.Formula import Formula, RelopTypes, EquationSides
from OPEN_CEP.tree.InternalNode import InternalNode
from OPEN_CEP.tree.Node import Node, PrimitiveEventDefinition
from OPEN_CEP.tree.PatternMatchStorage import TreeStorageParameters


class UnaryNode(InternalNode, ABC):
    """
    Represents an internal tree node with a single child.
    """

    def __init__(
        self,
        sliding_window: timedelta,
        parents: List[Node] = None,
        pattern_ids: int or Set[int] = None,
        event_defs: List[PrimitiveEventDefinition] = None,
        child: Node = None,
    ):
        super().__init__(sliding_window, parents, pattern_ids, event_defs)
        self._child = child

    def get_leaves(self):
        if self._child is None:
            raise Exception("Unary Node with no child")
        return self._child.get_leaves()

    def _propagate_condition(self, condition: Formula):
        self._child.apply_formula(condition)

    def set_subtree(self, child: Node):
        """
        Sets the child node of this node.
        """
        self._child = child
        self._event_defs = child.get_event_definitions()

    def propagate_sliding_window(self, sliding_window: timedelta):
        self.set_sliding_window(sliding_window)
        self._child.propagate_sliding_window(sliding_window)

    def propagate_pattern_id(self, pattern_id: int):
        self.add_pattern_ids({pattern_id})
        self._child.propagate_pattern_id(pattern_id)

    def replace_subtree(self, child: Node):
        """
        Replaces the child of this node with the given node.
        """
        self.set_subtree(child)
        child.add_parent(self)

    def create_parent_to_info_dict(self):
        if self._child is not None:
            self._child.create_parent_to_info_dict()
        super().create_parent_to_info_dict()

    def create_storage_unit(
        self,
        storage_params: TreeStorageParameters,
        sorting_key: callable = None,
        rel_op: RelopTypes = None,
        equation_side: EquationSides = None,
        sort_by_first_timestamp: bool = False,
    ):
        self._init_storage_unit(storage_params, sorting_key, rel_op, equation_side)
        self._child.create_storage_unit(storage_params)

    def get_child(self):
        """
        Returns the child of this unary node.
        """
        return self._child
