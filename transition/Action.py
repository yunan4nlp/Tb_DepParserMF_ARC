from enum import Enum
from data.Vocab import *


class CODE(Enum):
    SHIFT = 0
    ARC_LEFT = 1
    ARC_RIGHT = 2
    POP_ROOT = 3
    ARC_LABEL = 4
    NO_ACTION = 5


class Action:
    ## for hash
    __hash__ = object.__hash__

    def __init__(self, code, label):
        self.code = code
        self.label = label

    ## for dic key
    def __hash__(self):
        return hash(str(self.code) + str(self.label))

    def set(self, code, label):
        self.code = code
        self.label = label

    def is_shift(self):
        return self.code == CODE.SHIFT

    def is_arc_left(self):
        return self.code == CODE.ARC_LEFT

    def is_arc_right(self):
        return self.code == CODE.ARC_RIGHT

    def is_arc_label(self):
        return self.code == CODE.ARC_LABEL

    def is_finish(self):
        return self.code == CODE.POP_ROOT

    def is_none(self):
        return self.code == CODE.NO_ACTION

    def __eq__(self, other):
        return other.code == self.code and other.label == self.label

    def str(self, vocab):
        if self.is_shift():
            return "shift"
        elif self.is_arc_left():
            return "arc_left"
        elif self.is_arc_right():
            return "arc_right"
        elif self.is_arc_label():
            return "arc_label_" + vocab._id2rel[self.label]
        elif self.is_finish():
            return "pop_root"
        else:
            return "no_action"
