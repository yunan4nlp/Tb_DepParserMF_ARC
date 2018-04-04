from transition.Action import *
from transition.Instance import *
from transition.AtomFeat import *
from data.Dependency import *
import torch
from torch.autograd import Variable
import numpy as np

max_length = 512


class State:
    def __init__(self):
        self._stack = [-3] * max_length
        self._stack_size = 0
        self._rel = [-3] * max_length
        self._head = [-3] * max_length
        self._have_parent = [-1] * max_length
        self._next_index = 0
        self._word_size = 0
        self._is_start = True
        self._is_gold = True
        self._inst = None
        self._atom_feat = AtomFeat()
        self._pre_action = Action(CODE.NO_ACTION, -1)

    def ready(self, sentence, vocab):
        self._inst = Instance(sentence, vocab)
        self._word_size = len(self._inst.words)

    def clear(self):
        self._next_index = 0
        self._stack_size = 0
        self._word_size = 0
        self._is_gold = True
        self._is_start = True
        self._pre_action = Action(CODE.NO_ACTION, -1)

        self.done_mark()

    def done_mark(self):
        self._stack[self._stack_size] = -2
        self._head[self._next_index] = -2
        self._rel[self._next_index] = -2
        self._have_parent[self._next_index] = -2

    def allow_shift(self):
        if self._next_index < self._word_size:
            return True
        else:
            return False

    def allow_arc_left(self):
        if self._stack_size > 1:
            return True
        else:
            return False

    def allow_arc_right(self):
        if self._stack_size > 1:
            return True
        else:
            return False

    def allow_pop_root(self):
        if self._stack_size == 1 and self._next_index == self._word_size:
            return True
        else:
            return False

    def allow_arc_label(self):
        if self._pre_action.is_arc_left() or self._pre_action.is_arc_right():
            return True
        else:
            return False

    def shift(self, next_state):
        assert self._next_index < self._word_size
        next_state._next_index = self._next_index + 1
        next_state._stack_size = self._stack_size + 1
        self.copy_state(next_state)
        next_state._stack[next_state._stack_size - 1] = self._next_index
        next_state._have_parent[self._next_index] = 0
        next_state.done_mark()
        next_state._pre_action.set(CODE.SHIFT, -1)

    def arc_left(self, next_state):
        assert self._stack_size > 1
        next_state._next_index = self._next_index
        next_state._stack_size = self._stack_size
        self.copy_state(next_state)
        next_state.done_mark()
        next_state._pre_action.set(CODE.ARC_LEFT, -1)

    def arc_right(self, next_state):
        assert self._stack_size > 1
        next_state._next_index = self._next_index
        next_state._stack_size = self._stack_size
        self.copy_state(next_state)
        next_state.done_mark()
        next_state._pre_action.set(CODE.ARC_RIGHT, -1)

    def arc_label(self, next_state, dep):
        assert self._stack_size > 1
        next_state._next_index = self._next_index
        next_state._stack_size = self._stack_size - 1
        self.copy_state(next_state)
        top0 = self._stack[self._stack_size - 1]
        top1 = self._stack[self._stack_size - 2]
        if (self._pre_action.is_arc_left()):
            next_state._stack[next_state._stack_size - 1] = top0
            next_state._head[top1] = top0
            next_state._have_parent[top1] = 1
            next_state._rel[top1] = dep
        else:
            next_state._head[top0] = top1
            next_state._have_parent[top0] = 1
            next_state._rel[top0] = dep
        next_state.done_mark()
        next_state._pre_action.set(CODE.ARC_LABEL, dep)

    def pop_root(self, next_state, dep):
        assert  self._stack_size == 1 and self._next_index == self._word_size
        next_state._next_index = self._word_size
        next_state._stack_size = 0
        self.copy_state(next_state)
        top0 = self._stack[self._stack_size - 1]
        next_state._head[top0] = -1
        next_state._have_parent[top0] = 1
        next_state._rel[top0] = dep
        next_state.done_mark()
        next_state._pre_action.set(CODE.POP_ROOT, dep)

    def move(self, next_state, action):
        next_state._is_start = False
        next_state._is_gold = False
        if action.is_shift():
            self.shift(next_state)
        elif action.is_arc_left():
            self.arc_left(next_state)
        elif action.is_arc_right():
            self.arc_right(next_state)
        elif action.is_arc_label():
            self.arc_label(next_state, action.label)
        elif action.is_finish():
            self.pop_root(next_state, action.label)
        else:
            print(" error state ")

    def get_candidate_actions(self, vocab):
        mask = np.array([False]*vocab.ac_size)

        if self.allow_arc_label():
            mask = mask | vocab.mask_arc_label
            return ~mask
        if self.allow_arc_left():
            mask = mask | vocab.mask_arc_left
        if self.allow_arc_right():
            mask = mask | vocab.mask_arc_right

        if self.is_end():
            mask = mask | vocab.mask_no_action

        if self.allow_shift():
            mask = mask | vocab.mask_shift


        if self.allow_pop_root():
            mask = mask | vocab.mask_pop_root
        return ~mask

    def copy_state(self, next_state):
        next_state._inst = self._inst
        next_state._word_size = self._word_size
        next_state._stack[0:self._stack_size] = (self._stack[0:self._stack_size])
        next_state._rel[0:self._next_index] = (self._rel[0:self._next_index])
        next_state._head[0:self._next_index] = (self._head[0:self._next_index])
        next_state._have_parent[0:self._next_index] = (self._have_parent[0:self._next_index])

    def is_end(self):
        if self._pre_action.is_finish():
            return True
        else:
            return False

    def get_gold_action(self, vocab):
        gold_action = Action(CODE.NO_ACTION, -1)
        if self._stack_size == 0:
            gold_action.set(CODE.SHIFT, -1)
        elif self._stack_size == 1:
            if self._next_index == self._word_size:
                gold_action.set(CODE.POP_ROOT, vocab.ROOT)
            else:
                gold_action.set(CODE.SHIFT, -1)
        elif self._pre_action.is_arc_left() or self._pre_action.is_arc_right():# arc label
            assert self._stack_size > 1
            top0 = self._stack[self._stack_size - 1]
            top1 = self._stack[self._stack_size - 2]
            if self._pre_action.is_arc_left():
                gold_action.set(CODE.ARC_LABEL, vocab._rel2id[self._inst.rels[top1]])
            elif self._pre_action.is_arc_right():
                gold_action.set(CODE.ARC_LABEL, vocab._rel2id[self._inst.rels[top0]])
        elif self._stack_size > 1:  # arc
            top0 = self._stack[self._stack_size - 1]
            top1 = self._stack[self._stack_size - 2]
            assert top0 < self._word_size and top1 < self._word_size
            if top0 == self._inst.heads[top1]:  # top1 <- top0
                gold_action.set(CODE.ARC_LEFT, -1)
            elif top1 == self._inst.heads[top0]: # top1 -> top0,
                # if top0 have right child, shift.
                have_right_child = False
                for idx in range(self._next_index, self._word_size):
                    if self._inst.heads[idx] == top0:
                        have_right_child = True
                        break
                if have_right_child:
                    gold_action.set(CODE.SHIFT, -1)
                else:
                    gold_action.set(CODE.ARC_RIGHT, -1)
            else:  # can not arc
                gold_action.set(CODE.SHIFT, -1)
        return gold_action

    def get_result(self, vocab):
        result = []
        result.append(Dependency(0, vocab._root_form, vocab._root, 0, vocab._root))
        for idx in range(0, self._word_size):
            assert self._have_parent[idx] == 1
            relation = vocab.id2rel(self._rel[idx])
            head = self._head[idx]
            word = self._inst.words[idx]
            tag = self._inst.tags[idx]
            result.append(Dependency(idx + 1, word, tag, head + 1, relation))
        return result

    def prepare_index(self):
        if self._stack_size > 0:
            self._atom_feat.s0 = self._stack[self._stack_size - 1]
        else:
            self._atom_feat.s0 = self._word_size
        if self._stack_size > 1:
            self._atom_feat.s1 = self._stack[self._stack_size - 2]
        else:
            self._atom_feat.s1 = self._word_size
        if self._stack_size > 2:
            self._atom_feat.s2 = self._stack[self._stack_size - 3]
        else:
            self._atom_feat.s2 = self._word_size
        if self._next_index >= 0 and self._next_index < self._word_size:
            self._atom_feat.q0 = self._next_index
        else:
            self._atom_feat.q0 = self._word_size

        if self._pre_action.is_arc_left() or self._pre_action.is_arc_right():
            self._atom_feat.arc = True
        else:
            self._atom_feat.arc = False

        return self._atom_feat.index()
