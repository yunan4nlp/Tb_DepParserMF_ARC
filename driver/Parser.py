from transition.State import *
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import time

class TransitionBasedParser(object):
    def __init__(self, encoder, decoder, root_id, config, ac_size):
        self.config = config
        self.encoder = encoder
        self.decoder = decoder
        self.root = root_id
        encoder_p = next(filter(lambda p: p.requires_grad, encoder.parameters()))
        self.use_cuda = encoder_p.is_cuda
        self.bucket = Variable(torch.zeros(self.config.train_batch_size, 1, self.config.lstm_hiddens * 2)).type(torch.FloatTensor)
        self.cut = Variable(torch.zeros(self.config.train_batch_size, ac_size)).type(torch.FloatTensor)
        self.index = Variable(torch.zeros(self.config.train_batch_size * 4)).type(torch.LongTensor)
        self.device = encoder_p.get_device() if self.use_cuda else None
        if self.use_cuda:
            self.bucket = self.bucket.cuda(self.device)
            self.index = self.index.cuda(self.device)
            self.cut = self.cut.cuda(self.device)
        self.gold_pred_pairs = []
        self.training = True
        if self.config.train_batch_size > self.config.test_batch_size:
            batch_size = self.config.train_batch_size
        else:
            batch_size = self.config.test_batch_size
        self.batch_states = []
        self.step = []
        for idx in range(0, batch_size):
            self.batch_states.append([])
            self.step.append(0)
            for idy in range(0, 1024):
                self.batch_states[idx].append(State())

    def encode(self, words, extwords, tags, masks):
        if self.use_cuda:
            words, extwords = words.cuda(self.device), extwords.cuda(self.device),
            tags = tags.cuda(self.device)
            masks = masks.cuda(self.device)
        self.encoder_outputs = self.encoder.forward(words, extwords, tags, masks)

    def compute_loss(self, true_acs):
        b, l1, l2 = self.decoder_outputs.size()
        true_acs = _model_var(
            self.encoder,
            pad_sequence(true_acs, length=l1, padding=-1, dtype=np.int64))
        arc_loss = F.cross_entropy(
            self.decoder_outputs.view(b * l1, l2), true_acs.view(b * l1),
            ignore_index=-1)
        return arc_loss

    def compute_accuracy(self):
        total_num = 0
        correct = 0
        for iter in self.gold_pred_pairs:
            gold_len = len(iter[0])
            pred_len = len(iter[1])
            assert gold_len == pred_len
            total_num += gold_len
            for idx in range(0, gold_len):
                if iter[0][idx] == iter[1][idx]:
                    correct += 1
        return total_num, correct

    def decode(self, batch_data, batch_step_actions, batch_feats, batch_candid, vocab):
        self.b, self.l1, self.l2 = self.encoder_outputs.size()
        if self.b != self.bucket.size()[0]:
            self.bucket = Variable(torch.zeros(self.b, 1, self.l2)).type(torch.FloatTensor)
            if self.use_cuda:
                self.bucket = self.bucket.cuda(self.device)
        self.encoder_outputs = torch.cat((self.encoder_outputs, self.bucket), 1)
        global_step = 0
        if self.training:
            self.batch_feats = batch_feats
            self.batch_candid = batch_candid
            for idx in range(0, self.b):
                self.step[idx] = 0
            hidden_states, hidden_arc = self.batch_gold_prepare()
            self.get_global_cut(vocab)
            self.decoder_outputs = self.decoder.forward(batch_hidden_state=hidden_states,
                                                        batch_hidden_arc=hidden_arc,
                                                        cut=self.global_cut,
                                                        mask=self.mask)
            d_outputs = self.decoder_outputs.transpose(0,1)
            self.gold_pred_pairs.clear()
            for idx in range(0, self.a):
                action_scores = d_outputs[idx]
                pred_ac_ids = self.get_predicted_ac_id(action_scores, idx)
                pred_actions = self.get_predict_actions(pred_ac_ids, vocab)
                gold_actions = batch_step_actions[idx]
                self.gold_pred_pairs.append((gold_actions, pred_actions))
        else:
            for idx in range(0, self.b):
                start_state = self.batch_states[idx][0]
                start_state.clear()
                start_state.ready(batch_data[idx], vocab)
                self.step[idx] = 0
            while not self.all_states_are_finished():
                hidden_states, hidden_arc = self.batch_prepare()
                self.get_cut(vocab)
                action_scores = self.decoder.forward(batch_hidden_state=hidden_states,
                                                     batch_hidden_arc=hidden_arc,
                                                     cut=self.cut,
                                                     mask=self.mask)
                pred_ac_ids = self.get_predicted_ac_id(action_scores)
                pred_actions = self.get_predict_actions(pred_ac_ids, vocab)
                self.move(pred_actions, vocab)
                global_step += 1

    def next_gold_feats(self):
        for idx in range(0, self.b):
            cur_feats = self.batch_feats[idx]
            cur_step = self.step[idx]
            if cur_step <= len(cur_feats):
                self.step[idx] += 1

    def get_predict_actions(self, pred_ac_ids, vocab):
        pred_actions = []
        for ac_id in pred_ac_ids:
            pred_ac = vocab.id2ac(ac_id)
            pred_actions.append(pred_ac)
        return pred_actions

    def all_states_are_finished(self):
        is_finish = True
        for idx in range(0, self.b):
            cur_states = self.batch_states[idx]
            if not cur_states[self.step[idx]].is_end():
                is_finish = False
                break
        return is_finish

    def gold_states_are_finished(self):
        is_finish = True
        for idx in range(0, self.b):
            cur_feats = self.batch_feats[idx]
            cur_step = self.step[idx]
            if cur_step < len(cur_feats):
                is_finish = False
                break
        return is_finish

    def batch_prepare(self):
        if self.b != self.index.size()[0]:
            self.mask = Variable(torch.zeros(self.b, 1)).type(torch.ByteTensor)
            self.index = Variable(torch.ones(self.b * 4) * self.l1).type(torch.LongTensor)
            index_data = np.array([self.l1] * self.b * 4)
            self.index_arc = Variable(torch.ones(self.b * 2) * self.l1).type(torch.LongTensor)
            index_arc_data = np.array([self.l1] * self.b * 2)
            if self.use_cuda:
                self.index = self.index.cuda(self.device)
                self.index_arc = self.index_arc.cuda(self.device)
                self.mask = self.mask.cuda(self.device)
        mask_data = np.array([[0]] * self.b)
        for idx in range(0, self.b):
            cur_states = self.batch_states[idx]
            cur_step = self.step[idx]
            if not cur_states[cur_step].is_end():
                s0, s1, s2, q0, arc = cur_states[cur_step].prepare_index()
                offset_y = idx * (self.l1 + 1)
                if not arc:
                    offset_x = idx * 4
                    index_data[offset_x] = s0 + offset_y
                    index_data[offset_x + 1] = s1 + offset_y
                    index_data[offset_x + 2] = s2 + offset_y
                    index_data[offset_x + 3] = q0 + offset_y
                else:
                    offset_x = idx * 2
                    index_arc_data[offset_x] = s0 + offset_y
                    index_arc_data[offset_x + 1] = s1 + offset_y
                    mask_data[idx][0] = 1
        self.mask.data.copy_(torch.from_numpy(mask_data))
        if len(index_data) > 0:
            self.index.data.copy_(torch.from_numpy(index_data))
        h_s = torch.index_select(self.encoder_outputs.view(self.b * (self.l1 + 1), self.l2), 0, self.index)
        h_s = h_s.view(self.b, 4 * self.l2)
        if len(index_arc_data) > 0:
            self.index_arc.data.copy_(torch.from_numpy(index_arc_data))
        h_arc = torch.index_select(self.encoder_outputs.view(self.b * (self.l1 + 1), self.l2), 0, self.index_arc)
        h_arc = h_arc.view(self.b, 2 * self.l2)
        return h_s, h_arc

    def max_action_len(self):
        max_ac_len = -1
        for idx in range(0, self.b):
            cur_feats = self.batch_feats[idx]
            tmp = len(cur_feats)
            if tmp > max_ac_len:
                max_ac_len = tmp
        return max_ac_len

    def gold_arc_action_num(self):
        sum = 0
        for idx in range(0, self.b):
            cur_feats = self.batch_feats[idx]
            sum += ((len(cur_feats) + 1) // 3 - 1)
        return sum

    def get_global_cut(self, vocab):
        self.global_cut = Variable(torch.zeros(self.b, self.a, vocab.ac_size)).type(torch.FloatTensor)
        mask_data = np.array([[[0]*vocab.ac_size]*self.a]*self.b, dtype=float)
        if self.use_cuda:
            self.global_cut = self.global_cut.cuda(self.device)
        for idx in range(0, self.b):
            for idy in range(0, self.a):
                if idy < len(self.batch_feats[idx]):
                    mask_data[idx][idy] = self.batch_candid[idx][idy] * -1e+20
        self.global_cut.data.copy_(torch.from_numpy(mask_data))

    def batch_gold_prepare(self):
        self.arc_num = self.gold_arc_action_num()
        self.a = self.max_action_len()
        self.mask = Variable(torch.zeros(self.b * self.a, 1)).type(torch.ByteTensor)
        self.gold_index = Variable(torch.zeros(self.b * self.a * 4)).type(torch.LongTensor)
        self.gold_index_arc = Variable(torch.zeros(self.b * self.a * 2)).type(torch.LongTensor)
        index_data = np.array([self.l1] * self.b * self.a * 4)
        index_arc_data = np.array([self.l1] * self.b * self.a * 2)
        mask_data = np.array([[0]] * self.a * self.b)
        if self.use_cuda:
            self.gold_index = self.gold_index.cuda(self.device)
            self.gold_index_arc = self.gold_index_arc.cuda(self.device)
            self.mask = self.mask.cuda(self.device)
        for idx in range(0, self.b):
            self.step[idx] = 0
        while not self.gold_states_are_finished():
            for idx in range(0, self.b):
                cur_feats = self.batch_feats[idx]
                cur_step = self.step[idx]
                if cur_step < len(cur_feats):
                    s0, s1, s2, q0, arc = cur_feats[cur_step]
                    offset_y = idx * (self.l1 + 1)
                    if not arc:
                        offset_x = idx * 4 + cur_step * (self.b * 4)
                        index_data[offset_x] = s0 + offset_y
                        index_data[offset_x + 1] = s1 + offset_y
                        index_data[offset_x + 2] = s2 + offset_y
                        index_data[offset_x + 3] = q0 + offset_y
                    else:
                        offset_x = idx * 2 + cur_step * self.b * 2
                        index_arc_data[offset_x] = s0 + offset_y
                        index_arc_data[offset_x + 1] = s1 + offset_y
                        offset_mask = idx * self.a + cur_step
                        mask_data[offset_mask][0] = 1
            self.next_gold_feats()
        self.gold_index.data.copy_(torch.from_numpy(index_data))
        self.gold_index_arc.data.copy_(torch.from_numpy(index_arc_data))
        self.mask.data.copy_(torch.from_numpy(mask_data))
        h_s = torch.index_select(self.encoder_outputs.view(self.b * (self.l1 + 1), self.l2), 0, self.gold_index)
        h_s = h_s.view(self.a, self.b, self.l2 * 4).permute(1, 0, 2)
        h_arc = torch.index_select(self.encoder_outputs.view(self.b * (self.l1 + 1), self.l2), 0, self.gold_index_arc)
        h_arc = h_arc.view(self.a, self.b, self.l2 * 2).permute(1, 0, 2)
        return h_s, h_arc

    def get_predicted_ac_id(self, action_scores, cur_step=None):
        ac_ids = []
        action_scores = action_scores.data.cpu().numpy()
        if not self.training:
            for idx in range(0, self.b):
                cur_states = self.batch_states[idx]
                if not cur_states[self.step[idx]].is_end():
                    ac_id = np.argmax(action_scores[idx])
                    ac_ids.append(ac_id)
        else:
            for idx in range(0, self.b):
                cur_feats = self.batch_feats[idx]
                if cur_step < len(cur_feats):
                    ac_id = np.argmax(action_scores[idx])
                    ac_ids.append(ac_id)
        return ac_ids

    def move(self, pred_actions, vocab):
        #count = 0
        #for idx in range(0, self.b):
            #cur_states = self.batch_states[idx]
            #if not cur_states[self.step[idx]].is_end():
                #count += 1
        #assert len(pred_actions) == count
        offset = 0
        for idx in range(0, self.b):
            cur_states = self.batch_states[idx]
            cur_step = self.step[idx]
            if not cur_states[cur_step].is_end():
                next_state = self.batch_states[idx][cur_step + 1]
                cur_states[cur_step].move(next_state, pred_actions[offset])
                offset += 1
                self.step[idx] += 1


    def get_cut(self, vocab):
        all_mask = np.array([[False] * vocab.ac_size] * self.b)
        mask_data = np.array([[0] * vocab.ac_size] * self.b, dtype=float)
        for idx in range(0, self.b):
            cur_states = self.batch_states[idx]
            cur_step = self.step[idx]
            if not cur_states[cur_step].is_end():
                mask = cur_states[cur_step].get_candidate_actions(vocab)
                all_mask[idx] = mask
        if self.b != self.cut.size()[0]:
            self.cut = Variable(torch.zeros(self.b, vocab.ac_size)).type(torch.FloatTensor)
            if self.use_cuda:
                self.cut = self.cut.cuda(self.device)
        mask_data = all_mask * -1e+20
        self.cut.data.copy_(torch.from_numpy(mask_data))

def _model_var(model, x):
    p = next(filter(lambda p: p.requires_grad, model.parameters()))
    if p.is_cuda:
        x = x.cuda(p.get_device())
    return torch.autograd.Variable(x)

def pad_sequence(xs, length=None, padding=-1, dtype=np.float64):
    lengths = [len(x) for x in xs]
    if length is None:
        length = max(lengths)
    y = np.array([np.pad(x.astype(dtype), (0, length - l),
                         mode="constant", constant_values=padding)
                  for x, l in zip(xs, lengths)])
    return torch.from_numpy(y)
