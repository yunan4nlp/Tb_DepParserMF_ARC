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
        # = encoder_p.get_device() if self.use_cuda else None
        if self.use_cuda:
            self.bucket = self.bucket.cuda()
            self.index = self.index.cuda()
            self.cut = self.cut.cuda()
        self.gold_pred_pairs = []
        self.training = True
        self.batch_states = [] # only for predicting
        self.step = []
        for idx in range(self.config.test_batch_size):
            self.batch_states.append([])
            self.step.append(0)
            for idy in range(0, 1024):
                self.batch_states[idx].append(State())

    def train(self):
        self.encoder.train()
        self.decoder.train()
        self.training = True

    def eval(self):
        self.encoder.eval()
        self.decoder.eval()
        self.training = False

    def encode(self, words, extwords, tags, masks):
        if self.use_cuda:
            words, extwords = words.cuda(), extwords.cuda(),
            tags = tags.cuda()
            masks = masks.cuda()
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

    def compute_accuracy(self, predict_actions, gold_actions):
        total_num = 0
        correct = 0
        batch_size = len(predict_actions)
        assert batch_size == len(gold_actions)
        for b_iter in range(batch_size):
            action_num = len(predict_actions[b_iter])
            assert action_num == len(gold_actions[b_iter])
            for cur_step in range(action_num):
                if predict_actions[b_iter][cur_step] == gold_actions[b_iter][cur_step]:
                    correct += 1
                total_num += 1
        return total_num, correct

    def real_action_num(self, batch_feats):
        a = self.max_action_len(batch_feats)
        b = len(batch_feats)
        self.real_ac_num = []
        for idx in range(b):
            r_a = len(batch_feats[idx])
            self.real_ac_num.append(r_a)

    def get_feats_from_state(self):
        feats = []
        for idx in range(self.b):
            cur_states = self.batch_states[idx]
            cur_step = self.step[idx]
            if not cur_states[cur_step].is_end():
                feat = cur_states[cur_step].prepare_index()
                feats.append([feat])
            else:
                feats.append([None])
        return feats


    def compute(self, feats, vocab):
        hidden_states, hidden_arc, mask = self.hidden_prepare(feats, self.training)  # batch, action_num, hidden_size
        cut = self.get_action_cut(vocab)  # batch, action_num, hidden_size
        self.decoder_outputs = self.decoder.forward(batch_hidden_state=hidden_states,
                                                    batch_hidden_arc=hidden_arc,
                                                    cut=cut,
                                                    mask=mask)

    def decode(self, batch_insts, batch_feats, batch_candid, vocab):
        self.b, _, l2 = self.encoder_outputs.size() #batch, sent_len, hidden_size
        if self.b != self.bucket.size()[0]:
            self.bucket = Variable(torch.zeros(self.b, 1, l2)).type(torch.FloatTensor)
            if self.use_cuda:
                self.bucket = self.bucket.cuda()
        self.encoder_outputs = torch.cat((self.encoder_outputs, self.bucket), 1)#encoder output append the bucket
        global_step = 0
        for idx in range(0, self.b):
            self.step[idx] = 0

        if self.training:
            self.real_action_num(batch_feats)
            self.batch_candid = batch_candid
            self.compute(batch_feats, vocab)
            batch_scores = self.decoder_outputs.data.cpu().numpy()
            predict_actions = self.score2actions(batch_feats, batch_scores, vocab)
            return predict_actions
        else:
            for idx in range(0, self.b):
                start_state = self.batch_states[idx][0]
                start_state.clear()
                start_state.ready(batch_insts[idx], vocab)
            while not self.all_states_are_finished(self.batch_states, self.b):
                batch_feats = self.get_feats_from_state()
                self.compute(batch_feats, vocab)
                batch_scores = self.decoder_outputs.data.cpu().numpy()
                predict_actions = self.score2actions(batch_feats, batch_scores, vocab)
                self.move(predict_actions, self.b)
                global_step += 1

    '''
    def next_gold_feats(self):
        for idx in range(0, self.b):
            cur_feats = self.batch_feats[idx]
            cur_step = self.step[idx]
            if cur_step <= len(cur_feats):
                self.step[idx] += 1
    '''

    def score2actions(self, batch_feats, batch_scores, vocab):
        batch_size = len(batch_feats)
        assert batch_size == len(batch_scores)
        predict_actions = []
        for b_iter in range(batch_size):
            r_a = len(batch_feats[b_iter])
            actions = []
            for cur_step in range(r_a):
                cur_step_action_id = np.argmax(batch_scores[b_iter][cur_step])
                cur_step_action = vocab.id2ac(cur_step_action_id)
                actions.append(cur_step_action)
            predict_actions.append(actions)
        return predict_actions


    def get_predict_actions(self, pred_ac_ids, vocab):
        pred_actions = []
        for ac_id in pred_ac_ids:
            pred_ac = vocab.id2ac(ac_id)
            pred_actions.append(pred_ac)
        return pred_actions

    def all_states_are_finished(self, batch_states, batch_size):
        is_finish = True
        for idx in range(batch_size):
            cur_states = batch_states[idx]
            if not cur_states[self.step[idx]].is_end():
                is_finish = False
                break
        return is_finish

    '''
    def gold_states_are_finished(self):
        is_finish = True
        for idx in range(0, self.b):
            cur_feats = self.batch_feats[idx]
            cur_step = self.step[idx]
            if cur_step < len(cur_feats):
                is_finish = False
                break
        return is_finish
    '''

    def max_action_len(self, batch_feats):
        max_ac_len = -1
        b = len(batch_feats)
        for idx in range(0, b):
            cur_feats = batch_feats[idx]
            tmp = len(cur_feats)
            if tmp > max_ac_len:
                max_ac_len = tmp
        return max_ac_len

    def feat2IndexData(self, feat, data, shape, offset):
        s0, s1, s2, q0, arc = feat
        index_data, index_arc_data, mask_data = data
        idx, idy = offset
        a, b, l1 = shape

        offset_y = idx * l1
        if not arc:
            offset_x = idx * 4 + idy * (b * 4)
            index_data[offset_x] = s0 + offset_y
            index_data[offset_x + 1] = s1 + offset_y
            index_data[offset_x + 2] = s2 + offset_y
            index_data[offset_x + 3] = q0 + offset_y
        else:
            offset_x = idx * 2 + idy * b * 2
            index_arc_data[offset_x] = s0 + offset_y
            index_arc_data[offset_x + 1] = s1 + offset_y
            offset_mask = idx * a + idy
            mask_data[offset_mask][0] = 1

    def hidden_prepare(self, batch_feats, bTrain=True):
        b, l1, l2 = self.encoder_outputs.size() # previous l1 + bucket
        a = self.max_action_len(batch_feats)
        if bTrain:
            assert (l1 - 1) * 3 - 1 == a# training, whole step
        else:
            assert a == 1 # 4 predicting, only one step

        self.a = a
        mask = Variable(torch.zeros(b * a, 1)).type(torch.ByteTensor)
        index = Variable(torch.zeros(b * a * 4)).type(torch.LongTensor)
        index_arc = Variable(torch.zeros(b * a * 2)).type(torch.LongTensor)
        index_data = np.array([l1 - 1] * b * a * 4)
        index_arc_data = np.array([l1 - 1] * b * a * 2)
        mask_data = np.array([[0]] * a * b)
        if self.use_cuda:
            index = index.cuda()
            index_arc = index_arc.cuda()
            mask = mask.cuda()
        #if bTrain:
        for b_iter in range(b):
            feats = batch_feats[b_iter]
            r_a = len(feats)
            if not self.training:
                assert r_a == 1
            for cur_step in range(r_a):
                feat = feats[cur_step]
                if feat is None:
                    break
                offest = b_iter, cur_step
                data = index_data, index_arc_data, mask_data
                shape = a, b, l1
                self.feat2IndexData(feat, data, shape, offest)
        index.data.copy_(torch.from_numpy(index_data))
        index_arc.data.copy_(torch.from_numpy(index_arc_data))
        mask.data.copy_(torch.from_numpy(mask_data))
        h_s = torch.index_select(self.encoder_outputs.view(b * l1 , l2), 0, index)
        h_s = h_s.view(a, b, l2 * 4).permute(1, 0, 2)
        h_arc = torch.index_select(self.encoder_outputs.view(b * l1, l2), 0, index_arc)
        h_arc = h_arc.view(a, b, l2 * 2).permute(1, 0, 2)

        return h_s, h_arc, mask

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
                    ac_ids.append(None)
        else:
            for idx in range(0, self.b):
                r_a = self.real_ac_num[idx]
                if cur_step < r_a:
                    ac_id = np.argmax(action_scores[idx])
                    ac_ids.append(ac_id)
        return ac_ids

    def move(self, pred_actions, batch_size):
        #count = 0
        #for idx in range(0, self.b):
            #cur_states = self.batch_states[idx]
            #if not cur_states[self.step[idx]].is_end():
                #count += 1
        #assert len(pred_actions) == count
        for idx in range(batch_size):
            cur_states = self.batch_states[idx]
            cur_step = self.step[idx]
            if not cur_states[cur_step].is_end():
                next_state = self.batch_states[idx][cur_step + 1]
                cur_states[cur_step].move(next_state, pred_actions[idx][0])
                self.step[idx] += 1

    def get_action_cut(self, vocab): # cut off the impossible action.
        mask_data = np.array([[[0] * vocab.ac_size] * self.a] * self.b, dtype=float)

        if self.training:
            assert self.a > 1
        else:
            assert self.a == 1

        if self.training:#4 training, batch, action_len, hidden_size
            for idx in range(0, self.b):
                r_a = self.real_ac_num[idx]
                for idy in range(0, self.a):
                    #if idy < len(self.batch_feats[idx]):
                    if idy < r_a:
                        mask_data[idx][idy] = self.batch_candid[idx][idy] * -1e+20
        else:#4 predicting, batch, 1, hidden_size
            for idx in range(0, self.b):
                cur_states = self.batch_states[idx]
                cur_step = self.step[idx]
                if not cur_states[cur_step].is_end():
                    mask = cur_states[cur_step].get_candidate_actions(vocab)
                    mask_data[idx][0] = mask
            mask_data = mask_data * -1e+20

        cut = Variable(torch.from_numpy(mask_data).type(torch.FloatTensor))
        if self.use_cuda:
            cut = cut.cuda()
        return cut #batch, action_len, hidden_size

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
