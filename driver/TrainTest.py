import sys
sys.path.extend(["../../","../","./"])
import time
import torch.optim.lr_scheduler
import torch.nn as nn
import random
import argparse
from driver.Model import *
from data.Dataloader import *
from driver.Config import *
from driver.Parser import *
from transition.Action import *
from transition.State import *
from transition.Instance import *
from copy import deepcopy
import pickle

def get_gold_actions(data, vocab):
    all_actions = []
    states = []
    for idx in range(0, 1024):
        states.append(State())
    all_feats = []
    for sentence in data:
        start = states[0]
        start.clear()
        start.ready(sentence, vocab)
        actions = []
        step = 0
        inst_feats = []
        while not states[step].is_end():
            gold_action = states[step].get_gold_action(vocab)
            gold_feats = states[step].prepare_index()
            inst_feats.append(deepcopy(gold_feats))
            actions.append(gold_action)
            next_state = states[step + 1]
            states[step].move(next_state, gold_action)
            step += 1
        all_feats.append(inst_feats)
        all_actions.append(actions)
        result = states[step].get_result(vocab)
        arc_total, arc_correct, rel_total, rel_correct = evalDepTree(sentence, result)
        assert arc_total == arc_correct and rel_total == rel_correct
        assert len(actions) == (len(sentence) - 1) * 3 - 1
    return all_feats, all_actions

def get_gold_candid(data, vocab):
    states = []
    all_candid = []
    for idx in range(0, 1024):
        states.append(State())
    for sentence in data:
        start = states[0]
        start.clear()
        start.ready(sentence, vocab)
        step = 0
        inst_candid = []
        while not states[step].is_end():
            gold_action = states[step].get_gold_action(vocab)
            candid = states[step].get_candidate_actions(vocab)
            inst_candid.append(candid)
            next_state = states[step + 1]
            states[step].move(next_state, gold_action)
            step += 1
        all_candid.append(inst_candid)
    return all_candid

def inst(data, actions, feats, candid):
    assert len(data) == len(actions) == len(feats) == len(candid)
    inst = []
    for idx in range(len(data)):
        inst.append((data[idx], actions[idx], feats[idx], candid[idx]))
    return inst

def train(train_inst, dev_data, test_data, parser, vocab, config):
    encoder_optimizer = Optimizer(filter(lambda p: p.requires_grad, parser.encoder.parameters()), config)
    decoder_optimizer = Optimizer(filter(lambda p: p.requires_grad, parser.decoder.parameters()), config)

    global_step = 0
    best_UAS = 0
    batch_num = int(np.ceil(len(train_inst) / float(config.train_batch_size)))
    for iter in range(config.train_iters):
        start_time = time.time()
        print('Iteration: ' + str(iter))
        batch_iter = 0

        overall_action_correct,  overall_total_action = 0, 0
        for onebatch in data_iter(train_inst, config.train_batch_size, True):
            words, extwords, tags, heads, rels, lengths, masks, sents, \
            gold_actions, acs, gold_step_actions, gold_feats, gold_candid = \
                batch_data_variable_actions(onebatch, vocab)
            parser.encoder.train()
            parser.decoder.train()
            parser.training = True
            #with torch.autograd.profiler.profile() as prof:
            parser.encode(words, extwords, tags, masks)
            parser.decode(sents, gold_step_actions, gold_feats, gold_candid, vocab)
            loss = parser.compute_loss(acs)
            loss = loss / config.update_every
            loss_value = loss.data.cpu().numpy()
            loss.backward()
            #print(prof.key_averages())
            total_actions, correct_actions = parser.compute_accuracy()
            overall_total_action += total_actions
            overall_action_correct += correct_actions
            during_time = float(time.time() - start_time)
            acc = overall_action_correct / overall_total_action
            #acc = 0
            print("Step:%d, Iter:%d, batch:%d, time:%.2f, acc:%.2f, loss:%.2f"
                  %(global_step, iter, batch_iter,  during_time, acc, loss_value))
            batch_iter += 1

            if batch_iter % config.update_every == 0 or batch_iter == batch_num:
                nn.utils.clip_grad_norm(filter(lambda p: p.requires_grad, parser.encoder.parameters()), \
                                        max_norm=config.clip)
                nn.utils.clip_grad_norm(filter(lambda p: p.requires_grad, parser.decoder.parameters()), \
                                        max_norm=config.clip)
                encoder_optimizer.step()
                decoder_optimizer.step()
                parser.encoder.zero_grad()
                parser.decoder.zero_grad()
                global_step += 1

            if batch_iter % config.validate_every == 0 or batch_iter == batch_num:
                arc_correct, rel_correct, arc_total, dev_uas, dev_las = \
                evaluate(dev_data, parser, vocab, config.dev_file + '.' + str(global_step))
                print("Dev: uas = %d/%d = %.2f, las = %d/%d =%.2f" % \
                        (arc_correct, arc_total, dev_uas, rel_correct, arc_total, dev_las))
                arc_correct, rel_correct, arc_total, test_uas, test_las = \
                evaluate(test_data, parser, vocab, config.test_file + '.' + str(global_step))
                print("Test: uas = %d/%d = %.2f, las = %d/%d =%.2f" % \
                        (arc_correct, arc_total, test_uas, rel_correct, arc_total, test_las))
                if dev_uas > best_UAS:
                    print("Exceed best uas: history = %.2f, current = %.2f" % (best_UAS, dev_uas))
                    best_UAS = dev_uas
                if config.save_after > 0 and iter > config.save_after:
                    torch.save(parser.model.state_dict(), config.save_model_path)


def evaluate(data, parser, vocab, outputFile):
    start = time.time()
    parser.training = False
    parser.encoder.eval()
    parser.decoder.eval()
    output = open(outputFile, 'w', encoding='utf-8')
    arc_total_test, arc_correct_test, rel_total_test, rel_correct_test = 0, 0, 0, 0
    for onebatch in data_iter(data, config.test_batch_size, False):
        words, extwords, tags, heads, rels, lengths, masks = \
            batch_data_variable(onebatch, vocab)
        count = 0
        parser.encode(words, extwords, tags, masks)
        parser.decode(onebatch, None, None, None, vocab)
        for idx in range(0, parser.b):
            cur_states = parser.batch_states[idx]
            cur_step = parser.step[idx]
            tree = cur_states[cur_step].get_result(vocab)
            printDepTree(output, tree)
            arc_total, arc_correct, rel_total, rel_correct = evalDepTree(onebatch[idx], tree)
            arc_total_test += arc_total
            arc_correct_test += arc_correct
            rel_total_test += rel_total
            rel_correct_test += rel_correct
            count += 1
    output.close()

    uas = arc_correct_test * 100.0 / arc_total_test
    las = rel_correct_test * 100.0 / rel_total_test

    end = time.time()
    during_time = float(end - start)
    print("sentence num: %d,  parser time = %.2f " % (len(data), during_time))
    return arc_correct_test, rel_correct_test, arc_total_test, uas, las





class Optimizer:
    def __init__(self, parameter, config):
        self.optim = torch.optim.Adam(parameter, lr=config.learning_rate, betas=(config.beta_1, config.beta_2),
                                      eps=config.epsilon)
        decay, decay_step = config.decay, config.decay_steps
        l = lambda epoch: decay ** (epoch // decay_step)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=l)

    def step(self):
        self.optim.step()
        self.schedule()
        self.optim.zero_grad()

    def schedule(self):
        self.scheduler.step()

    def zero_grad(self):
        self.optim.zero_grad()

    @property
    def lr(self):
        return self.scheduler.get_lr()

if __name__ == '__main__':
    random.seed(666)
    np.random.seed(666)
    torch.cuda.manual_seed(666)
    torch.manual_seed(666)

    ### gpu
    gpu = torch.cuda.is_available()
    print("GPU available: ", gpu)
    print("CuDNN: \n", torch.backends.cudnn.enabled)

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config_file', default='examples/default.cfg')
    argparser.add_argument('--model', default='BaseParser')
    argparser.add_argument('--thread', default=4, type=int, help='thread num')
    argparser.add_argument('--use-cuda', action='store_true', default=True)

    args, extra_args = argparser.parse_known_args()
    config = Configurable(args.config_file, extra_args)

    vocab = creatVocab(config.train_file, config.min_occur_count)
    vec = vocab.load_pretrained_embs(config.pretrained_embeddings_file)
    pickle.dump(vocab, open(config.save_vocab_path, 'wb'))
    torch.set_num_threads(args.thread)

    config.use_cuda = False
    if gpu and args.use_cuda: config.use_cuda = True
    print("\nGPU using status: ", config.use_cuda)


    train_data = read_train_corpus(config.train_file, vocab)
    dev_data = read_corpus(config.dev_file, vocab)
    test_data = read_corpus(config.test_file, vocab)

    start_a = time.time()
    train_feats, train_actions = get_gold_actions(train_data, vocab)
    print("Get Action Time: ", time.time() - start_a)

    assert len(train_data) == len(train_actions)

    vocab.create_action_table(train_actions)
    start_a = time.time()
    train_candid = get_gold_candid(train_data, vocab)
    print("Get Candidates Time: ", time.time() - start_a)

    train_insts = inst(train_data, train_actions, train_feats, train_candid)

    encoder = Encoder(vocab, config, vec)
    decoder = Decoder(vocab, config)

    if config.use_cuda:
        torch.backends.cudnn.enabled = True
        #torch.backends.cudnn.benchmark = True
        encoder = encoder.cuda()
        decoder = decoder.cuda()
    parser = TransitionBasedParser(encoder, decoder, vocab.ROOT, config, vocab.ac_size)
    train(train_insts, dev_data, test_data, parser, vocab, config)
