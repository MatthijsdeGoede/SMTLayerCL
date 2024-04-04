import itertools
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import z3
from torch.utils.data import TensorDataset
from tqdm import tqdm
from datetime import datetime

from scorer import VisualAdditionScorer
from smtlayer import SMTLayer

# Download CUDA: https://pytorch.org/get-started/locally/
print(f"Cuda available: {torch.cuda.is_available()}")


class MNISTAddition(torch.utils.data.Dataset):
    def __init__(self, x_by_classes, label_pairs):
        super(MNISTAddition).__init__()

        self.by_classes = x_by_classes
        self.label_pairs = label_pairs

        self.len = sum([len(x_by_classes[cl]) for cl in x_by_classes])

        self.bin_target = np.vectorize(
            lambda x: np.array(list(np.binary_repr(x, 5)), dtype=np.float32),
            signature="()->({})".format(5))

        self.bin_symbolic = np.vectorize(
            lambda x: np.array(list(np.binary_repr(x[0], 4)) + list(np.binary_repr(x[1], 4)), dtype=np.float32),
            signature="(m)->(n)"
)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        pair = random.choice(self.label_pairs)
        inst1 = random.choice(self.by_classes[pair[0]]).unsqueeze(0)
        inst2 = random.choice(self.by_classes[pair[1]]).unsqueeze(0)

        return (inst1, inst2), torch.tensor(self.bin_target(sum(pair))).float(), torch.tensor(self.bin_symbolic(pair)).float()


class MNISTExtractor(nn.Module):
    def __init__(self, n_feats):
        super(MNISTExtractor, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, 2)
        self.conv2 = nn.Conv2d(64, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1)
        self.fc1 = nn.Linear(6272, 256)
        self.fc2 = nn.Linear(256, n_feats)

        nn.init.orthogonal_(self.conv1.weight)
        nn.init.orthogonal_(self.conv2.weight)
        nn.init.orthogonal_(self.conv3.weight)
        nn.init.orthogonal_(self.conv4.weight)
        nn.init.orthogonal_(self.fc1.weight)
        nn.init.orthogonal_(self.fc2.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)
        nn.init.zeros_(self.conv3.bias)
        nn.init.zeros_(self.conv4.bias)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)

        return out


class MNISTAdder(nn.Module):
    def __init__(self, use_maxsmt=False):
        super(MNISTAdder, self).__init__()
        self.extractor = MNISTExtractor(4)
        self.fc1 = nn.Linear(8, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 5)

        inputs = z3.Bools('v0 v1 v2 v3 v4 v5 v6 v7')  # 8 input booleans, 4 per digit
        outputs = z3.Bools('v8 v9 v10 v11 v12')  # 5 output booleans
        x0, x1, x2, x3, x4, x5, x6, x7 = inputs
        y0, y1, y2, y3, y4 = outputs
        z1, z2, y = z3.BitVecs('z1 z2 y', 5)

        cl9 = z1 == z3.Concat(z3.BitVecVal(0, 1),
                              z3.If(x0, z3.BitVecVal(1, 1), z3.BitVecVal(0, 1)),
                              z3.If(x1, z3.BitVecVal(1, 1), z3.BitVecVal(0, 1)),
                              z3.If(x2, z3.BitVecVal(1, 1), z3.BitVecVal(0, 1)),
                              z3.If(x3, z3.BitVecVal(1, 1), z3.BitVecVal(0, 1)))
        cl10 = z2 == z3.Concat(z3.BitVecVal(0, 1),
                               z3.If(x4, z3.BitVecVal(1, 1), z3.BitVecVal(0, 1)),
                               z3.If(x5, z3.BitVecVal(1, 1), z3.BitVecVal(0, 1)),
                               z3.If(x6, z3.BitVecVal(1, 1), z3.BitVecVal(0, 1)),
                               z3.If(x7, z3.BitVecVal(1, 1), z3.BitVecVal(0, 1)))
        cl11 = y == z1 + z2
        cl12 = y4 == (z3.Extract(0, 0, y) == z3.BitVecVal(1, 1))
        cl13 = y3 == (z3.Extract(1, 1, y) == z3.BitVecVal(1, 1))
        cl14 = y2 == (z3.Extract(2, 2, y) == z3.BitVecVal(1, 1))
        cl15 = y1 == (z3.Extract(3, 3, y) == z3.BitVecVal(1, 1))
        cl16 = y0 == (z3.Extract(4, 4, y) == z3.BitVecVal(1, 1))

        clauses = [cl9, cl10, cl11, cl12, cl13, cl14, cl15, cl16]
        mask = torch.tensor(
            [1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0.])  # to mask which are input and output vars

        self.sat = SMTLayer(
            input_size=8,
            output_size=5,
            variables=inputs + outputs,
            theory=clauses,
            default_mask=mask,
            solverop='smt' if not use_maxsmt else 'maxsmt')

    def forward(self, x, return_sat=True, return_feats=False, do_maxsat=False):
        out1 = self.extractor(x[0])
        out2 = self.extractor(x[1])
        combined = torch.cat([out1, out2], dim=1)
        symbolic_rep = torch.empty((1, 1))

        if return_feats:
            return combined
        else:
            if return_sat:
                pads = torch.zeros((x[0].shape[0], 5), dtype=out1.dtype, device=out1.device)
                combined = torch.cat([combined, pads], dim=1)
                out, symbolic_rep = self.sat(combined, do_maxsat_forward=do_maxsat)
            else:
                out = F.relu(self.fc1(combined))
                out = F.relu(self.fc2(out))
                out = self.fc3(out)

            return out, symbolic_rep


def get_by_class(
        data_dir='/data/data',
        data_fraction=1.,
):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    mnist_train = torchvision.datasets.MNIST(
        data_dir,
        train=True,
        download=True,
        transform=transform
    )
    mnist_test = torchvision.datasets.MNIST(
        data_dir,
        train=False,
        download=True,
        transform=transform
    )

    x_train, y_train = mnist_train.data / 255., mnist_train.targets
    x_test, y_test = mnist_test.data / 255., mnist_test.targets

    x_tr_by_classes = {}
    x_te_by_classes = {}

    for cl in range(10):
        train_class = x_train[y_train == cl]
        x_tr_by_classes[cl] = train_class[:int(len(train_class) * data_fraction)]
        test_class = x_test[y_test == cl]
        x_te_by_classes[cl] = test_class[:int(len(test_class) * data_fraction)]

    return x_tr_by_classes, x_te_by_classes


def get_dataloader(
        train_label_pairs,
        test_label_pairs,
        batch_size=128,
        data_fraction=1.,
):
    x_tr_by_classes, x_te_by_classes = get_by_class(data_fraction=data_fraction)

    train_data = MNISTAddition(x_tr_by_classes, train_label_pairs)
    test_data = MNISTAddition(x_te_by_classes, test_label_pairs)

    train_load = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    test_load = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    return train_load, test_load


def get_curriculum_dataloader(
        train_label_triplets,
        epochs,
        use_curriculum,
        batch_size=128,
        data_fraction=1.,
):
    x_tr_by_classes, _ = get_by_class(data_fraction=data_fraction)

    train_loaders = []

    # Sort the triplets ascending based on the symbolic uncertainty score of the output label
    sorted_train_triplets = sorted(train_label_triplets, key=lambda x: x[-1])
    N = len(sorted_train_triplets)
    delta = int(np.ceil(N / epochs))

    # Build the curriculum criteria
    for t in range(epochs):
        if use_curriculum:
            criteria_pairs = [(x[0], x[1]) for x in sorted_train_triplets[:min((t + 1) * delta, N)]]
        else:
            # when curriculum learning is turned off, return the same data loader for each epoch
            criteria_pairs = [(x[0], x[1]) for x in sorted_train_triplets]

        criteria_data = MNISTAddition(x_tr_by_classes, criteria_pairs)
        criteria_load = torch.utils.data.DataLoader(criteria_data, batch_size=batch_size)
        train_loaders.append((len(criteria_pairs), criteria_load))

    return train_loaders


def train_epoch(epoch_idx, model, optimizer, train_load, use_satlayer=True,
                clip_norm=None, sched=None, do_maxsat_forward=False):

    tloader = tqdm(enumerate(train_load), total=len(train_load))
    acc_total = 0.
    sym_acc_total = 0.
    loss_total = 0.
    total_samp = 0.

    model.train()

    for batch_idx, (data, target, symbolic_truth) in tloader:

        (data1, data2), target, symbolic_truth = data, target.cuda(), symbolic_truth.cuda()
        data1, data2 = data1.cuda(), data2.cuda()
        data = (data1, data2)

        optimizer.zero_grad()
        output, symbolic_rep = model(data, return_sat=use_satlayer, do_maxsat=do_maxsat_forward)
        loss = F.binary_cross_entropy_with_logits(output, target)
        loss.backward()
        if clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()

        if sched is not None:
            sched.step()

        with torch.no_grad():
            acc = torch.sum((torch.all(torch.sign(output) == 2 * (target - 0.5), dim=1)).type(torch.FloatTensor))
            symbolic_acc = torch.sum((torch.all(torch.sign(symbolic_rep) == 2 * (symbolic_truth - 0.5), dim=1)).type(torch.FloatTensor))

        acc_total += acc.item()
        sym_acc_total += symbolic_acc.item()
        loss_total += loss.item()
        total_samp += float(len(data1))

        tloader.set_description('train {} loss={:.4} output_acc={:.4} symbolic_acc={:.4}, lr={:.4}'.format(epoch_idx,
                                                                                loss_total / (batch_idx + 1),
                                                                                acc_total / total_samp,
                                                                                sym_acc_total / total_samp,
                                                                                optimizer.param_groups[0]['lr']))

    train_acc = acc_total / total_samp
    train_sym_acc = sym_acc_total / total_samp
    train_loss = loss_total / (1 + len(train_load))

    return train_acc, train_sym_acc, train_loss, tloader.format_dict['elapsed']


def test_epoch(epoch_idx, model, test_load, use_satlayer=True):
    tloader = tqdm(enumerate(test_load), total=len(test_load))
    acc_total = 0.
    sym_acc_total = 0.
    loss_total = 0.
    total_samp = 0.

    model.eval()

    for batch_idx, (data, target, symbolic_truth) in tloader:
        with torch.no_grad():

            (data1, data2), target, symbolic_truth = data, target.cuda(), symbolic_truth.cuda()
            data1, data2 = data1.cuda(), data2.cuda()
            data = (data1, data2)

            output, symbolic_rep = model(data, return_sat=use_satlayer, do_maxsat=True)

            loss = F.binary_cross_entropy_with_logits(output, target)
            acc = torch.sum((torch.all(torch.sign(output) == 2 * (target - 0.5), dim=1)).type(torch.FloatTensor))
            symbolic_acc = torch.sum((torch.all(torch.sign(symbolic_rep) == 2 * (symbolic_truth - 0.5), dim=1)).type(torch.FloatTensor))

            label = torch.argmax(output, dim=1)

            acc_total += acc.item()
            sym_acc_total += symbolic_acc.item()
            loss_total += loss.item()
            total_samp += float(len(data1))

            tloader.set_description('test {} loss={:.4} acc={:.4} symbolic_acc={:.4}'.format(epoch_idx,
                                                                          loss_total / (batch_idx + 1),
                                                                          acc_total / total_samp,
                                                                          sym_acc_total / total_samp))

    test_acc = acc_total / total_samp
    test_sym_acc = sym_acc_total / total_samp
    test_loss = loss_total / (1 + len(test_load))

    return test_acc, test_sym_acc, test_loss


def pretrain(model, optimizer, train_load, epochs, clip_norm=None):
    for epoch in range(1, epochs + 1):

        tloader = tqdm(enumerate(train_load), total=len(train_load))
        acc_total = 0.
        loss_total = 0.
        total_samp = 0.

        model.train()

        for batch_idx, (data, target) in tloader:

            (data1, data2), target = data, target.cuda()
            data1, data2 = data1.cuda(), data2.cuda()
            data = (data1, data2)

            optimizer.zero_grad()
            output = model(data, return_sat=False)
            loss = F.binary_cross_entropy_with_logits(output, target)
            loss.backward()
            if clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()

            with torch.no_grad():
                acc = torch.sum((torch.all(torch.sign(output) == 2 * (target - 0.5), dim=1)).type(torch.FloatTensor))

            acc_total += acc.item()
            loss_total += loss.item()
            total_samp += float(len(data1))

            tloader.set_description('pretrain {} loss={:.4} acc={:.4}'.format(epoch,
                                                                              loss_total / (batch_idx + 1),
                                                                              acc_total / total_samp))


def train(model, optimizer, train_curriculum_load, test_load, epochs, trial,
          use_satlayer=True, clip_norm=None, sched=None, do_sched_batch=False,
          do_maxsat_forward=None):
    times = []

    trial_data = {
        "trial": trial,
        "epoch": [i for i in range(1, epochs + 1)],
        "train_acc": [],
        "train_sym_acc": [],
        "test_acc": [],
        "test_sym_acc": [],
        "pairs": [j[0] for j in train_curriculum_load],
        "samples": [sum(len(train_curriculum_load[j][1]) * 128 for j in range(0, i+1)) for i in range(len(train_curriculum_load))],
        "duration": [],
    }

    for epoch in range(1, epochs + 1):

        if do_sched_batch:
            train_acc, train_sym_acc, train_loss, elapsed = train_epoch(epoch, model, optimizer, train_curriculum_load[epoch-1][1],
                                                         use_satlayer=use_satlayer, clip_norm=clip_norm, sched=sched,
                                                         do_maxsat_forward=do_maxsat_forward)
        else:
            train_acc, train_sym_acc, train_loss, elapsed = train_epoch(epoch, model, optimizer, train_curriculum_load[epoch-1][1],
                                                         use_satlayer=use_satlayer, clip_norm=clip_norm, sched=None,
                                                         do_maxsat_forward=do_maxsat_forward)
            if sched is not None:
                sched.step()
        times.append(elapsed)

        trial_data["train_acc"].append(train_acc)
        trial_data["train_sym_acc"].append(train_sym_acc)
        trial_data["duration"].append(elapsed)

        test_acc, test_sym_acc, test_loss = test_epoch(epoch, model, test_load, use_satlayer=use_satlayer)

        trial_data["test_acc"].append(test_acc)
        trial_data["test_sym_acc"].append(test_sym_acc)

        if train_acc > 0.999:
            break

    return train_acc, train_sym_acc, test_sym_acc, test_acc, sum(times) / float(epochs), pd.DataFrame(trial_data)

def run(
        lr=1.,
        pretrain_epochs=0,
        pct=100,
        epochs=10,
        data_fraction=0.1,
        batch_size=128,
        trials=1,
        clip_norm=0.1,
        maxsat_forward=False,
        maxsat_backward=False,
        use_curriculum=True,
):
    test_label_pairs = list(itertools.product(list(range(0, 10)), repeat=2))
    if pct <= 10:
        train_label_pairs = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9)]
    else:
        train_label_pairs = test_label_pairs[:int(pct)]

    # Augment the label_pairs with the symbolic uncertainty scores
    scorer = VisualAdditionScorer()
    s1_dom = {s1 for s1 in range(10)}
    s2_dom = {s2 for s2 in range(10)}
    y_dom = {y for y in range(19)}
    scores = scorer.score(s1_dom, s2_dom, y_dom)
    train_label_triples = [(a, b, scores[a + b]) for (a, b) in train_label_pairs]

    print(f"Starting experiment with trials={trials}, epochs={epochs}, data_fraction={data_fraction} and use_curriculum={use_curriculum}")

    # Take the number of epochs to be the number of curriculum criteria, create a custom data loader for each step
    train_curriculum_load = get_curriculum_dataloader(train_label_triples, epochs, use_curriculum, batch_size=batch_size, data_fraction=data_fraction)

    _, test_load = get_dataloader(train_label_pairs, test_label_pairs, batch_size=batch_size)
    pretrain_load, _ = get_dataloader(train_label_pairs, test_label_pairs, batch_size=512)

    train_accs = []
    train_sym_accs = []
    test_accs = []
    test_sym_accs = []
    times = []
    trial_dfs = []

    for i in range(trials):
        model = MNISTAdder(use_maxsmt=maxsat_backward).cuda()
        optimizer = optim.SGD([{'params': model.parameters(), 'lr': lr, 'momentum': 0.9, 'nesterov': True}])
        sched = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                    lr,
                                                    epochs=epochs,
                                                    steps_per_epoch=len(train_curriculum_load[0][1]),
                                                    pct_start=1. / float(epochs))

        pretrain(model, optimizer, pretrain_load, pretrain_epochs, clip_norm=clip_norm)

        train_acc, train_sym_acc, test_sym_acc, test_acc, elapsed, trial_df = train(
            model, optimizer, train_curriculum_load, test_load, epochs, i+1,
            clip_norm=clip_norm, sched=sched, do_sched_batch=True,
            do_maxsat_forward=maxsat_forward
        )

        train_accs.append(train_acc)
        train_sym_accs.append(train_sym_acc)
        test_accs.append(test_acc)
        test_sym_accs.append(test_sym_acc)
        times.append(elapsed)
        trial_dfs.append(trial_df)

        print('\n[trial {} result]: train={:.4},{:.4}, test={:.4},{:.4} time={:.4}\n'.format(i+1, train_acc, train_sym_acc, test_acc, test_sym_acc, elapsed))
        print('-' * 20)

    result_df = pd.concat(trial_dfs, ignore_index=True)
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    result_file = f"../results/{now}_{int(data_fraction*100)}_{epochs}{'_curriculum' if use_curriculum else ''}.csv"
    result_df.to_csv(result_file, index=False)
    print(f"Results output written to {result_file}")

    train_accs = np.array(train_accs)
    train_sym_accs = np.array(train_sym_accs)
    test_accs = np.array(test_accs)
    test_sym_accs = np.array(test_sym_accs)
    times = np.array(times)

    print('\nsmt pct={} stats: train={:.4} ({:.8}), {:.4} ({:.8}) test={:.4} ({:.8}), {:.4} ({:.8}) time={:.4} ({:.8})'.format(
        pct, train_accs.mean(), train_accs.std(), train_sym_accs.mean(), train_sym_accs.std(), test_accs.mean(),
        test_accs.std(), test_sym_accs.mean(), test_sym_accs.std(), times.mean(), times.std()))


if __name__ == '__main__':
    run()
