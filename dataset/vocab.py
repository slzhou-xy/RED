import pickle
import pandas as pd
import numpy as np


class WordVocab:
    def __init__(self, traj_path, roadnetwork_path, use_mask=False, use_sep=False, use_start=False, use_extract=False):
        self.pad_index = 0
        specials = ["<pad>"]
        if use_mask:
            self.mask_index = len(specials)
            specials = specials + ["<mask>"]

        if use_sep:  # this is not used in our model, but we still add it into vocab
            self.sep_index = len(specials)
            specials = specials + ["<sep>"]

        if use_start:
            self.start_index = len(specials)
            specials = specials + ["<start>"]

        if use_extract:
            self.extract_index = len(specials)
            specials = specials + ["<extract>"]

        self.specials_length = len(specials)

        roadnetwork = pd.read_csv(roadnetwork_path, sep=',')
        traj = pd.read_csv(traj_path, sep=',')

        self.index2loc = list(specials)

        index = list(roadnetwork.index.values)
        self.index2loc = list(specials) + index
        self.loc2index = {tok: i for i, tok in enumerate(self.index2loc)}
        self.vocab_size = len(self.index2loc)

        users = traj['taxi_id'].values
        users = np.unique(users)
        self.user_num = len(users) + 1

    def to_seq(self, sentence, seq_len=None, with_eos=False, with_sos=False, with_len=False):
        if isinstance(sentence, str):
            sentence = sentence.split()

        seq = [self.loc2index.get(word) for word in sentence]

        if with_sos:
            seq = [self.sos_index] + seq

        origin_seq_len = len(seq)

        if seq_len is None:
            pass
        elif len(seq) <= seq_len:
            seq += [self.pad_index for _ in range(seq_len - len(seq))]
        else:
            seq = seq[:seq_len]

        return (seq, origin_seq_len) if with_len else seq

    def from_seq(self, seq, join=False, with_pad=False):
        words = [self.index2loc[idx]
                 if idx < len(self.index2loc)
                 else "<%d>" % idx
                 for idx in seq
                 if not with_pad or idx != self.pad_index]

        return " ".join(words) if join else words

    @staticmethod
    def load_vocab(vocab_path: str) -> 'WordVocab':
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

    def save_vocab(self, vocab_path):
        with open(vocab_path, "wb") as f:
            pickle.dump(self, f)

    def __eq__(self, other):
        if self.loc2index != other.loc2index:
            return False
        if self.index2loc != other.index2loc:
            return False
        return True

    def __len__(self):
        return len(self.index2loc)

    def vocab_rerank(self):
        self.loc2index = {word: i for i, word in enumerate(self.index2loc)}

    def extend(self, v, sort=False):
        words = sorted(v.index2loc) if sort else v.index2loc
        for w in words:
            if w not in self.loc2index:
                self.index2loc.append(w)
                self.loc2index[w] = len(self.index2loc) - 1
