import os
import pickle
import numpy as np
import torch
import json
from data.process_sequence import generate_data_sequence
from data.custom_dataset import VideoDataset
import pdb
from torch.utils.data import DataLoader, WeightedRandomSampler

def get_dataloader(args, shuffle_train=True, drop_last_train=True):
    # with open(os.path.join(args.database_path, 'intent_database_train.pkl'), 'rb') as fid:
    #     imdb_train = pickle.load(fid)
    # train_seq = generate_data_sequence('train', imdb_train, args)
    # with open(os.path.join(args.database_path, 'intent_database_val.pkl'), 'rb') as fid:
    #     imdb_val = pickle.load(fid)
    # val_seq = generate_data_sequence('val', imdb_val, args)
    # with open(os.path.join(args.database_path, 'intent_database_test.pkl'), 'rb') as fid:
    #     imdb_test = pickle.load(fid)
    # test_seq = generate_data_sequence('test', imdb_test, args)
    # with open('database/train_seq.pkl', 'wb') as f:
    #     pickle.dump(train_seq, f)
    # with open('database/val_seq.pkl', 'wb') as f:
    #     pickle.dump(val_seq, f)
    # with open('database/test_seq.pkl', 'wb') as f:
    #     pickle.dump(test_seq, f)
    with open('database/train_seq.pkl', 'rb') as f:
        train_seq = pickle.load(f)
    with open('database/val_seq.pkl', 'rb') as f:
        val_seq = pickle.load(f)
    with open('database/test_seq.pkl', 'rb') as f:
        test_seq = pickle.load(f)
    train_d = get_train_val_data(train_seq, args, overlap=1) # returned tracks
    val_d = get_train_val_data(val_seq, args, overlap=1)
    test_d = get_test_data(test_seq, args, overlap=1)

    # Create video dataset and dataloader
    train_dataset = VideoDataset(train_d, args)
    val_dataset = VideoDataset(val_d, args)
    test_dataset = VideoDataset(test_d, args)
    # Create a WeightedRandomSampler
    portion = 1/3
    num_samples = int(len(train_dataset) * portion)

    sampler = WeightedRandomSampler([1.35682197] * len(train_dataset), num_samples, replacement=False)


    # Use the sampler in your DataLoader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, # Set shuffle to False
                                            sampler=sampler, pin_memory=True, drop_last=drop_last_train, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False,
                                              pin_memory=True, sampler=None, drop_last=False, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False,
                                              pin_memory=True, sampler=None, drop_last=False, num_workers=4)
    return train_loader, val_loader, test_loader


def get_train_val_data(data, args, overlap=0.5):  
    seq_len = args.max_track_size
    overlap = overlap
    tracks = get_tracks(data, seq_len, args.observe_length, overlap, args)
    return tracks


def get_test_data(data, args, overlap=1): 
    # return splited train/val dataset
    seq_len = args.max_track_size
    overlap = overlap
    tracks = get_tracks(data, seq_len, args.observe_length, overlap, args)
    return tracks


def get_tracks(data, seq_len, observed_seq_len, overlap, args):
    overlap_stride = observed_seq_len if overlap == 0 else \
        int((1 - overlap) * observed_seq_len)  # default: int(0.5*15) == 7

    overlap_stride = 1 if overlap_stride < 1 else overlap_stride # when test, overlap=1, stride=1
    d_types = ['video_id', 'ped_id', 'frame', 'bbox', 'intention_binary', 'intention_prob', 'disagree_score', 'description', 'skeleton']

    d = {}

    for k in d_types:
        d[k] = data[k]

    for k in d.keys():
        tracks = []
        for track_id in range(len(d[k])):
            track = d[k][track_id]
            ''' There are some sequences not adjacent '''
            frame_list = data['frame'][track_id]
            if len(frame_list) < args.max_track_size: #60:
                print('too few frames: ', d['video_id'][track_id][0], d['ped_id'][track_id][0])
                continue
            splits = []
            start = -1
            for fid in range(len(frame_list) - 1):
                if start == -1:
                    start = fid  # frame_list[f]
                if frame_list[fid] + 1 == frame_list[fid + 1]:
                    if fid + 1 == len(frame_list) - 1:
                        splits.append([start, fid + 1])
                    continue
                else:
                    # current f is the end of current piece
                    splits.append([start, fid])
                    start = -1
            if len(splits) != 1:
                print('NOT one missing split found: ', splits)
                raise Exception()
            else: # len(splits) == 1, No missing frames from the database.
                pass
            sub_tracks = []
            for spl in splits:
                # explain the boundary:  end_idx - (15-1=14 gap) + cover last idx
                for i in range(spl[0], spl[1] - (seq_len - 1) + 1, overlap_stride):
                    sub_tracks.append(track[i:i + seq_len])
            tracks.extend(sub_tracks)
        d[k] = np.array(tracks)
    return d
