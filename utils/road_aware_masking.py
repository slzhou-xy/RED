import numpy as np
import pandas as pd
from datetime import datetime
from collections import Counter


def time_interpolation(t):
    start = 0
    end = 1

    while end != len(t):
        if t[end] != 0 and t[end - 1] == 0 and t[start] != 0:
            end_time = datetime.fromtimestamp(t[end])
            start_time = datetime.fromtimestamp(t[start])

            gap = end_time - start_time
            per_gap = gap / (end - start)
            for k in range(start + 1, end):
                t[k] = int((start_time + (k - start) * per_gap).timestamp() + 0.5)
            if end == len(t) - 1:
                break
            start = end
            end += 1
        elif t[end] == 0:
            end += 1
        else:
            start += 1
            end += 1
    return t


def get_mask_traj(traj_df, average_length, edge, average_points, edge_points):
    cpath_list = list(traj_df['cpath'].values)
    opath_list = list(traj_df['opath'].values)
    temporal_list = list(traj_df['time'].values)
    user_list = list(traj_df['taxi_id'].values)

    short_cpath_list = []
    short_temporal_list = []
    mask_cpath_list = []
    new_temporal_list = []
    new_user_list = []
    new_cpath_list = []

    for i in range(len(cpath_list)):
        cpath = eval(cpath_list[i])
        opath = eval(opath_list[i])
        temporal = eval(temporal_list[i])

        # if cpath[0] == cpath[-1]:
        #     continue
        assert len(opath) == len(temporal)

        c_idx = 0
        o_idx = 0
        short_cpath = []
        short_temporal = []
        mask_cpath = []
        new_temporal = []

        while c_idx != len(cpath) and o_idx != len(opath):
            if cpath[c_idx] != opath[o_idx]:
                mask_cpath.append(-1)
                new_temporal.append(0)
                c_idx += 1
                continue

            same_opath_list_idx = []
            while o_idx != len(opath) and cpath[c_idx] == opath[o_idx]:
                same_opath_list_idx.append(o_idx)
                o_idx += 1
            length = len(same_opath_list_idx)
            if length % 2 == 0:
                flag = int(length / 2 - 1)
            else:
                flag = int(length / 2)
            mask_cpath.append(cpath[c_idx])
            new_temporal.append(temporal[same_opath_list_idx[flag]])
            short_cpath.append(cpath[c_idx])
            short_temporal.append(temporal[same_opath_list_idx[flag]])
            c_idx += 1
        cpath = cpath[:len(new_temporal)]

        mask1 = []
        mask2 = []
        mask3 = []
        for j in range(len(cpath)):
            if mask_cpath[j] == -1:
                mask1.append(j)
            if edge.iloc[cpath[j]]['length'] < average_length:
                mask2.append(j)
            if edge_points[cpath[j]] < average_points:
                mask3.append(j)

        masks = list(set(mask1) & (set(mask2) | set(mask3)))
        masks = list(set(masks))

        mask_cpath = cpath.copy()
        for mask in masks:
            mask_cpath[mask] = -1

        new_temporal = time_interpolation(new_temporal)
        short_cpath = []
        short_temporal = []
        for j, path in enumerate(mask_cpath):
            if path != -1:
                short_cpath.append(path)
                short_temporal.append(new_temporal[j])
        assert len(short_cpath) == len(short_temporal)
        assert len(cpath) == len(new_temporal) == len(mask_cpath)

        short_cpath_list.append(short_cpath)
        short_temporal_list.append(short_temporal)
        mask_cpath_list.append(mask_cpath)
        new_temporal_list.append(new_temporal)
        new_user_list.append(user_list[i])
        new_cpath_list.append(cpath)

    count = 0
    ratio = 0.0
    for i in range(len(short_cpath_list)):
        mask_cpath = mask_cpath_list[i]
        if -1 in mask_cpath:
            count += 1
            ratio += (Counter(mask_cpath)[-1] / len(mask_cpath))
    ratio = ratio / count

    for i in range(len(short_cpath_list)):
        mask_cpath = mask_cpath_list[i]
        temporal = new_temporal_list[i]
        short_cpath = []
        short_temporal = []
        if -1 not in mask_cpath:
            scores = np.random.randn(len(mask_cpath))
            rank = list(scores.argsort(axis=0)[::-1])
            rank.remove(0)
            rank.remove(len(mask_cpath) - 1)
            end_pos = int(len(mask_cpath) * ratio)
            mask_index = rank[:end_pos]
            for idx in mask_index:
                mask_cpath[idx] = -1
            for idx, segment in enumerate(mask_cpath):
                if segment != -1:
                    short_cpath.append(segment)
                    short_temporal.append(temporal[idx])
            short_temporal_list[i] = short_temporal
            short_cpath_list[i] = short_cpath
            mask_cpath_list[i] = mask_cpath

    mapping = {user_id: i for i, user_id in enumerate(set(new_user_list))}
    new_user_list = [mapping[user_id] for user_id in new_user_list]

    new_df = pd.DataFrame(
        {
            'taxi_id': new_user_list,
            'cpath': new_cpath_list,
            'align_time': new_temporal_list,
            'key_cpath': short_cpath_list,
            'key_time': short_temporal_list,
            'mask_traj': mask_cpath_list,
        }
    )

    return new_df



def get_average_points(traj_df, edge):
    opath_list = list(traj_df['opath'].values)
    edge_points = [0] * edge.shape[0]
    for opath in opath_list:
        opath = eval(opath)
        for path in opath:
            edge_points[path] += 1
    return sum(edge_points) / edge.shape[0], edge_points


if __name__ == "__main__":
    traj_df = pd.read_csv('porto_traj_after_fmm.csv', sep=';')
    edge = pd.read_csv('porto/rn/edge.csv')
    average_length = sum(list(edge['length'])) / edge.shape[0]
    average_points, edge_points = get_average_points(traj_df, edge)
    df = get_mask_traj(traj_df, average_length, edge, average_points, edge_points)
    df.to_csv('porto_traj_after_fmm.csv', index=False)
