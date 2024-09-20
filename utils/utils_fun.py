import pandas as pd


def add_dis(traj, edge):
    cpath = list(traj.cpath)
    mask_cpath = list(traj.mask_cpath)
    full_dis_list = []
    short_dis_list = []

    for i in range(len(cpath)):
        path = eval(cpath[i])
        mask_path = eval(mask_cpath[i])

        full_dis = [0]
        short_dis = [0]
        for j in range(1, len(path)):
            prev_edge = edge.iloc[path[j - 1]]['length']
            cur_edge = edge.iloc[path[j]]['length']
            length = (prev_edge + cur_edge) / 2
            full_dis.append(full_dis[-1] + length)
            if mask_path[j] != -1:
                short_dis.append(full_dis[-1] + length)
        full_dis_list.append(str(full_dis))
        short_dis_list.append(str(short_dis))

    dis_pd = pd.DataFrame({
        'full_dis': full_dis_list,
        'key_dis': short_dis_list
    })

    traj = pd.concat([traj, dis_pd], axis=1)

    return traj
