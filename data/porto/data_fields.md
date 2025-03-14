taxi_id: user_id

cpath: the actual path of the taxi, which is used to calculate the loss in the decoder

mask_traj: the masked cpath by road-aware masking.

align_time: some paths of cpaths may not have timestamps (because of the sampling rate of GPS-device), this is enhanced timestamps by interpolating the timestamps of the cpath.

key_cpath: unmasked paths of cpath as inputs of the encoder

key_time: the timestamps of the key_cpath

full_dis: accumulated distance sequence of the cpath

key_dis: accumulated distance sequence of the key_cpath
