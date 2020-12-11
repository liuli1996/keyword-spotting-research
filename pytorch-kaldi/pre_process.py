import torch
from data_io import load_dataset, read_lab_fea_refac01
import kaldiio
import os
import numpy as np

def check_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

if __name__ == '__main__':
    feat_opts = "apply-cmvn --utt2spk=ark:/users/liuli/project/kaldi/egs/timit/s5/data/GSC_enroll_customized/utt2spk ark:/users/liuli/project/kaldi/egs/timit/s5/GSC_fbank/cmvn_GSC_enroll_customized.ark ark:- ark:- |"
    scp_file = "/users/liuli/project/kaldi/egs/timit/s5/data/GSC_enroll_customized/feats.scp"
    output_dir = "/users/liuli/database/features/GSC_V2/win25ms_hop10ms_41fbank_cmvn/enroll_customized"

    ark_file = "ark:copy-feats scp:" + scp_file + " ark:- | " + feat_opts
    idx = 0
    with kaldiio.ReadHelper(ark_file) as reader:
        for key, numpy_array in reader:
            idx += 1
            label = key.split("-")[0]
            file_name = key.split("-")[1]
            save_dir = os.path.join(output_dir, label)
            check_dir(save_dir)
            save_path = os.path.join(save_dir, file_name + ".npy")
            np.save(save_path, numpy_array)
            if idx % 1000 == 0:
                print("{}/{} files finished".format(idx, 3600))
