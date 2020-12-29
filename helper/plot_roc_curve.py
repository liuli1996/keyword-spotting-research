import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from model import TradFPool3
from test import TestProcessor
import numpy as np

def compute_fpr_fnr(model_name):
    scores_file = "../roc_data/" + model_name + "_scores.npy"
    labels_file = "../roc_data/" + model_name + "_labels.npy"
    # scores_file = "../roc_data/TIMIT_" + model_name + "_scores.npy"
    # labels_file = "../roc_data/TIMIT_" + model_name + "_labels.npy"

    # Load scores and labels
    scores = np.load(scores_file)
    labels = np.load(labels_file)

    # Compute ROC curve and ROC area for each class
    fpr, tpr, threshold = roc_curve(labels, scores)  # 计算真正率和假正率
    roc_auc = auc(fpr, tpr)  # 计算auc的值
    fnr = 1 - tpr
    print("model: {}, best threshold: {}".format(model_name, threshold[np.argmin(abs(fnr - fpr))]))
    return fpr, fnr


if __name__ == '__main__':
    # save_data()

    # plot
    fpr_1, fnr_1 = compute_fpr_fnr("res15_fulllabel_finetune13layer_customized_only")
    fpr_2, fnr_2 = compute_fpr_fnr("tradfpool3_fulllabel_finetune4layer_customized_only")
    fpr_3, fnr_3 = compute_fpr_fnr("tradfpool3")
    fpr_4, fnr_4 = compute_fpr_fnr("res15")
    fpr_5, fnr_5 = compute_fpr_fnr("lstm")
    fpr_6, fnr_6 = compute_fpr_fnr("AmResNet")
    # fpr_7, fnr_7 = compute_fpr_fnr("e2e_asr_free")

    font = {'size': 16}

    plt.figure()
    lw = 2
    plt.figure(figsize=(7, 7), dpi=300)
    plt.plot(fpr_1, fnr_1, color='royalblue', lw=lw, label="res15 with proposed techniques")
    plt.plot(fpr_4, fnr_4, color='royalblue', lw=lw, ls='--', label="res15")
    plt.plot(fpr_2, fnr_2, color='red', lw=lw, label="cnn-trad-fpool3 with proposed techniques")
    plt.plot(fpr_3, fnr_3, color='red', lw=lw, ls='--', label="cnn-trad-fpool3")
    plt.plot(fpr_5, fnr_5, color='darkorange', lw=lw, label="LSTM")
    plt.plot(fpr_6, fnr_6, color='darkgreen', lw=lw, label="AC-ResNet")
    # plt.plot(fpr_7, fnr_7, color='darkviolet', lw=lw, label="E2E_ASR_free")

    plt.legend()
    frame = plt.legend().get_frame()
    frame.set_alpha(1)
    frame.set_facecolor('none')  # 设置图例legend背景透明

    plt.xlim([0.0, 0.20])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Alarm Rate')
    plt.ylabel('False Rejects Rate')
    plt.title('ROC')
    plt.show()
    # plt.savefig('./test1.png', transparent=True)