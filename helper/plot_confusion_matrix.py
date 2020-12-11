import itertools
import matplotlib.pyplot as plt
import numpy as np
from model import Res15
from test import TestProcessor
import os


plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Tahoma', 'DejaVu Sans',
                                   'Lucida Grande', 'Verdana']


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    model = Res15(n_labels=12).cuda()
    model.load('../models_pt/res15/model_32000.pt')
    test_processor = TestProcessor(model, model_type=" ")
    _, class_names, cnf_matrix = test_processor.compute_embedding_accuracy('/users/liuli/project/kws_end2end/csv_linux/enroll_customized.csv',
                                                                 '/users/liuli/project/kws_end2end/csv_linux/test_customized.csv',
                                                                           plot_confusion_matrix=True)

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=False, title='Confusion matrix')

    # plt.get_backend()
