from model import Res15
from dataset import TrainDataset
from loss import GE2ELoss
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os
from sklearn.metrics import roc_curve, auc

def incremental_training():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    pretrained_model_file = "trained_models/res15_fine_grained.pt"
    fine_tuned_model_file = "trained_models/res15_fine_grained_fine_tuned.pt"
    num_utter = 5
    num_word = 5
    n_batches = 200
    epoches = 3

    model = Res15(n_labels=26)
    model.load(pretrained_model_file)
    model = model.to(device)
    # print(model)
    for index, (name, para) in enumerate(model.named_parameters()):
        if index in [14, 15]:
            para.requires_grad = True
        else:
            para.requires_grad = False
        print("{} Layer name: {}, Shape: {}, Requires_grad: {}".format(index, name, para.shape, para.requires_grad))

    # optimizer
    step = 0
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    criterion = GE2ELoss(device)

    # dataset
    train_dataset = TrainDataset("fine_tune_append.txt", "features/train", num_utter, num_word, n_batches)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=2)

    for epoch in range(epoches):
        model.train()
        for data in train_loader:
            step += 1
            optimizer.zero_grad()
            data = data.squeeze(0)  # [1, num_utter * num_word, 101, 40] -> [num_utter * num_word, 101, 40]
            data = data.to(device)
            _, out = model(data)
            out = out.reshape(num_word, num_utter, -1)  # [10, 10, 12]
            loss = criterion(out)
            loss.backward()
            optimizer.step()
            print("step #{} loss: {}".format(step, loss))
    model.save(fine_tuned_model_file)
    print("Training Finished. Fine tuned model file are save at {}".format(fine_tuned_model_file))

def compute_similarity(txt_file):
    model = Res15(n_labels=26)
    model.load("trained_models/res15_fine_grained_fine_tuned.pt")
    model.cuda()

    model.eval()
    with open(txt_file, "r") as f:
        contents = f.readlines()
        contents = list(map(lambda x: tuple(x.strip().split(",")), contents))
    df = pd.DataFrame(contents, columns=["filename", "keyword"])
    embedding_dict = dict()
    personalized_keyword_embeddings = list()
    for keyword, sub_df in df.groupby(df["keyword"]):
        buffer = []
        for index, row in sub_df.iterrows():
            x = np.load(os.path.join("features/train", keyword, row["filename"] + ".npy"))
            with torch.no_grad():
                y, embedding = model(torch.from_numpy(x).cuda())
            buffer.append(embedding)
            if keyword == "personalized_keyword":
                personalized_keyword_embeddings = buffer
        avg_embedding = torch.mean(torch.cat(buffer, dim=0), dim=0).unsqueeze(0)
        embedding_dict[keyword] = avg_embedding
    return embedding_dict, personalized_keyword_embeddings

def compute_fpr_fnr(scores, labels):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, threshold = roc_curve(labels, scores)  # compute fpr and tpr
    roc_auc = auc(fpr, tpr)  # compute auc value
    fnr = 1 - tpr
    idx = np.argmin(abs(fnr - fpr))
    print("best threshold: {}, fnr: {:.2f}%, fpr: {:.2f}%".format(threshold[idx], fnr[idx] * 100, fpr[idx] * 100))
    return threshold[idx], fpr, fnr

if __name__ == '__main__':
    incremental_training()

    # from torch.nn.functional import cosine_similarity
    # embedding_dict, personalized_keyword_embeddings = compute_similarity("fine_tune_append.txt")
    # scores = []
    # labels = []
    # personalized_keyword_centroid = embedding_dict["personalized_keyword"]
    # for embedding in personalized_keyword_embeddings:
    #     score = cosine_similarity(personalized_keyword_centroid, embedding)
    #     scores.append(score.item())
    #     labels.append(1)
    # for keyword, embedding in embedding_dict.items():
    #     if keyword == "personalized_keyword":
    #         continue
    #     score = cosine_similarity(personalized_keyword_centroid, embedding)
    #     scores.append(score.item())
    #     labels.append(0)
    # threshold, fpr, fnr = compute_fpr_fnr(scores, labels)
    pass