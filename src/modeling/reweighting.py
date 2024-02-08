import numpy as np
import torch as ch
from torch.cuda.amp import autocast
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from .models import get_feature_extractor, update_classifier


def get_features_and_labels(model, loader):
    feature_extractor = get_feature_extractor(model, copy=True)
    feature_extractor.eval()
    features = []
    labels = []
    for x, y in loader:
        with ch.no_grad(), autocast():
            features.append(feature_extractor(x).cpu().numpy())
        labels.append(y.cpu().numpy())
    features = np.concatenate(features)
    labels = np.concatenate(labels)
    return features, labels, feature_extractor


def reweight(model, loader, groups, relevant_groups, penalty="l2", c=0.001, verbose=False):
    features, labels, _ = get_features_and_labels(model, loader)
    
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    train_mask = np.isin(groups, relevant_groups)
    train_groups = groups[train_mask]
    sample_weights = np.zeros(len(train_groups))
    for group in relevant_groups:
        count = (train_groups == group).sum()
        sample_weights[train_groups==group] = len(train_groups) / (len(relevant_groups) * count)
    classifier = LogisticRegression(penalty=penalty, C=c, solver="lbfgs", random_state=0, verbose=verbose, max_iter=1_000)
    classifier.fit(features[train_mask], labels[train_mask], sample_weight=sample_weights)
    weight_np = classifier.coef_ / scaler.scale_
    bias_np = classifier.intercept_ - weight_np @ scaler.mean_

    weight = ch.from_numpy(weight_np).type(ch.float32).cuda()
    bias = ch.from_numpy(bias_np).type(ch.float32).cuda()
    update_classifier(model, weight, bias)