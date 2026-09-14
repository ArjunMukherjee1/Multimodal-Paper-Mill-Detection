import json
import pickle

import numpy as np

from train_models import (load_papers, by_split, select_threshold, image_scores,
                           score_papers, DEVICE)
from transformers import BertForSequenceClassification, BertTokenizerFast
import open_clip


def metrics(scores, labels_by_doi, dois, threshold):
    tp = fp = tn = fn = 0

    for doi in dois:
        if doi not in scores:
            continue

        pred = scores[doi] >= threshold
        actual = labels_by_doi[doi] == "paper_mill"

        if pred and actual:
            tp += 1
        elif pred and not actual:
            fp += 1
        elif not pred and actual:
            fn += 1
        else:
            tn += 1


    n = tp + fp + tn + fn
    acc = (tp + tn) / n if n else 0
    sens = tp / (tp + fn) if (tp + fn) else 0
    spec = tn / (tn + fp) if (tn + fp) else 0

    return n, acc, sens, spec


def report(name, modality, scores, labels_by_doi, papers):
    opt_dois = [p["doi"] for p in by_split(papers, modality, "optimization")]
    threshold = select_threshold({d: scores[d] for d in opt_dois if d in scores}, labels_by_doi)

    for split in ["internal_validation", "external_validation"]:

        dois = [p["doi"] for p in by_split(papers, modality, split)]
        n, acc, sens, spec = metrics(scores, labels_by_doi, dois, threshold)

        print(f"{name:20s} {split:20s} n={n:5d} acc={acc*100:.2f}% sens={sens*100:.2f}% spec={spec*100:.2f}%")


def main():
    papers =load_papers()
    labels_by_doi ={p["doi"]: p["label"] for p in papers}

    with open("scores.json") as f:
        stored =json.load(f)

    report("abstract-only","text", stored["abstract"], labels_by_doi, papers)
    report("full-text","text", stored["fulltext"], labels_by_doi, papers)
    report("image","image",stored["image"], labels_by_doi, papers)

    with open("fusion_abstract.pkl","rb") as f:
        fusion_abstract =pickle.load(f)
    with open("fusion_fulltext.pkl","rb") as f:
        fusion_fulltext =pickle.load(f)

    fusion_abstract_scores,fusion_fulltext_scores = {},{}

    for p in papers:
        doi = p["doi"]

        if doi in stored["abstract"] and doi in stored["image"]:
            x = [[stored["abstract"][doi], p["citation_rate"], stored["image"][doi]]]
            fusion_abstract_scores[doi] =fusion_abstract.predict_proba(np.array(x))[0, 1]

        if doi in stored["fulltext"] and doi in stored["image"]:
            x = [[stored["fulltext"][doi], p["citation_rate"], stored["image"][doi]]]
            fusion_fulltext_scores[doi] =fusion_fulltext.predict_proba(np.array(x))[0, 1]

    report("fusion-abstract", "fusion", fusion_abstract_scores,labels_by_doi,papers)
    report("fusion-fulltext", "fusion", fusion_fulltext_scores,labels_by_doi,papers)


if __name__ == "__main__":
    main()
