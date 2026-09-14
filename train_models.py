import json
import pickle
import random
import re

import numpy as np
import open_clip

import torch
from PIL import Image


from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, Dataset
from transformers import BertForSequenceClassification, BertTokenizerFast

device = "mps" if torch.backends.mps.is_available() else "cpu"
max_len = 64
batchSize = 32
epochs = 3

TrainingSplit = {"train": 0.70, "optimization": 0.175, "internal_validation": 0.125}
SplitSeeds = {"text": 1, "image": 2, "fusion": 3}


def load_papers(path="built_dataset.jsonl"):
    with open(path) as f:
        return [json.loads(l) for l in f]



def split_internal(papers, modality):

    key= f"{modality}_split"
    pool = [p for p in papers if p.get(key) == "internal"]
    rng = random.Random(SplitSeeds[modality])
    rng.shuffle(pool)

    n =len(pool)
    n_train = round(n * TrainingSplit["train"])
    n_opt = round(n * TrainingSplit["optimization"])
    return {
        "train": pool[:n_train],
        "optimization": pool[n_train:n_train + n_opt],
        "internal_validation": pool[n_train + n_opt:],
    }


def by_split(papers, modality, split):

    if split == "external_validation":
        key= f"{modality}_split"
        return [p for p in papers if p.get(key) == "external"]
    return split_internal(papers, modality)[split]


def sentences_for(paper, use_fulltext):
    text = paper["abstract"] or ""

    if use_fulltext and paper.get("fulltext"):
        text+= " " + paper["fulltext"]
    return [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


class SentenceDataset(Dataset):
    def __init__(self, papers, tokenizer, use_fulltext):
        self.rows = []
        for p in papers:
            label = 1 if p["label"] == "paper_mill" else 0
            for s in sentences_for(p, use_fulltext):
                self.rows.append((s, label, p["doi"]))
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        text, label, doi = self.rows[i]
        enc = self.tokenizer(text, truncation=True, max_length=max_len, padding="max_length",
                              return_tensors="pt")
        return enc["input_ids"][0], enc["attention_mask"][0], label, doi


def train_bert(papers, use_fulltext):
    tokenizer= BertTokenizerFast.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(device)
    ds =SentenceDataset(by_split(papers, "text", "train"), tokenizer, use_fulltext)
    loader = DataLoader(ds, batch_size=batchSize, shuffle=True,
                         collate_fn=lambda b: (torch.stack([x[0] for x in b]),
                                                torch.stack([x[1] for x in b]),
                                                torch.tensor([x[2] for x in b])))
    
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
    model.train()


    for epoch in range(epochs):
        for input_ids, attention_mask, labels in loader:
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            opt.zero_grad()
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            out.loss.backward()
            opt.step()
        print("epoch "+ str(epoch+1) + "/" + str(epochs)+ " finished")

    return model, tokenizer


def score_papers(model, tokenizer, papers, use_fulltext):

    model.eval()
    ds = SentenceDataset(papers, tokenizer, use_fulltext)
    loader = DataLoader(ds, batch_size=64,
                         collate_fn=lambda b: (torch.stack([x[0] for x in b]),
                                                torch.stack([x[1] for x in b]),
                                                [x[3] for x in b]))
    sums, counts = {}, {}

    with torch.no_grad():
        for input_ids, attention_mask, dois in loader:
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            probs = torch.softmax(model(input_ids=input_ids, attention_mask=attention_mask).logits, dim=-1)[:, 1]
            for doi, p in zip(dois, probs.cpu().tolist()):
                sums[doi] = sums.get(doi, 0.0) + p
                counts[doi] = counts.get(doi, 0) + 1
    return {doi: sums[doi]/counts[doi] for doi in sums}

#ensures specificity greater than min_specificity
def select_threshold(scores, labels_by_doi, min_specificity=0.95):
    n_ctrl = sum(1 for doi in scores if labels_by_doi[doi] == "control")
    if n_ctrl == 0:
        return 0.5
    best = None

    for t in [i/100 for i in range(1,100)]:
        tn = sum(1 for doi, s in scores.items() if s < t and labels_by_doi[doi] == "control")
        spec = tn/n_ctrl
        if spec>= min_specificity:
            return t
        
        if best is None or spec> best[1]:
            best = (t,spec)
    return best[0] if best else 0.5


def train_image_classifier(papers):
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="openai")
    model = model.to(device).eval()


    def embed(paper_list):
        X, y, dois = [], [], []
        for p in paper_list:
            if not p["image_paths"]:
                continue
            vecs = []

            for path in p["image_paths"]:
                try:
                    img = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
                    with torch.no_grad():
                        vecs.append(model.encode_image(img).cpu().numpy()[0])
                except Exception:
                    continue

            if not vecs:
                continue
            X.append(np.mean(vecs, axis=0))
            y.append(1 if p["label"] == "paper_mill" else 0)
            dois.append(p["doi"])
        return np.array(X),np.array(y),dois

    X_train,y_train, _ = embed(by_split(papers, "image", "train"))
    clf =LogisticRegression(max_iter=2000).fit(X_train, y_train)
    return clf, model, preprocess


def image_scores(clf, clip_model, preprocess, papers):
    scores = {}

    for p in papers:
        if not p["image_paths"]:
            continue
        vecs= []

        for path in p["image_paths"]:
            try:
                img =preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
                with torch.no_grad():
                    vecs.append(clip_model.encode_image(img).cpu().numpy()[0])
            except Exception:
                continue
        if vecs:
            vec =np.mean(vecs, axis=0).reshape(1, -1)
            scores[p["doi"]]= clf.predict_proba(vec)[0, 1]
    return scores


def train_fusion(papers, text_scores, img_scores):
    train = by_split(papers, "fusion", "train")
    X, y = [],[]
    for p in train:
        doi = p["doi"]
        if doi not in text_scores or doi not in img_scores:
            continue
        X.append([text_scores[doi], p["citation_rate"], img_scores[doi]])
        y.append(1 if p["label"] == "paper_mill" else 0)
    return LogisticRegression(max_iter=2000).fit(np.array(X), np.array(y))


def main():
    papers = load_papers()
    labels_by_doi = {p["doi"]: p["label"] for p in papers}

    abstract_model,abstract_tok= train_bert(papers, use_fulltext=False)
    abstract_model.save_pretrained("abstract_model")
    abstract_tok.save_pretrained("abstract_model")

    fulltext_model, fulltext_tok = train_bert(papers, use_fulltext=True)
    fulltext_model.save_pretrained("fulltext_model")
    fulltext_tok.save_pretrained("fulltext_model")

    image_clf,clip_model,preprocess = train_image_classifier(papers)
    with open("image_classifier.pkl", "wb") as f:
        pickle.dump(image_clf, f)

    abstract_scores =score_papers(abstract_model, abstract_tok, papers, use_fulltext=False)
    fulltext_scores =score_papers(fulltext_model, fulltext_tok, papers, use_fulltext=True)
    img_scores =image_scores(image_clf, clip_model, preprocess, papers)


    fusion_abstract = train_fusion(papers, abstract_scores, img_scores)
    fusion_fulltext = train_fusion(papers, fulltext_scores, img_scores)

    
    with open("fusion_abstract.pkl", "wb") as f:
        pickle.dump(fusion_abstract, f)
    with open("fusion_fulltext.pkl", "wb") as f:
        pickle.dump(fusion_fulltext, f)

    with open("scores.json", "w") as f:
        json.dump({"abstract": abstract_scores, "fulltext": fulltext_scores, "image": img_scores}, f)

    print("training complete")


if __name__ == "__main__":
    main()
