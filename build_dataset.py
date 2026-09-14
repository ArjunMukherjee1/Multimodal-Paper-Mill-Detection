#Creates the dataset from the doi links
import csv
import io

import json
import os

import re

import time
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict

import requests
from PIL import Image

imageDirectory = "images"


def resolve_pmcid(pmid):
    if not pmid:
        return None
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={pmid}&retmode=json"


    try:
        r= requests.get(url, timeout=20)
        if r.status_code != 200:
            return None
        rec =r.json().get("result", {}).get(pmid, {})

        return next((a["value"] for a in rec.get("articleids", []) if a["idtype"] == "pmc"), None)
    except requests.RequestException:
        return None


def openalex_work(doi):
    try:
        r = requests.get(f"https://api.openalex.org/works/doi:{doi}", timeout=20)
    except requests.RequestException:
        return None
    if r.status_code != 200:
        return None
    return r.json()


def fetch_abstract_pubmed(pmid):
    if not pmid:
        return None
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={pmid}&rettype=abstract&retmode=xml"
    try:
        r= requests.get(url, timeout=20)
        if r.status_code != 200:
            return None
        root = ET.fromstring(r.content)
        parts =[el.text for el in root.iter("AbstractText") if el.text]
        return " ".join(parts) if parts else None
    except(requests.RequestException, ET.ParseError):
        return None


def reconstruct_abstract(inv_index):
    if not inv_index:
        return None
    positions= {}
    for word,idxs in inv_index.items():
        for i in idxs:
            positions[i]=word
    return " ".join(positions[i] for i in sorted(positions))


def fetch_fulltext_sections(pmcid):
    if not pmcid:
        return None
    url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
    try:
        r = requests.get(url, timeout=30)
    except requests.RequestException:
        return None
    if r.status_code != 200:
        return None
    try:
        root = ET.fromstring(r.content)
    except ET.ParseError:
        return None

    sentences = []
    for sec in root.iter("sec"):
        sec_type = (sec.get("sec-type") or "").lower()

        title_el = sec.find("title")
        title = (title_el.text or "").lower() if title_el is not None else ""

        if "method" in sec_type or "method" in title or "result" in sec_type or "result" in title:
            text = " ".join(p.text for p in sec.iter("p") if p.text)
            sentences.extend(re.split(r"(?<=[.!?])\s+", text)[:8])

    captions = []
    for fig in root.iter("fig"):
        cap = fig.find("caption")

        if cap is not None:
            text = " ".join(t for t in cap.itertext() if t)
            captions.append(text.strip())

    sentences.extend(captions[:4])

    return " ".join(sentences)


def fetch_images(pmcid, doi):
    if not pmcid:
        return []
    url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/supplementaryFiles"
    try:
        r = requests.get(url, timeout=20)
    except requests.RequestException:
        return []
    
    if r.status_code != 200 or r.content[:2] != b"PK":
        return []
    
    os.makedirs(imageDirectory, exist_ok=True)
    paths = []
    try:
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            for name in zf.namelist():
                if not name.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".tif", ".tiff")):
                    continue
                data = zf.read(name)

                if len(data) < 3000:
                    continue
                try:
                    w, h = Image.open(io.BytesIO(data)).size
                except Exception:
                    continue

                if w < 100 or h < 100 or max(w, h) / min(w, h) > 8:
                    continue
                
                path = os.path.join(imageDirectory, f"{doi.replace('/', '_')}_{len(paths)}{os.path.splitext(name)[1]}")
                with open(path, "wb") as f:
                    f.write(data)
                paths.append(path)
                if len(paths) >= 6:
                    break
    except zipfile.BadZipFile:
        return []
    return paths


def fetch_references(doi):
    work = openalex_work(doi)
    if not work:
        return []
    return [r.split("/")[-1] for r in (work.get("referenced_works") or [])]


def load_rows(path="dataset_dois.csv"):
    with open(path) as f:
        return {r["doi"]: r for r in csv.DictReader(f) if r.get("doi")}


def load_model_splits(path="model_splits.csv"):
    """{doi: {modality: "internal" or "external"}}, the real internal vs.
    external validation assignment actually used in the paper for each
    modality. train_models.py divides each modality's "internal" pool
    into train / optimization / internal_validation itself at runtime
    (see split_internal there), with a fixed seed per modality."""
    splits = defaultdict(dict)
    with open(path) as f:
        for row in csv.DictReader(f):
            splits[row["doi"]][row["modality"]] = row["split"]
    return splits


def load_json(path, default):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default


def already_done(path="built_dataset.jsonl"):
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path) as f:
        for line in f:
            try:
                done.add(json.loads(line)["doi"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def resolve_known_mill_ids(row_by_doi, model_splits, cache_path="known_mill_ids.json"):
    cached = load_json(cache_path, None)
    if cached is not None:
        return set(cached)
    
    mill_dois = [doi for doi, row in row_by_doi.items()
                 if doi in model_splits and row["label"] == "paper_mill"]
    ids = set()

    for i, doi in enumerate(mill_dois):

        work = openalex_work(doi)
        if work and work.get("id"):
            ids.add(work["id"].split("/")[-1])

        if i % 200 == 0:
            print(str(i) + " out of " + str(len(mill_dois)))
            with open(cache_path, "w") as f:
                json.dump(list(ids), f)
        time.sleep(0.05)
    with open(cache_path,"w") as f:
        json.dump(list(ids), f)

    print("Resolved " + str(len(ids)) + " paper mill ids")
    return ids


def main():
    row_by_doi = load_rows()
    model_splits = load_model_splits()
    known_mill_ids = resolve_known_mill_ids(row_by_doi, model_splits)
    done = already_done()
    print(str(len(done)) + " Papers have already been built")

    kept = len(done)
    
    with open("built_dataset.jsonl", "a") as out_f:
        for i, doi in enumerate(model_splits):

            if doi in done:
                continue
            row = row_by_doi.get(doi)

            if not row:
                continue
            try:
                work = openalex_work(doi)
                abstract = reconstruct_abstract(work.get("abstract_inverted_index")) if work else None

                if not abstract:
                    abstract = fetch_abstract_pubmed(row.get("pmid"))

                if not abstract:
                    time.sleep(0.05)
                    continue


                pmcid = row.get("pmcid") or resolve_pmcid(row.get("pmid"))
                fulltext = fetch_fulltext_sections(pmcid)
                image_paths = fetch_images(pmcid, doi)
                refs = fetch_references(doi)
                cited = sum(1 for rid in refs if rid in known_mill_ids)
                citation_rate = cited / len(refs) if refs else 0.0

                record = {
                    "doi": doi,
                    "label": row["label"],
                    "abstract": abstract,
                    "fulltext": fulltext,
                    "image_paths": image_paths,
                    "citation_rate": citation_rate,
                    "text_split": model_splits[doi].get("text"),
                    "image_split": model_splits[doi].get("image"),
                    "fusion_split": model_splits[doi].get("fusion"),
                }
                
                out_f.write(json.dumps(record) + "\n")
                out_f.flush()
                kept += 1
            except Exception as e:
                print("ERROR on " + str(doi) + ": " + str(e))
            if i % 100 == 0:
                print(str(i) + "/"+ str(len(model_splits)) +" scanned and "+ str(kept) +" kept")
            time.sleep(0.05)

    print(str(kept) + " papers total")


if __name__ == "__main__":
    main()
