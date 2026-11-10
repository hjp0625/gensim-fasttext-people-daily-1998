"""Reproducible Gensim representation and newspaper-page classification experiment."""
from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import gensim
import jieba
import numpy as np
import scipy
import sklearn
from gensim.models import FastText, Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "work" / "corpus"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
ID = re.compile(r"^(\d{8})-(\d{2})-(\d{3})-(\d{3})\s+(.*)")
PRON = re.compile(r"\{[a-z]+\d\}", re.I)
WORD = re.compile(r"^[\u4e00-\u9fff]{2,}$|^[A-Za-z]{2,}$")
PAGES = ["07", "08", "01", "09"]
NAMES = {"07": "国际版", "08": "体育版", "01": "头版", "09": "第九版（混合内容）"}


def load_articles():
    articles = defaultdict(list)
    replacement_lines = 0
    for path in sorted(RAW.rglob("*.txt")):
        with path.open(encoding="gb18030", errors="replace") as file:
            for line in file:
                match = ID.match(line)
                if match is None:
                    continue
                date, page, article, paragraph, text = match.groups()
                replacement_lines += "\ufffd" in text
                articles[(date, page, article)].append((paragraph, text.strip()))
    records = []
    for (date, page, article), parts in sorted(articles.items()):
        words = []
        for _, part in sorted(parts):
            part = PRON.sub("", part)
            words.extend(w for w in part.split() if WORD.fullmatch(w))
        if len(words) >= 30:
            records.append({"date": date, "page": page, "id": f"{date}-{page}-{article}",
                            "tokens": words[:500], "length": len(words)})
    return records, {"raw_articles": len(articles), "decoded_replacement_lines": replacement_lines,
                     "articles_after_filter": len(records)}


def mean_vectors(model, docs, weights=None):
    result = np.zeros((len(docs), model.vector_size), dtype=np.float32)
    for row, doc in enumerate(docs):
        vectors = []
        scales = []
        for word in doc:
            if word in model:
                vectors.append(model[word])
                scales.append(1.0 if weights is None else weights.get(word, 0.0))
        if vectors and sum(scales) > 0:
            result[row] = np.average(vectors, axis=0, weights=scales)
    return result


def score(y_test, pred, labels):
    return {"accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "macro_f1": round(float(f1_score(y_test, pred, labels=labels, average="macro")), 4),
            "confusion_matrix": confusion_matrix(y_test, pred, labels=labels).tolist()}


def main():
    records, ingest = load_articles()
    train_all = [r for r in records if r["date"] < "19980501"]
    test_all = [r for r in records if r["date"] >= "19980501"]
    print("Ingest", ingest, "train", len(train_all), "test", len(test_all), flush=True)
    # Learn every representation on training months only. The held-out months never enter vocabulary fitting.
    sentences = [r["tokens"] for r in train_all]
    common = dict(vector_size=100, window=5, min_count=3, workers=1, epochs=5, seed=SEED)
    w2v = Word2Vec(sentences=sentences, sg=1, negative=5, **common)
    ft = FastText(sentences=sentences, sg=1, negative=5, min_n=2, max_n=4,
                  bucket=200000, **common)
    w2v.save(str(OUT / "word2vec.model"))
    ft.save(str(OUT / "fasttext.model"))
    results = {"versions": {"python_packages": {"gensim": gensim.__version__, "scikit_learn": sklearn.__version__,
                                               "scipy": scipy.__version__, "jieba": jieba.__version__}},
               "source": {"files": 6, "format": "GB18030; date-page-article-paragraph plus presegmented text"},
               "ingest": ingest, "settings": {**common, "fasttext_min_n": 2, "fasttext_max_n": 4,
                                                "fasttext_bucket": 200000, "train_dates": "1998-01 through 1998-04",
                                                "test_dates": "1998-05 through 1998-06", "min_article_tokens": 30,
                                                "max_article_tokens": 500, "pages": NAMES},
               "representation": {}, "classification": {}}
    examples = ["经济", "足球", "外交", "计算机", "超级足球联赛"]
    results["representation"]["neighbors"] = {}
    for model_name, model in [("word2vec", w2v.wv), ("fasttext", ft.wv)]:
        results["representation"]["neighbors"][model_name] = {}
        for word in examples:
            try:
                results["representation"]["neighbors"][model_name][word] = [
                    [w, round(float(s), 4)] for w, s in model.most_similar(word, topn=5)]
            except KeyError:
                results["representation"]["neighbors"][model_name][word] = "OOV"
    results["representation"]["vocabulary_sizes"] = {"word2vec": len(w2v.wv), "fasttext": len(ft.wv)}

    for n_classes in [2, 3, 4]:
        pages = PAGES[:n_classes]
        tr = [r for r in train_all if r["page"] in pages]
        te = [r for r in test_all if r["page"] in pages]
        y_train = [r["page"] for r in tr]
        y_test = [r["page"] for r in te]
        train_tokens = [r["tokens"] for r in tr]
        test_tokens = [r["tokens"] for r in te]
        report = {"train_counts": dict(Counter(y_train)), "test_counts": dict(Counter(y_test)), "models": {}}
        baseline = make_pipeline(TfidfVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False,
                                                 min_df=3, max_df=0.95, ngram_range=(1, 2)),
                                 LogisticRegression(max_iter=1000, random_state=SEED))
        baseline.fit([" ".join(x) for x in train_tokens], y_train)
        report["models"]["tfidf_word_1_2"] = score(y_test, baseline.predict([" ".join(x) for x in test_tokens]), pages)
        for model_name, model in [("word2vec_mean", w2v.wv), ("fasttext_mean", ft.wv)]:
            x_train = mean_vectors(model, train_tokens)
            x_test = mean_vectors(model, test_tokens)
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEED))
            clf.fit(x_train, y_train)
            report["models"][model_name] = score(y_test, clf.predict(x_test), pages)
        results["classification"][str(n_classes)] = report
        print("Classes", n_classes, {k: v["macro_f1"] for k, v in report["models"].items()}, flush=True)
    (OUT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved results.json", flush=True)


if __name__ == "__main__":
    main()
