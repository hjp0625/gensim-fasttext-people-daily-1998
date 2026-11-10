# Gensim 中文文本向量与版面分类实验

使用《人民日报》1998 年 1—6 月语料，训练 Gensim 的 Word2Vec、FastText，并以文章平均词向量做版面分类。项目包含复现脚本、完整结果、实验报告和训练好的模型。分类标签是报纸版面，**不是人工标注的新闻主题**。

## 已有结果

按时间划分：1—4 月训练，5—6 月测试。指标为宏平均 F1。

| 类别数 | 版面 | TF-IDF | Word2Vec 平均向量 | FastText 平均向量 |
|---:|---|---:|---:|---:|
| 2 | 07 国际、08 体育 | 0.9415 | 0.9373 | 0.9421 |
| 3 | 增加 01 头版 | 0.8981 | 0.8712 | 0.8779 |
| 4 | 增加 09 混合版 | 0.8311 | 0.8039 | 0.8096 |

详见 [实验报告](outputs/实验报告.md) 和 [完整结果](outputs/results.json)。

## 复现

需要 Python 3.10。安装依赖：

```powershell
python -m pip install -r requirements.txt
```

将用户提供的《1998-上半年-带音(1).rar》解压到 `work/corpus/`，使其下方包含六个 `.txt` 文件。原始语料不随项目发布。随后运行：

```powershell
python outputs/run_experiment.py
```

脚本会更新 `outputs/results.json`、`outputs/fasttext.model`、`outputs/fasttext.model.wv.vectors_ngrams.npy` 和 `outputs/word2vec.model`。FastText 的 `.model` 与 `.npy` 文件必须一起保留。脚本固定单线程和随机种子，便于复现；不同机器或依赖版本仍可能造成细小数值差异。

## 项目内容

- `outputs/run_experiment.py`：语料解析、表示学习、版面分类与评估。
- `outputs/实验报告.md`：数据清洗、结果分析、FastText 论文与 Gensim 源码对照。
- `outputs/results.json`：每组任务的样本数、准确率、宏平均 F1、混淆矩阵及词向量近邻。
- `outputs/*.model`、`outputs/*.npy`：训练好的 Gensim 模型。

本实验使用的是按词切分的语料，因此脚本没有再调用 jieba。若替换成原始新闻正文，请先分词，并保持相同的文章编号格式。论文与源码参考链接在实验报告末尾。
