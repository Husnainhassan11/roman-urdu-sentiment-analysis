# Roman Urdu Sentiment Analysis

Classifies Roman Urdu text (Urdu written in the Latin alphabet, as people write on social media) as **Negative**, **Neutral** or **Positive**. The project compares a simple TF-IDF baseline with a fine-tuned XLM-RoBERTa transformer, and looks closely at why scores stop around 67%.

**Fine-tuned model:** https://huggingface.co/husnainhassan/roman-urdu-sentiment-xlmr

**Status:** Models are trained and compared on the validation set. The final test-set evaluation, deployment and a web demo are next.

## Key findings so far

- A tiny TF-IDF + Logistic Regression model (macro-F1 **0.671**) is **as good as** the fine-tuned XLM-RoBERTa (macro-F1 **0.670**) on the validation set.
- Averaging the two models' probabilities gives a higher validation score (**0.687**), which suggests they make different mistakes. This number is slightly optimistic, because the weight was chosen on the same validation set.
- The labels are subjective. I labeled 100 validation sentences without seeing the dataset labels and agreed with the dataset on **62%**, so scores near 0.67 may be close to what this data allows.
- Training the transformer longer helped up to epoch 4, and after that it overfit.

---

## Dataset

- **Source:** [community-datasets/roman_urdu](https://huggingface.co/datasets/community-datasets/roman_urdu) on Hugging Face, originally released at [Smat26/Roman-Urdu-Dataset](https://github.com/Smat26/Roman-Urdu-Dataset). Please see these pages for the authors and the citation.
- **Size:** 20,229 sentences labeled Positive, Negative or Neutral (e-commerce reviews, Facebook comments and tweets)
- **Original class balance:** Neutral 44.1%, Positive 29.7%, Negative 26.1%

### Cleaning

Notebook: [`notebooks/02_baselines.ipynb`](notebooks/02_baselines.ipynb)

| Step | Rows removed | Why |
|---|---|---|
| Empty sentences | 113 | A label with no text has nothing to learn from |
| Same text with conflicting labels | 106 | The correct label cannot be known |
| Exact duplicates (ignoring case) | 515 | Would leak into both training and test sets |
| **Total** | **734 (3.6%)** | **19,495 sentences kept** |

Class balance after cleaning: Neutral 43.3%, Positive 29.8%, Negative 26.8%.

### Split

Stratified (same class proportions in every part), `random_state=42`:

| Set | Sentences | Used for |
|---|---|---|
| Train | 15,596 | Learning |
| Validation | 1,949 | Comparing models and settings |
| Test | 1,950 | One final evaluation, after all choices are fixed |

The three files are in [`Data/`](Data/), so every notebook uses exactly the same split.

### Exploration

Notebook: [`notebooks/01_data_exploration.ipynb`](notebooks/01_data_exploration.ipynb)

- Sentences are short: median 9 words, 75% have 17 words or fewer, the maximum is 301
- With the XLM-RoBERTa tokenizer, the 50th, 90th, 95th and 99th percentiles are 17, 44, 60 and 100 tokens, so a maximum length of 128 covers almost everything
- The same word has many spellings (`nhi`, `bt`, `h`, `hn`), and English is mixed in
- Some labels look wrong, for example "nice" labeled Neutral

---

## Models and validation results

All scores are on the 1,949 validation sentences. **Macro-F1** gives every class equal weight, and I use it as the main number because always predicting "Neutral" already reaches 43% accuracy.

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Dummy baseline (always "Neutral") | 0.434 | 0.202 |
| TF-IDF + Logistic Regression | 0.681 | 0.672 |
| XLM-RoBERTa, 3 epochs (v1) | 0.653 | 0.645 |
| XLM-RoBERTa, best of 6 epochs (v2) | 0.674 | 0.670 |
| Average of TF-IDF and XLM-RoBERTa v2 (50/50) | not recorded | 0.687 |

(The TF-IDF model scored 0.671 in the notebook where I combined it with the transformer, because of a tiny difference in how the text was prepared.)

**TF-IDF + Logistic Regression** uses word 1 to 2 grams and character 2 to 5 grams. The character n-grams help with Roman Urdu spelling variation, because `nhi` and `nahi` share many letter pieces.

### Fine-tuning XLM-RoBERTa

Notebooks: [`03_transformer.ipynb`](notebooks/03_transformer.ipynb) and [`04_improving.ipynb`](notebooks/04_improving.ipynb), run on a free Colab T4 GPU.

- **v1:** learning rate 2e-5, 3 epochs (about 8.5 minutes). The validation loss was still falling, so it stopped too early.
- **v2:** learning rate 3e-5, 6 epochs (about 14 minutes), keeping the best epoch by validation macro-F1.

| Epoch | Training loss | Validation loss | Accuracy | Macro-F1 |
|---|---|---|---|---|
| 1 | not logged | 0.898 | 0.614 | 0.603 |
| 2 | 0.984 | 0.842 | 0.633 | 0.607 |
| 3 | 0.833 | 0.795 | 0.668 | 0.659 |
| **4** | 0.724 | 0.828 | **0.674** | **0.670** |
| 5 | 0.618 | 0.892 | 0.661 | 0.657 |
| 6 | 0.534 | 0.925 | 0.663 | 0.658 |

From epoch 4 onwards the training loss keeps falling while the validation loss rises, which is overfitting. The model with the best validation macro-F1 (epoch 4) was kept.

### Combining the two models

Averaging the predicted probabilities, with a weight on the transformer:

| Weight on XLM-R | Validation macro-F1 |
|---|---|
| 0.0 (TF-IDF only) | 0.671 |
| 0.3 | 0.683 |
| **0.5** | **0.687** |
| 0.7 | 0.681 |
| 1.0 (XLM-R only) | 0.670 |

### Per-class results (validation)

| Class | TF-IDF F1 | XLM-R v2 F1 |
|---|---|---|
| Negative | 0.627 | 0.648 |
| Neutral | 0.715 | 0.703 |
| Positive | 0.674 | 0.659 |

### Error analysis

- **Neutral is where mistakes go.** For TF-IDF, 506 of 622 mistakes (81%) involve the Neutral class. Positive and Negative are rarely confused with each other (116 cases).
- **The transformer flips polarity more often.** It confuses Negative with Positive in 159 cases, against 116 for TF-IDF. Calling a bad review good is the most costly mistake for a review tool.
- **Many "mistakes" are label problems.** Among the sentences the transformer was most sure about and still got wrong, many are prayers labeled inconsistently, plain compliments labeled Neutral, or sarcasm.
- **Easy and hard examples:** the transformer handled "bohat acha product hai" (Positive, 0.95) and "kal delivery aa jaye gi" (Neutral, 0.92), but called "bakwas service, paise zaya", which is clearly negative, Neutral with only 0.55 confidence.

---

## Test set

**Not evaluated yet.** The choices are fixed in advance: the transformer from epoch 4 and an ensemble weight of 0.5. The test set will be scored once, and the results will be added here.

## Limitations

- **Subjective labels.** My agreement with the dataset on 100 validation sentences was 62% (one reader and a small sample, so roughly plus or minus 10 points). Scores near 0.67 may be close to the ceiling of this data.
- The labels come from a few social media and review sources, so the model may not transfer to other kinds of text.
- Religious phrases, sarcasm, emojis and English mixed into Roman Urdu are often misjudged.
- Roman Urdu spelling varies a lot, and Urdu in the Arabic script was not tested.
- The ensemble weight and the epoch were chosen on the validation set, so the validation scores are slightly optimistic.
- Near-duplicates (small variations of the same sentence) may still exist across the splits.

---

## Project structure

```
roman-urdu-sentiment-analysis/
├── Data/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── images/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baselines.ipynb
│   ├── 03_transformer.ipynb
│   └── 04_improving.ipynb
├── src/
└── README.md
```

## How to reproduce

1. Open the notebooks in Google Colab. For notebooks 03 and 04, choose **Runtime, Change runtime type, T4 GPU**.
2. The notebooks load the splits directly from this repository:
   `https://raw.githubusercontent.com/Husnainhassan11/roman-urdu-sentiment-analysis/main/Data/`
3. Run the cells in order. Notebook 04 takes about 20 minutes.

## Roadmap

- [x] Data exploration
- [x] Cleaning, stratified split and baselines
- [x] Fine-tune XLM-RoBERTa, and a longer training run
- [x] Compare and combine models on the validation set
- [ ] Final evaluation on the test set
- [ ] Choose the model to deploy and build an API
- [ ] Web demo
- [ ] Update this README with the test results and a live link

## Credits

Dataset by its original authors (see the links above). Base model: [xlm-roberta-base](https://huggingface.co/FacebookAI/xlm-roberta-base) by Facebook AI.

## Author

**Husnain Hassan**, Software Engineering graduate (UMT), aspiring AI engineer
GitHub: [Husnainhassan11](https://github.com/Husnainhassan11)