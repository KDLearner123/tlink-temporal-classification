# Temporal Link (TLINK) Binary Classification

An end-to-end NLP project focused on fine-tuning and evaluating small language models on an imbalanced temporal link classification dataset. This project contrasts the performance and configuration trade-offs between a Causal Decoder model (Pythia-70M) and a Bidirectional Encoder model (DistilBERT).

## 📌 Project Overview
The task is to predict whether a temporal link exists (`YES` or `NO`) between two highlighted event or time spans within a given sentence. 
* **Dataset:** `mdg-nlp/tlink-extr-classification-sentence-2-label` on HuggingFace
* **Input Formatting:** Spans are marked using XML-style tags: `<e1>`, `<e2>` for events and `<t1>`, `<t2>` for time expressions.
  * *Example (YES):* "The US embassy `<e1>`filed`</e1>` a note `<t1>`Wednesday`</t1>`"
  * *Example (NO):* "The marine was `<e1>`sentenced`</e1>` for `<e2>`raping`</e2>` a woman"

### The Core Challenge: Data Imbalance
* **Train Set:** 61,440 examples
* **Validation Set:** 7,422 examples
* **Test Set:** 200 examples
* **Label Split:** 91% `YES` / 9% `NO`

Because of this severe imbalance, standard classification accuracy is a misleading metric. **Macro F1** was chosen as the primary evaluation metric to weight both classes equally during validation and testing.

---

## 🧠 Models & Architecture Solutions

### 1. Pythia-70M (Causal Decoder)
Pythia-70M (70 million parameters) was chosen for its training efficiency on limited hardware. Because decoder-only architectures are fundamentally designed for text generation rather than classification, **five distinct workarounds** were implemented to adapt the pipeline:
1. **No pad token:** Reused the EOS token as the pad token.
2. **Left-padding:** Required because Pythia classifies using the final token's hidden state; right-padding would cause the model to look at `[PAD]` tokens instead.
3. **Embedding matrix resize:** Expanded the matrix to accommodate the 8 custom special XML span tags.
4. **Explicit padding configuration:** Set `model.config.pad_token_id` manually.
5. **FP32 Training:** Disabled `fp16` execution due to recurrent gradient numerical instability (`NaN` losses).

### 2. DistilBERT (Bidirectional Encoder)
As an architectural baseline, `distilbert-base-uncased` (66M parameters) was introduced. Encoder models are inherently bidirectional, utilizing a full attention matrix that processes the entire sentence simultaneously. This makes them natively suited for token-relationship classification and significantly simpler to configure, requiring only a token embedding resize.

---

## 🛠️ Training Setup & Loss Engineering
* **Environment:** Google Colab (Free Tier T4 GPU)
* **Framework:** HuggingFace Transformers & PyTorch
* **Hyperparameters:** Learning rate of `2e-6`, batch size of `32`, max sequence length of `256`, and early stopping with a patience of 3.
* **Class Weights (WeightedTrainer):** The HuggingFace `Trainer` class was subclassed to override `compute_loss` with a weighted cross-entropy loss. To prevent training instability while accounting for the minority class, square-root scaling was applied to compute the loss weights:
  * **`NO` weight:** $\sqrt{61440 / (2 \times 5500)} \approx 2.36$
  * **`YES` weight:** $\sqrt{61440 / (2 \times 55940)} \approx 0.74$

---

## 📊 Experimental Results

### Model Performance Comparison

| Metric | Pythia-70M (Decoder) | DistilBERT (Encoder) | Notes |
| :--- | :---: | :---: | :--- |
| **Architecture** | Causal / Left-to-Right | Bidirectional | Encoders are structurally optimized for relationship tracking. |
| **Parameters** | 70M | 66M | Similar computational footprint. |
| **Accuracy** | 0.9101 | 0.9621 | Heavily skewed by majority class performance. |
| **Macro F1** | 0.6673 | 0.8853 | Primary benchmark for balanced evaluation. |
| **F1 (YES Class)** | 0.9515 | 0.9792 | Performance on majority data. |
| **F1 (NO Class)**
