# Temporal Link (TLINK) Binary Classification

An end-to-end NLP project focused on fine-tuning and evaluating small 
language models on an imbalanced temporal link classification dataset. 
This project contrasts the performance and configuration trade-offs 
between a Causal Decoder model (Pythia-70M) and a Bidirectional Encoder 
model (DistilBERT).

## 📌 Project Overview
[cite_start]The task is to predict whether a temporal link exists (`YES` 
or `NO`) between two highlighted event or time spans within a given 
sentence[cite: 5, 6]. 
* [cite_start]**Dataset:** 
`mdg-nlp/tlink-extr-classification-sentence-2-label` on HuggingFace [cite: 
7]
* [cite_start]**Input Formatting:** Spans are marked using XML-style tags: 
`<e1>`, `<e2>` for events and `<t1>`, `<t2>` for time expressions[cite: 
8].
  * [cite_start]*Example (YES):* "The US embassy `<e1>`filed`</e1>` a note 
`<t1>`Wednesday`</t1>`" [cite: 9]
  * [cite_start]*Example (NO):* "The marine was `<e1>`sentenced`</e1>` for 
`<e2>`raping`</e2>` a woman" [cite: 10]

### The Core Challenge: Data Imbalance
* [cite_start]**Train Set:** 61,440 examples [cite: 12]
* [cite_start]**Validation Set:** 7,422 examples [cite: 13]
* [cite_start]**Test Set:** 200 examples [cite: 14]
* [cite_start]**Label Split:** 91% `YES` / 9% `NO` [cite: 15]

[cite_start]Because of this severe imbalance, standard classification 
accuracy is a misleading metric[cite: 16]. [cite_start]**Macro F1** was 
chosen as the primary evaluation metric to weight both classes equally 
during validation and testing[cite: 16, 37].

---

## 🧠 Models & Architecture Solutions

### 1. Pythia-70M (Causal Decoder)
[cite_start]Pythia-70M (70 million parameters) was chosen for its training 
efficiency on limited hardware[cite: 18, 20]. [cite_start]Because 
decoder-only architectures are fundamentally designed for text generation 
rather than classification, **five distinct workarounds** were implemented 
to adapt the pipeline[cite: 23, 55]:
1. [cite_start]**No pad token:** Reused the EOS token as the pad 
token[cite: 24].
2. [cite_start]**Left-padding:** Required because Pythia classifies using 
the final token's hidden state; right-padding would cause the model to 
look at `[PAD]` tokens instead[cite: 25].
3. [cite_start]**Embedding matrix resize:** Expanded the matrix to 
accommodate the 8 custom special XML span tags[cite: 26].
4. [cite_start]**Explicit padding configuration:** Set 
`model.config.pad_token_id` manually[cite: 27].
5. [cite_start]**FP32 Training:** Disabled `fp16` execution due to 
recurrent gradient numerical instability (`NaN` losses)[cite: 28, 32].

### 2. DistilBERT (Bidirectional Encoder)
[cite_start]As an architectural baseline, `distilbert-base-uncased` (66M 
parameters) was introduced[cite: 57, 58]. [cite_start]Encoder models are 
inherently bidirectional, utilizing a full attention matrix that processes 
the entire sentence simultaneously[cite: 59]. [cite_start]This makes them 
natively suited for token-relationship classification and significantly 
simpler to configure, requiring only a token embedding resize[cite: 58, 
59].

---

## 🛠️ Training Setup & Loss Engineering
* [cite_start]**Environment:** Google Colab (Free Tier T4 GPU) [cite: 30]
* [cite_start]**Framework:** HuggingFace Transformers & PyTorch [cite: 31]
* [cite_start]**Hyperparameters:** Learning rate of `2e-6`, batch size of 
`32`, max sequence length of `256`, and early stopping with a patience of 
3[cite: 32, 33, 34, 35].
* [cite_start]**Class Weights (WeightedTrainer):** The HuggingFace 
`Trainer` class was subclassed to override `compute_loss` with a weighted 
cross-entropy loss[cite: 38, 39]. [cite_start]To prevent training 
instability while accounting for the minority class, square-root scaling 
was applied to compute the loss weights[cite: 40]:
  * [cite_start]**`NO` weight:** $\sqrt{61440 / (2 \times 5500)} \approx 
2.36$ [cite: 41]
  * [cite_start]**`YES` weight:** $\sqrt{61440 / (2 \times 55940)} \approx 
0.74$ [cite: 42]

---

## 📊 Experimental Results

### Model Performance Comparison
| Metric | Pythia-70M (Decoder) | DistilBERT (Encoder) | Notes |
| :--- | :---: | :---: | :--- |
| **Architecture** | [cite_start]Causal / Left-to-Right [cite: 58, 60] | 
[cite_start]Bidirectional [cite: 58, 59] | [cite_start]Encoders are 
structurally optimized for relationship tracking[cite: 58, 59]. |
| **Parameters** | [cite_start]70M [cite: 18, 20] | [cite_start]66M [cite: 
58] | [cite_start]Similar computational footprint[cite: 58]. |
| **Accuracy** | [cite_start]0.9101 [cite: 44] | *0.9621* | 
[cite_start]Heavily skewed by majority class performance[cite: 16]. |
| **Macro F1** | [cite_start]0.6673 [cite: 47] | *0.8853* | 
[cite_start]Primary benchmark for balanced evaluation[cite: 16, 37]. |
| **F1 (YES Class)** | [cite_start]0.9515 [cite: 45] | *0.9792* | 
[cite_start]Performance on majority data[cite: 58]. |
| **F1 (NO Class)** | [cite_start]0.3830 [cite: 46] | *0.7914* | 
[cite_start]Performance on minority data[cite: 58]. |
| **Training Speed** | [cite_start]~10 min/epoch [cite: 48] | 
[cite_start]~20 min/epoch [cite: 58] | [cite_start]DistilBERT is slower 
due to bidirectional full-attention overhead[cite: 58]. |
| **Workarounds** | [cite_start]5 needed [cite: 58] | [cite_start]1 needed 
[cite: 58] | [cite_start]Encoders are much simpler to deploy for this 
task[cite: 58]. |

### Hardware Benchmarking: T4 GPU vs. TPU v5e-1
[cite_start]An evaluation was performed comparing the runtime hardware 
environments available on Google Colab[cite: 61, 62]:
* [cite_start]**T4 GPU:** Native PyTorch compatibility, proving highly 
efficient for models under 1B parameters[cite: 63, 64].
* [cite_start]**TPU v5e-1:** While offering a 3x higher theoretical peak 
throughput (~197 TFLOPS vs ~65 TFLOPS), the XLA compilation overhead 
completely negated any hardware acceleration advantages for a small 70M 
model[cite: 63, 65].
* [cite_start]**Conclusion:** The T4 GPU is the optimal architecture 
choice for this scale[cite: 64]. [cite_start]TPUs become advantageous 
primarily when parameters scale beyond 1B over prolonged training 
intervals[cite: 66].

---

## 📂 Repository Structure
* [cite_start]`tlink_training_colab.ipynb`: Notebook containing the 
comprehensive dataset pipeline, custom `WeightedTrainer` implementation, 
and the Pythia-70M fine-tuning process[cite: 69].
* [cite_start]`tlink_distilbert_comparison.ipynb`: Notebook tracking the 
baseline run and comparative configuration of the DistilBERT model[cite: 
70].
* [cite_start]*Note: Saved weights (`tlink_model/` and 
`tlink_model_distilbert/`) are backed up via Google Drive and local zip 
storage due to file-size constraints[cite: 72, 73, 74].*

---

## 💻 How to Run Inference

To load the trained checkpoint locally and execute a forward pass:

```python
from transformers import pipeline

# Load the trained classifier pipeline
clf = pipeline('text-classification', model='./tlink_model')

# Run inference on a raw input sentence containing span tags
sample_input = "The department <e1>said</e1> <t1>Wednesday</t1>..."
prediction = clf(sample_input)
print(prediction)
