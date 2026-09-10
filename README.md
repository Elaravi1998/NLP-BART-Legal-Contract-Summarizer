# ⚖️ Legal Contract Summarizer

> 🚀 **AI-powered abstractive summarization for legal and compliance documents using a pretrained BART Transformer**

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-Hugging%20Face-yellow.svg)](https://huggingface.co/transformers/)
[![NLP](https://img.shields.io/badge/NLP-Legal%20Text-purple.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Overview

**Legal Contract Summarizer** is an end-to-end Natural Language Processing (NLP) project designed to generate concise summaries from legal and compliance documents.

The system uses a pretrained **BART (`facebook/bart-large-cnn`) encoder-decoder Transformer** through transfer learning. It is designed to handle both short legal clauses and long, multi-page contracts.

The pipeline combines:

* 🧹 Faithfulness-preserving legal text preprocessing
* 🔤 BART byte-level BPE tokenization
* 🤖 Pretrained BART Transformer
* 🧠 Self-attention and multi-head attention
* 📚 Sentence-aware document chunking
* 🔄 Map-reduce summarization for long documents
* 🛡️ Lightweight faithfulness/fact-grounding verification
* 📊 ROUGE-1, ROUGE-2 and ROUGE-L evaluation
* 🧪 Testing across short, medium and long legal documents

The complete workflow is implemented in a Google Colab-compatible Jupyter Notebook.

---

## 🎯 Project Objective

Legal agreements and compliance documents can contain hundreds or thousands of words, making manual review time-consuming.

The goal of this project is to automatically transform lengthy legal text into a concise summary while preserving important information such as:

* 👥 Parties
* 📅 Dates
* 💰 Monetary amounts
* 📜 Obligations
* ⚖️ Legal requirements
* 🔐 Compliance conditions
* 📝 Important contractual clauses

The system targets approximately **30% of the original input length**, while maintaining important legal information.

---

## ✨ Key Features

### 🤖 1. Abstractive Text Summarization

The project uses:

```text
facebook/bart-large-cnn
```

BART is an encoder-decoder Transformer architecture already trained for summarization.

Instead of simply selecting sentences from the original document, the model generates a new, coherent summary.

---

### 🧠 2. Transformer Architecture

The project demonstrates important Transformer concepts including:

* Self-attention
* Multi-head attention
* Positional embeddings
* Encoder-decoder architecture
* Autoregressive text generation
* Transfer learning

The notebook also exposes encoder attention weights using the eager attention implementation.

---

### 📚 3. Long Document Support

BART has a limited input token window.

Instead of silently truncating long legal documents, this project uses **sentence-aware map-reduce chunking**.

```text
Long Legal Document
        │
        ▼
Sentence Segmentation
        │
        ▼
Token-Aware Chunking
        │
        ├──────────────┐
        ▼              ▼
    Chunk 1          Chunk 2
        │              │
        ▼              ▼
    Summary          Summary
        │              │
        └──────┬───────┘
               ▼
        Combined Summaries
               │
               ▼
        Final Reduction
               │
               ▼
       Final Legal Summary
```

This allows the system to process documents longer than the model's native input window.

---

### 🛡️ 4. Faithfulness Check

ROUGE measures textual overlap, but it cannot reliably detect hallucinated legal facts.

For example:

```text
Source:
Party A shall pay Party B $50,000 on 1 July 2026.

Generated:
Party A shall pay Party C $60,000 on 1 August 2026.
```

A summary can potentially look linguistically reasonable while containing incorrect facts.

To address this, the project extracts:

* 💰 Monetary values
* 📅 Dates
* 👤 Multi-word named entities / defined terms

and checks whether those facts are grounded in the original source.

The system reports:

```text
Faithfulness precision
Grounded facts
Ungrounded facts
```

> ⚠️ This is intentionally a lightweight heuristic. A production system should use stronger NER and Natural Language Inference (NLI) based verification.

---

### 📊 5. ROUGE Evaluation

The project evaluates generated summaries using:

| Metric        | Purpose                                            |
| ------------- | -------------------------------------------------- |
| 🟢 ROUGE-1    | Unigram/content overlap                            |
| 🟡 ROUGE-2    | Bigram/phrase overlap                              |
| 🔵 ROUGE-L    | Longest common subsequence / structural similarity |
| 🟣 ROUGE-Lsum | Summary-level ROUGE-L                              |

The evaluation uses the **BillSum** legal-domain dataset when available.

A small evaluation sample is used by default so the notebook remains practical on CPU.

---

### 🧪 6. Multiple Document Lengths

The project tests three scenarios:

```text
1️⃣ Short legal clause
2️⃣ Medium compliance notice
3️⃣ Long multi-page-style contract
```

The long contract intentionally exceeds the BART input window to demonstrate the chunking pipeline.

---

# 🏗️ System Architecture

```text
                ┌───────────────────────┐
                │   Legal Document      │
                └───────────┬───────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │    Preprocessing      │
                │                       │
                │ • Remove PDF noise    │
                │ • Fix whitespace      │
                │ • Preserve casing     │
                │ • Preserve legal text │
                └───────────┬───────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │     Tokenization      │
                │                       │
                │   BART Byte-Level BPE │
                └───────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │  Input Length Check  │
                 └──────────┬───────────┘
                            │
                 ┌──────────┴──────────┐
                 │                     │
             Short Text            Long Text
                 │                     │
                 │                     ▼
                 │             Sentence Chunking
                 │                     │
                 │                     ▼
                 │              MAP Summaries
                 │                     │
                 │                     ▼
                 │             REDUCE Summary
                 │                     │
                 └──────────┬──────────┘
                            ▼
                 ┌───────────────────────┐
                 │   BART Generation     │
                 │                       │
                 │ facebook/bart-large- │
                 │ cnn                   │
                 └───────────┬───────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │  Faithfulness Check   │
                 │                       │
                 │ Dates / Money / NER   │
                 └───────────┬───────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │   Final Summary       │
                 └───────────┬───────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │   ROUGE Evaluation    │
                 └───────────────────────┘
```

---

# 🔬 Methodology

## 1. Preprocessing

Legal documents require conservative preprocessing.

The project deliberately avoids:

* ❌ Lowercasing
* ❌ Stemming
* ❌ Stop-word removal
* ❌ Aggressive text normalization

Instead, it performs lightweight cleanup:

* Remove form-feed/control characters
* Remove artifacts such as `Page X of Y`
* Repair broken line wrapping
* Normalize whitespace

This is important because exact casing, numbers and terminology can carry legal meaning.

---

## 2. Tokenization

The tokenizer associated with the BART checkpoint is loaded using:

```python
AutoTokenizer.from_pretrained(MODEL_NAME)
```

BART uses **byte-level Byte Pair Encoding (BPE)**.

The tokenizer converts text into subword tokens and then into token IDs understood by the Transformer.

---

## 3. Model

The project uses:

```python
MODEL_NAME = "facebook/bart-large-cnn"
```

BART provides an encoder-decoder architecture suitable for abstractive summarization.

The model is used directly without additional fine-tuning.

### Why BART?

✅ Strong summarization capability
✅ Pretrained Transformer architecture
✅ Encoder-decoder design
✅ Suitable for abstractive generation
✅ Easy integration through Hugging Face Transformers

---

# ⚙️ Configuration

Important configuration values include:

```python
MODEL_NAME = "facebook/bart-large-cnn"

MAX_INPUT_TOKENS = 1024
OVERLAP_SENTENCES = 1

TARGET_COMPRESSION = 0.30

MIN_SUMMARY_TOKENS = 40
MAX_SUMMARY_TOKENS = 220

NUM_BEAMS = 4
NO_REPEAT_NGRAM = 3
LENGTH_PENALTY = 2.0
```

### 🎯 Compression

The target compression ratio is approximately:

```text
30% of the original input length
```

This provides meaningful compression without aggressively removing potentially important legal information.

---

# 🔄 Decoding Strategy

The project uses **beam search** rather than greedy generation.

```python
num_beams = 4
```

Additional controls include:

```python
no_repeat_ngram_size = 3
length_penalty = 2.0
```

### Why?

Legal summarization requires:

* Coherence
* Deterministic generation
* Reduced repetition
* Reasonable output length

Sampling is intentionally disabled because creative variation is undesirable for high-stakes legal text.

---

# 🛡️ Faithfulness Verification

The system extracts potentially important factual information from the generated summary.

### Extracted information

```text
💰 Monetary amounts
📅 Dates
👤 Named entities / defined terms
```

The extracted facts are compared against the original source.

Example:

```text
Source:
Party A shall pay Party B $50,000 on 1 July 2026.

Summary:
Party A shall pay Party C $60,000 on 1 August 2026.
```

Expected behavior:

```text
❌ Party C
❌ $60,000
❌ 1 August 2026
```

The system therefore provides an additional warning layer beyond ROUGE.

---

# 📊 Evaluation Dataset

The notebook attempts to use:

```text
BillSum
```

BillSum is a legal-domain summarization dataset containing legislative bills and human-written summaries.

The default evaluation sample is intentionally small:

```python
N_EVAL = 5
```

This keeps inference practical, particularly when running on CPU.

If BillSum cannot be downloaded, the notebook contains a small inline fallback dataset so the complete pipeline can still execute.

---

# 🧪 Test Cases

The project includes three test scenarios.

### 1️⃣ Short Clause

Example:

```text
Either Party may terminate this Agreement upon
thirty (30) days' prior written notice.
```

### 2️⃣ Medium Compliance Notice

A compliance requirement involving:

* Data breaches
* Notification deadlines
* Supervisory authorities
* Documentation requirements

### 3️⃣ Long Contract

A multi-clause contract intentionally constructed to exceed the 1024-token input window.

This demonstrates the map-reduce summarization mechanism.

---

# 💻 Technologies Used

| Technology                   | Purpose                       |
| ---------------------------- | ----------------------------- |
| 🐍 Python                    | Programming language          |
| 🔥 PyTorch                   | Deep learning framework       |
| 🤗 Hugging Face Transformers | Transformer model + tokenizer |
| 🧠 BART                      | Abstractive summarization     |
| 📚 Hugging Face Datasets     | BillSum dataset               |
| 📊 Evaluate                  | Evaluation framework          |
| 📈 ROUGE                     | Summarization evaluation      |
| 📝 NLTK                      | Sentence segmentation         |
| ⚡ Accelerate                 | Device handling               |
| 📓 Google Colab              | Notebook execution            |

---

# 📦 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/legal-contract-summarizer.git
cd legal-contract-summarizer
```

Install dependencies:

```bash
pip install -U transformers datasets evaluate rouge_score nltk sentencepiece accelerate
```

PyTorch is expected to be available in the Google Colab environment.

---

# 🚀 Running the Project

## Option 1 — Google Colab

Open the notebook:

```text
legal_contract_summarizer.ipynb
```

Then execute all cells from top to bottom.

The notebook automatically:

1. Installs dependencies
2. Detects GPU/CPU
3. Downloads NLTK resources
4. Loads the tokenizer
5. Loads BART
6. Preprocesses text
7. Performs tokenization
8. Generates summaries
9. Handles long documents
10. Performs faithfulness checking
11. Calculates ROUGE scores
12. Runs sample test cases

---

# ⚡ GPU Recommendation

A GPU is recommended for faster inference.

In Google Colab:

```text
Runtime
   ↓
Change runtime type
   ↓
Hardware accelerator
   ↓
GPU
```

A T4 GPU can significantly reduce inference time compared with CPU execution.

---

# 🧩 Example Usage

```python
legal_text = """
The Tenant shall pay a monthly rent of $2,000 beginning
1 July 2026. The Tenant shall maintain the premises and
shall not sublet the property without the Landlord's
prior written consent.
"""

summary = summarize_document(legal_text)

print(summary)
```

Check factual grounding:

```python
report = faithfulness_report(
    legal_text,
    summary
)

print(report)
```

---

# 📈 Example Pipeline Output

```text
==============================================================
LONG CONTRACT (chunked) | input XXXX tok -> summary XXX tok
--------------------------------------------------------------

Generated legal summary...

--------------------------------------------------------------
Faithfulness precision: 100%
All extracted facts grounded in source
```

Actual ROUGE values depend on the downloaded dataset/model versions and runtime execution.

---

# 🆚 Extractive vs Abstractive Summarization

This project chooses **abstractive summarization**.

### 📌 Extractive

Extractive summarization selects existing sentences from the source.

Advantages:

* Lower hallucination risk
* Source sentences remain unchanged

Disadvantages:

* Can produce disconnected summaries
* May contain unnecessary information
* Less concise

### 📌 Abstractive

Abstractive summarization generates new text representing the source.

Advantages:

* More fluent
* More concise
* Better coherence
* Can combine information from multiple sentences

Disadvantages:

* Potential hallucination
* Generated facts must be verified

### 🏆 Project Choice

```text
Abstractive BART
       +
Faithfulness Guardrail
```

This provides a practical balance between readability and factual grounding.

---

# 🔐 Important Legal Safety Note

⚠️ **This project is an NLP research/engineering demonstration and is NOT a substitute for legal advice, professional legal review, or attorney judgment.**

Generated summaries may omit, misinterpret, or incorrectly represent important legal information.

For production legal systems, additional validation should be implemented before relying on generated output.

---

# ⚠️ Limitations

Although the system includes a faithfulness check, several limitations remain.

### 1. Lightweight fact matching

The current verification mechanism primarily checks whether extracted facts appear in the source.

It does not fully understand:

* Negation
* Legal entailment
* Context
* Contradictions
* Numerical relationships
* Complex references

### 2. Model domain mismatch

`facebook/bart-large-cnn` is a general summarization checkpoint and is not specifically fine-tuned for every type of legal contract.

### 3. Context window

BART's input limit requires long documents to be processed using chunking and map-reduce.

### 4. ROUGE limitations

ROUGE measures lexical overlap and should not be treated as a complete measure of legal correctness.

---

# 🚀 Future Improvements

Several upgrades could make this project significantly stronger.

## 🔹 1. Legal-domain fine-tuning

Fine-tune a Transformer on a larger legal summarization corpus.

Potential models/datasets:

```text
Legal-domain BART
Legal-domain T5
LongT5
LED
Longformer Encoder-Decoder
```

---

## 🔹 2. Stronger NER

Replace the regex-based entity extraction with a dedicated legal NER model.

Potential entities:

```text
PERSON
ORGANIZATION
PARTY
DATE
MONEY
LOCATION
LAW
REGULATION
CONTRACT_TERM
```

---

## 🔹 3. NLI-Based Faithfulness

Add a Natural Language Inference model to determine whether generated statements are actually entailed by the source.

```text
Source Sentence
       │
       ▼
Generated Claim
       │
       ▼
NLI Model
       │
 ┌─────┼─────┐
 ▼     ▼     ▼
Entailment Neutral Contradiction
```

This would provide a much stronger hallucination-detection mechanism.

---

## 🔹 4. Better Long-Document Models

Future versions could use architectures specifically designed for long-context documents.

Examples:

```text
Longformer
LED
LongT5
Modern long-context LLMs
```

---

## 🔹 5. Legal Clause Classification

Add automatic classification for:

* Termination
* Payment
* Confidentiality
* Liability
* Indemnification
* Data protection
* Intellectual property
* Governing law

---

## 🔹 6. Interactive Web Application

A future version could provide a web interface:

```text
┌─────────────────────────────────────┐
│       ⚖️ Legal AI Summarizer        │
├─────────────────────────────────────┤
│                                     │
│   📄 Upload Contract                │
│                                     │
│   [ Choose File ]                   │
│                                     │
│   Compression: 30%                  │
│                                     │
│   [ Generate Summary ]              │
│                                     │
├─────────────────────────────────────┤
│ 📋 Summary                          │
│                                     │
│ ...                                 │
│                                     │
├─────────────────────────────────────┤
│ 🛡️ Faithfulness: 96%               │
└─────────────────────────────────────┘
```

Possible technologies:

```text
Streamlit
FastAPI
React
Gradio
```

---

# 📁 Suggested Repository Structure

```text
legal-contract-summarizer/
│
├── 📓 legal_contract_summarizer.ipynb
├── 📄 README.md
├── 📄 requirements.txt
├── 📄 LICENSE
│
├── 📁 src/
│   ├── preprocessing.py
│   ├── summarizer.py
│   ├── chunking.py
│   ├── faithfulness.py
│   └── evaluation.py
│
├── 📁 data/
│   └── .gitkeep
│
├── 📁 examples/
│   └── sample_contract.txt
│
└── 📁 outputs/
    └── .gitkeep
```

> 💡 For the current submission, the notebook can remain the main implementation. The `src/` structure is a recommended future refactor.

---

# 🎥 Explanation Video Talking Points

The project can be explained in approximately this order:

### 1️⃣ Problem

> Legal documents are long and information-dense. The objective is to generate concise summaries while preserving important legal facts.

### 2️⃣ Model Choice

> I selected pretrained BART because it provides an encoder-decoder Transformer architecture already designed for summarization.

### 3️⃣ Preprocessing

> I intentionally use lightweight preprocessing because aggressive normalization can remove legally meaningful information.

### 4️⃣ Tokenization

> BART uses byte-level BPE tokenization, allowing the model to process words and subword units.

### 5️⃣ Attention

> Self-attention allows tokens to incorporate context from other tokens, while multi-head attention allows the model to capture multiple relationships simultaneously.

### 6️⃣ Long Documents

> Because BART has a limited input window, long documents are divided into sentence-aware chunks and summarized using a map-reduce strategy.

### 7️⃣ Faithfulness

> ROUGE alone cannot detect fabricated legal facts, so I added a lightweight fact-grounding check for dates, monetary values and named entities.

### 8️⃣ Evaluation

> The system evaluates summaries using ROUGE-1, ROUGE-2 and ROUGE-L on a legal-domain dataset.

### 9️⃣ Generalization

> Finally, I test short clauses, medium compliance notices and long contract-style documents to demonstrate robustness across document lengths.

---

# 📚 Project Highlights

```text
🤖 Pretrained Transformer
⚖️ Legal NLP
📝 Abstractive Summarization
🧠 BART Encoder-Decoder
🔤 Byte-Level BPE
🔄 Map-Reduce Chunking
🛡️ Faithfulness Verification
📊 ROUGE Evaluation
📚 BillSum Dataset
🚀 Google Colab Compatible
```

---

# 🎓 Skills Demonstrated

This project demonstrates practical knowledge of:

* Natural Language Processing
* Transformer architectures
* Attention mechanisms
* Transfer learning
* Text preprocessing
* Subword tokenization
* Abstractive summarization
* Long-document processing
* Model inference
* Evaluation metrics
* Hallucination/faithfulness awareness
* Legal-domain NLP
* PyTorch
* Hugging Face Transformers
* Dataset handling
* Experiment design

---

# ⭐ If You Find This Project Useful

If this project helped you understand legal NLP, Transformer-based summarization, or long-document processing:

⭐ **Star this repository**

🍴 **Fork the repository**

💬 **Share your feedback**

---

# 📜 License

This project is released under the **MIT License**.

See the `LICENSE` file for details.

---

## 💡 Final Note

This project demonstrates an end-to-end approach to legal text summarization using a pretrained Transformer, with additional engineering considerations for long documents and factual grounding.

> **Summarization should not only be concise — for legal text, it should also be trustworthy. ⚖️🤖**
