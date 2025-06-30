# AI Clinician Response Prediction – HealthTech Hackathon 2025

## Overview

This repository was developed for the HealthTech Hackathon 2025. The project focuses on building a machine learning model to replicate human clinician responses to clinical case scenarios (vignettes) commonly encountered in Kenyan healthcare settings. 

The aim is to support frontline decision-making in low-resource environments, where access to specialists and diagnostic tools is often limited. By modeling expert clinician responses, the system is intended to assist healthcare workers with accurate, reliable, and context-sensitive guidance.

## Dataset Description

The dataset contains:

- **Prompt**: A clinical vignette that presents a realistic medical case scenario.
- **Response**: A written response by a qualified human clinician.

Although the full dataset includes responses from large language models (LLMs), this challenge is focused solely on replicating the responses written by human clinicians.

## Objective

The primary goal of this project is to:

- Predict accurate and contextually appropriate clinician responses.
- Simulate real-world medical decision-making in low-resource settings.
- Enhance clinical support tools with human-aligned AI models.

## Suggested Approaches

Possible modeling strategies include:

- Fine-tuning pretrained language models (e.g., BERT, RoBERTa, GPT-style models).
- Retrieval-augmented generation using medical knowledge sources.
- Encoder-decoder architectures for response generation.
- Ranking or scoring systems for response selection.

## Technology Stack

- **Programming Language**: Python 3.10+
- **Core Libraries**: PyTorch, Hugging Face Transformers, scikit-learn, pandas, numpy
- **Experiment Tracking**: Weights & Biases or TensorBoard
- **Development Tools**: Jupyter Notebooks, Google Colab (optional)

# Install dependencies
```bash
pip install -q datasets accelerate bitsandbytes peft transformers
```
## Model and Training

The training pipeline fine-tunes Google's `flan-t5-small` model using two key techniques:

- **4-bit Quantization** via `bitsandbytes`: reduces memory usage and speeds up training
- **LoRA (Low-Rank Adaptation)** via the `peft` library: enables efficient fine-tuning without updating all model parameters

## Script: `training pipeline.py`

This script performs the full training process:

- Loads and splits the clinical vignette dataset
- Tokenizes input and output text for the model
- Loads a pre-trained `flan-t5-small` model with 4-bit quantization
- Applies LoRA for parameter-efficient fine-tuning
- Trains the model using Hugging Face's `Trainer` API
- Evaluates performance using the ROUGE metric
- Saves the final fine-tuned model and tokenizer for future use

The script is optimized for training on GPUs and can be run on environments with limited memory.
----
## Script: `pipeline csv generator.py`

### Purpose

This script generates clinical assessment predictions for unseen test prompts using a fine-tuned sequence-to-sequence language model (`flan-t5-small`). It formats model outputs into structured clinician responses and prepares them for submission.

### Use Case

Designed for the test phase of the Kenya Clinical Reasoning Challenge, this script takes raw test data, constructs rich contextual prompts, generates multiple candidate responses, re-ranks them using a rule-based scoring function, and outputs the most relevant clinician-like answer in a required format.

### Key Features

- **Prompt Engineering**: Enhances each test input by including metadata (county, health level, experience, etc.) to give the model richer context.
- **Structured Output**: Encourages the model to produce responses in a defined clinical format:
Summary: ...
Diagnosis: ...
Plan: ...
- **Candidate Generation**: Produces multiple outputs per input (using top-p sampling and beam search).
- **Re-ranking**: Applies a lightweight scoring system to select the most complete and coherent answer from multiple candidates.
- **Formatting & Normalization**: Ensures the response sections are labeled consistently and ordered logically.
- **Fail-safe Handling**: If generation fails, defaults to a safe fallback: _"Refer to higher level facility."_

### Output

- Produces a CSV file with the final predictions:
- `Master_Index`: Unique identifier from the input test set
- `Clinician`: Generated and formatted model response

### How It Works (Theory Overview)

1. **Load Model**: Loads a fine-tuned `flan-t5-small` model with 4-bit quantization for memory efficiency.
2. **Read Test Set**: Reads the test CSV and builds structured prompts using both the case description and metadata.
3. **Generate Responses**: Uses `transformers.generate()` to create multiple output sequences for each prompt.
4. **Normalize and Rank**: Cleans, reorders, and scores outputs to choose the best candidate.
5. **Save Submission**: Writes final responses into the submission CSV file.

----
## ⚠️ Important Notice

Well... here's a confession from your humble junior.

Due to a tragic case of **"I picked the wrong model at submission time"**, my final leaderboard rank dropped from **46th to 331st** after the hackathon ended. Yes, seriously.

I accidentally selected my **worst checkpoint** for the final submission—totally ignoring the better model I had trained earlier. Let’s just say: lesson learned the hard way. The clumsy one strikes again.

### Proof of my blunder:

![image](https://github.com/user-attachments/assets/34f438a5-9c4c-4aa2-a899-96d2cb5590d9)


If you're reviewing this repo, don't worry—the **better model is safely attached here**, clearly marked and ready for your inspection or reproduction. Please ignore the catastrophe I submitted. 😅

_Maybe next time I’ll label my folders something smarter than `checkpoint-uhhh`..._

---
