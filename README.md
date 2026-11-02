# 🏷️ E-Commerce Price Predictor (Llama-3.2 Fine-Tune)

An AI engineering project that fine-tunes **Llama-3.2-3B** to predict product prices from unstructured text descriptions.

## 🚀 Key Results
| Model | Error (MAE) | Cost / 1k Tokens |
|-------|-------------|------------------|
| GPT-4o (Baseline) | $76.00 | $0.03 |
| **Fine-Tuned Llama-3** | **$47.00** | **$0.002** (Est.) |


## Folder Structure
```text
├── datasets/
│   ├── dataset_links.txt
├── pricer/
│   ├── evaluator.py
│   └── items.py
├── 01_Data_Engineering_and_Baselines.ipynb
├── 02_QLoRA_FineTuning_Llama3.ipynb
├── 03_Training.ipynb
├── 04_Evaluate_Finetuned_Open_Source_Model.ipynb
├── app.py
├── main.py
├── README.md
├── requirements.txt
├── results.ipynb
└── util.py
```

## 🛠️ Architecture
* **Training:** QLoRA (4-bit) on T4 GPU using `Unsloth` & `TRL`.
* **Dataset:** 50k+ Amazon product descriptions .
* **Serving:** FastAPI microservice with LoRA adapter loading.


Built with Llama-3.2, Unsloth, and FastAPI.
ShoutOut to https://www.udemy.com/course/llm-engineering-master-ai-and-large-language-models/
