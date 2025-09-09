# Fine-tuning GPT-2 with LoRA on SQuAD

## 📌 Project Overview
In this project, we fine-tuned **GPT-2** on the **SQuAD question answering dataset** using **LoRA (Low-Rank Adaptation)**.  
The goal was to achieve efficient fine-tuning with limited computational resources while maintaining reasonable performance.

---

## ⚙️ Methodology

### LoRA Configuration
- **r**: 8  
- **alpha**: 16  
- **dropout**: 0.1  
- **target modules**: `attn_c` layers in GPT-2 attention blocks  

Chosen based on LoRA reference papers and prior experiments. `r=8` provides a good trade-off between accuracy and efficiency.

### Hyperparameters
- **Batch size**: 4  
- **Learning rate**: 2e-4  
- **Epochs**: 3  
- **Evaluation strategy**: per epoch  

### Data Preprocessing
- Prompt format:  
  ```
  Answer + Question + Context
  ```
- Added `<eos>` token to mark the end of the answer  
- Masked the prompt tokens in the labels with `-100` to ensure **loss only considers the answer**  

---

## 📊 Experimental Results

- **Trainable parameters**: 294,912 (~0.23% of total GPT-2 parameters)  
- **Metrics**:  
  - Exact Match (EM): 40%  
  - F1: 40%  
- **Loss values**:  
  - Epoch 1 → Train: 0.9869 | Eval: 0.7964  
  - Epoch 2 → Train: 0.8443 | Eval: 0.7621  
  - Epoch 3 → Train: 0.9474 | Eval: 0.7560  

### Observations
- Training and validation loss decreased initially, showing effective learning.  
- LoRA achieved similar performance to full fine-tuning while requiring **far fewer parameters and resources**.  
- Validation loss fluctuations in later epochs are likely due to dataset difficulty and small batch size.  

---

## 🔬 Ablation Study
Comparison with **full fine-tuning**:
- **Full fine-tuning** consumed much more memory and time.  
- No significant improvement in EM/F1 compared to LoRA.  
- In some cases, validation loss was worse.  

➡️ **Conclusion**: LoRA provides almost the same performance with **~100x fewer trainable parameters**.  

---

## 📈 Evaluation Metrics
- **Exact Match (EM)**:  
  Percentage of predictions that exactly match the ground-truth answer.  
- **F1 Score**:  
  Harmonic mean of precision and recall at the token level, better for partially correct answers.  

---

## 💡 Discussion
- **Advantages of LoRA**:  
  - Efficient training (fewer trainable parameters)  
  - Lower memory and compute requirements  
  - Suitable for resource-constrained environments  

- **Limitations**:  
  - May underperform on tasks requiring very fine-grained adaptation  
  - Performance can depend heavily on hyperparameter tuning  

---

## 📦 Requirements
See [`requirements.txt`](./requirements.txt) for all dependencies.

---

## 🚀 How to Run
```bash
# Clone repo
git clone https://github.com/your-username/gpt2-lora-squad.git
cd gpt2-lora-squad

# Install dependencies
pip install -r requirements.txt

# Run training
python train.py
```

---

## 📜 Conclusion
LoRA-based fine-tuning of GPT-2 on SQuAD achieved **~40% EM/F1** with only **0.23% trainable parameters**.  
This demonstrates that **parameter-efficient fine-tuning** is a practical alternative to full fine-tuning, especially when compute resources are limited.
