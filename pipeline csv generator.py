import pandas as pd
import torch
from tqdm import tqdm
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig, AutoConfig

dev = "cuda" if torch.cuda.is_available() else "cpu"

test_path = "/content/drive/My Drive/Kenya Clinical Reasoning Challenge/Data-set/test.csv"
out_path = "/content/drive/My Drive/Kenya Clinical Reasoning Challenge/submission_flan_rerank2.csv"
mod_path = "/content/drive/My Drive/Kenya Clinical Reasoning Challenge/flan_small/checkpoint-480"

config = AutoConfig.from_pretrained(mod_path)
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(mod_path)
model = AutoModelForSeq2SeqLM.from_pretrained(
    mod_path,
    config=config,
    device_map="auto",
    quantization_config=quant_config,
    low_cpu_mem_usage=True
).to(dev)

def build_prmt(r):
    return (
        f"{r['Prompt']}\n"
        f"Metadata:\n"
        f"- County: {r['County']}\n"
        f"- Health level: {r['Health level']}\n"
        f"- Experience: {r['Years of Experience']} years\n"
        f"- Nursing Competency: {r['Nursing Competency']}\n"
        f"- Clinical Panel: {r['Clinical Panel']}\n\n"
        "Please provide a clinical assessment in this format:\n"
        "Summary: <summary>\nDiagnosis: <diagnosis>\nPlan: <plan>"
    )

def norm_lbls(t):
    t = re.sub(r"(?i)summary\s*[:\-]", "Summary:", t)
    t = re.sub(r"(?i)diagnosis\s*[:\-]", "Diagnosis:", t)
    t = re.sub(r"(?i)plan\s*[:\-]", "Plan:", t)
    return t

def reorder_sec(t):
    l = t.split("\n")
    s = next((x for x in l if x.lower().startswith("summary")), "")
    d = next((x for x in l if x.lower().startswith("diagnosis")), "")
    p = next((x for x in l if x.lower().startswith("plan")), "")
    return "\n".join([s, d, p]).strip()

def has_all(t):
    return (
        re.search(r"(?i)^summary:", t) and
        re.search(r"(?i)^diagnosis:", t) and
        re.search(r"(?i)^plan:", t)
    )

def score_out(t):
    t = t.lower()
    s = 0
    s += 1 if "summary:" in t else 0
    s += 1 if "diagnosis:" in t else 0
    s += 1 if "plan:" in t else 0
    s += 1 if len(t.split()) >= 50 else 0
    s += 0.5 if t.startswith("summary:") else 0
    s += 0.5 if has_all(t) else 0
    s -= 0.5 if t.count("plan") > 1 else 0
    return s

decod_arg = {
    "do_sample": True,
    "temperature": 0.11118,
    "top_p": 0.91954,
    "num_beams": 2,
    "num_return_sequences": 2,
    "max_new_tokens": 160,
    "no_repeat_ngram_size": 3,
    "repetition_penalty": 1.1,
    "length_penalty": 1.2,
    "early_stopping": True,
    "pad_token_id": tokenizer.eos_token_id,
    "eos_token_id": tokenizer.eos_token_id
}

test = pd.read_csv(test_path).fillna("")
pred = []

for i in tqdm(range(len(test)), desc="rerank"):
    try:
        row = test.iloc[i]
        prmt = build_prmt(row)
        inpt = tokenizer(prmt, return_tensors="pt", truncation=True, max_length=768).to(dev)
        output = model.generate(**inpt, **decod_arg)
        candi = [norm_lbls(tokenizer.decode(o, skip_special_tokens=True).split("Clinician:")[-1].strip()) for o in output]
        best = sorted(candi, key=score_out, reverse=True)[0]
        final = reorder_sec(best)
        pred.append(final)
    except Exception as e:
        print(f"Error at index {i}: {e}")
        pred.append("Refer to higher level facility.")

subm = pd.DataFrame({
    "Master_Index": test["Master_Index"],
    "Clinician": pred
})
subm.to_csv(out_path, index=False)
print(f"Saved to: {out_path}")
