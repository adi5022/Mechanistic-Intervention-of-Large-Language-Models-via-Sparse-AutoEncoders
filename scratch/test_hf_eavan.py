from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

prompt = "Eavan Boland was born in"
inputs = tokenizer(prompt, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
probs = torch.softmax(logits[0, -1, :], dim=-1)

top10 = torch.topk(probs, k=10)
print("Top 10 completions for prompt:", repr(prompt))
for r, (p, idx) in enumerate(zip(top10.values, top10.indices), 1):
    print(f"Rank {r}: {tokenizer.decode([idx])!r} (Prob: {p.item()*100:.4f}%)")

target_id = tokenizer.encode(" Dublin")[0]
target_prob = probs[target_id].item()
sorted_indices = torch.argsort(probs, descending=True)
target_rank = (sorted_indices == target_id).nonzero().item() + 1
print(f"\nTarget ' Dublin' (ID {target_id}): Rank #{target_rank}, Prob: {target_prob*100:.6f}%")
