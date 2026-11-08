from huggingface_hub import snapshot_download
for m in ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-7B"]:
    p = snapshot_download(m, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "tokenizer*", "merges.txt", "vocab.json"])
    print(m, p, flush=True)
