from eetq.models import AutoEETQForCausalLM
from transformers import AutoTokenizer

model_name = "/data/backup/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
quant_path = "/data/Llama-2-7b-chat-eetq"
model = AutoEETQForCausalLM.from_pretrained(model_name)
model.quantize(quant_path)
tokenizer.save_pretrained(quant_path)