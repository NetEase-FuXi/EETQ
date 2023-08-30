Mappings = {
    "llama": "transformers_llama_mapping",
}

transformers_llama_mapping = {
    "decoder": "model.model.layers",
    "decoder_layer": "LlamaDecoderLayer",
    "attention": "LlamaAttention",
    "mlp": "LlamaMLP",
    "layernorm": "LlamaRMSNorm",
    "embedding": "embed_tokens",
}

def get_submodule_name(module, name, sub_name):
    if name == "":
        return module
    node = eval(Mappings[name])
    return node[sub_name]