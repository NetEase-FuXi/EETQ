import os
from transformers import AutoConfig
from eetq.models import *
from eetq.models.base import BaseEETQForCausalLM

EETQ_CAUSAL_LM_MODEL_MAP = {
    "llama": LlamaEETQForCausalLM,
    "baichuan": BaichuanEETQForCausalLM,
    "gemma": GemmaEETQForCausalLM
}

def check_and_get_model_type(model_dir, trust_remote_code=True):
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=trust_remote_code)
    if config.model_type not in EETQ_CAUSAL_LM_MODEL_MAP.keys():
        raise TypeError(f"{config.model_type} isn't supported yet.")
    model_type = config.model_type
    return model_type

class AutoEETQForCausalLM:
    def __init__(self):
        raise EnvironmentError('You must instantiate AutoEETQForCausalLM with\n'
                               'AutoEETQForCausalLM.from_quantized or AutoEETQForCausalLM.from_pretrained')
    
    @classmethod
    def from_pretrained(self, model_path, trust_remote_code=True, safetensors=True,
                              device_map=None, **model_init_kwargs) -> BaseEETQForCausalLM:
        model_type = check_and_get_model_type(model_path, trust_remote_code)

        return EETQ_CAUSAL_LM_MODEL_MAP[model_type].from_pretrained(
            model_path, model_type, trust_remote_code=trust_remote_code, safetensors=safetensors,
            device_map=device_map, **model_init_kwargs
        )

    @classmethod
    def from_quantized(self, quant_path, quant_filename='', max_new_tokens=None,
                       trust_remote_code=True, fuse_layers=True, safetensors=True,
                       device_map="balanced", offload_folder=None, **config_kwargs) -> BaseEETQForCausalLM:
        pass