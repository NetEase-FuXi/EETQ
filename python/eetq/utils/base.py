import torch
import torch.nn as nn
from eetq.modules import W8A16Linear

def find_submodule(module, sub_name):
    res = None
    if hasattr(module, sub_name):
        return getattr(module, sub_name)
    else:
        for name, m in module.named_children():
            res = find_submodule(m, sub_name)
            if res is not None:
                return res
    raise ValueError(f"Cannot find submodule {sub_name} in module {module}")


def get_op_by_name(module, op_name):
    # get the op by its name relative to the module
    for name, m in module.named_modules():
        if name == op_name:
            return m
    raise ValueError(f"Cannot find op {op_name} in module {module}")


def set_op_by_name(layer, name, new_module):
    levels = name.split('.')
    if len(levels) > 1:
        mod_ = layer
        for l_idx in range(len(levels)-1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        delattr(mod_, levels[-1])
        setattr(mod_, levels[-1], new_module)
    else:
        delattr(layer, name)
        setattr(layer, name, new_module)

def replace_fused_qkv(layer, name, new_module):
    levels = name.split('.')
    mod_ = layer
    for l_idx in range(len(levels)-1):
        if levels[l_idx].isdigit():
            mod_ = mod_[int(levels[l_idx])]
        else:
            mod_ = getattr(mod_, levels[l_idx])

    setattr(mod_, levels[-1], new_module)

    
def replace_fused_gateup(layer, name, new_module):
    levels = name.split('.')
    mod_ = layer
    for l_idx in range(len(levels)-1):
        if levels[l_idx].isdigit():
            mod_ = mod_[int(levels[l_idx])]
        else:
            mod_ = getattr(mod_, levels[l_idx])

    setattr(mod_, levels[-1], new_module)

    
def replace_split_qkv(layer, name, old_module, index_map):
    device = old_module.qweight.device
    q_weight = old_module.qweight[:, index_map[0]: index_map[1]]
    k_weight = old_module.qweight[:, index_map[1]: index_map[2]]
    v_weight = old_module.qweight[:, index_map[2]: ]
    
    q_weight_scales = old_module.weight_scales[index_map[0]: index_map[1]]
    k_weight_scales = old_module.weight_scales[index_map[1]: index_map[2]]
    v_weight_scales = old_module.weight_scales[index_map[2]: ]
    
    q = W8A16Linear(q_weight.shape[0], q_weight.shape[1])
    k = W8A16Linear(k_weight.shape[0], k_weight.shape[1])
    v = W8A16Linear(v_weight.shape[0], v_weight.shape[1])
    
    q.qweight = q_weight.to(device)
    q.weight_scales = q_weight_scales.to(device)
    
    k.qweight = k_weight.to(device)
    k.weight_scales = k_weight_scales.to(device)
    
    v.qweight = v_weight.to(device)
    v.weight_scales = v_weight_scales.to(device)

    levels = name.split('.')
    mod_ = layer
    for l_idx in range(len(levels)-1):
        if levels[l_idx].isdigit():
            mod_ = mod_[int(levels[l_idx])]
        else:
            mod_ = getattr(mod_, levels[l_idx])
    if mod_.q_proj.bias is not None:
        q.bias = mod_.q_proj.bias.to(device)
    if mod_.k_proj.bias is not None:
        k.bias = mod_.k_proj.bias.to(device)
    if mod_.v_proj.bias is not None:
        v.bias = mod_.v_proj.bias.to(device)
    delattr(mod_, 'qkv_proj')
    setattr(mod_, 'q_proj', q)
    setattr(mod_, 'k_proj', k)
    setattr(mod_, 'v_proj', v)


def replace_split_gateup(layer, name, old_module, index_map):
    device = old_module.qweight.device
    gate_weight = old_module.qweight[:, index_map[0]: index_map[1]]
    up_weight = old_module.qweight[:, index_map[1]: ]
    
    gate_weight_scales = old_module.weight_scales[index_map[0]: index_map[1]]
    up_weight_scales = old_module.weight_scales[index_map[1]: ]
    
    gate = W8A16Linear(gate_weight.shape[0], gate_weight.shape[1])
    up = W8A16Linear(up_weight.shape[0], up_weight.shape[1])
    
    gate.qweight = gate_weight.to(device)
    gate.weight_scales = gate_weight_scales.to(device)
    
    up.qweight = up_weight.to(device)
    up.weight_scales = up_weight_scales.to(device)
    
    levels = name.split('.')
    mod_ = layer
    for l_idx in range(len(levels)-1):
        if levels[l_idx].isdigit():
            mod_ = mod_[int(levels[l_idx])]
        else:
            mod_ = getattr(mod_, levels[l_idx])
    if mod_.gate_proj.bias is not None:
        gate.bias = mod_.gate_proj.bias.to(device)
    if mod_.up_proj.bias is not None:
        up.bias = mod_.up_proj.bias.to(device)
    delattr(mod_, 'gateup_proj')
    setattr(mod_, 'gate_proj', gate)
    setattr(mod_, 'up_proj', up)


def split_tp_column(layer, name, old_module, tp):
    device = old_module.weight.device
    levels = name.split('.')
    mod_ = layer
    for l_idx in range(len(levels)-1):
        if levels[l_idx].isdigit():
            mod_ = mod_[int(levels[l_idx])]
        else:
            mod_ = getattr(mod_, levels[l_idx])
    tp_linear = []
    weight = old_module.weight
    step = weight.shape[0] // tp
    
    assert step * tp == weight.shape[0]
    for i in range(tp):
        current_tp_linear = nn.Linear(step, weight.shape[1])
        
        if "qkv" in levels[-1] or "W_pack" in levels:
            hidden_size = weight.shape[0] // 3
            start_idx = i * step // 3
            end_idx = (i + 1) * step // 3
            weights = [weight[start_idx: end_idx, :], weight[hidden_size + start_idx: hidden_size + end_idx, :], weight[2 * hidden_size + start_idx: 2 * hidden_size + end_idx, :]]
            current_tp_linear.weight = nn.Parameter(torch.cat(weights, dim=0), requires_grad=False)
        elif "gateup" in levels[-1]:
            hidden_size = weight.shape[0] // 2
            start_idx = i * step // 2
            end_idx = (i + 1) * step // 2
            weights = [weight[start_idx: end_idx, :], weight[hidden_size + start_idx: hidden_size + end_idx, :]]
            current_tp_linear.weight = nn.Parameter(torch.cat(weights, dim=0), requires_grad=False)
        else:
            raise ValueError("Incorrect layer to split")

        setattr(mod_, f'{levels[-1]}_tp{i}', current_tp_linear)
        
    delattr(mod_, levels[-1])

def split_tp_row(layer, name, old_module, tp):
    device = old_module.weight.device
    levels = name.split('.')
    mod_ = layer
    for l_idx in range(len(levels)-1):
        if levels[l_idx].isdigit():
            mod_ = mod_[int(levels[l_idx])]
        else:
            mod_ = getattr(mod_, levels[l_idx])

    weight = old_module.weight
    step = weight.shape[1] // tp
    assert step * tp == weight.shape[1]
    for i in range(tp):
        current_tp_linear = nn.Linear(weight.shape[1], step)
        current_tp_linear.weight = nn.Parameter(weight[:, i * step: (i + 1) * step], requires_grad=False)
        setattr(mod_, f'{levels[-1]}_tp{i}', current_tp_linear)
        
    delattr(mod_, levels[-1])


def replace_tp(layer, name, new_module, tp):
    levels = name.split('.')
    mod_ = layer
    for l_idx in range(len(levels)-1):
        if levels[l_idx].isdigit():
            mod_ = mod_[int(levels[l_idx])]
        else:
            mod_ = getattr(mod_, levels[l_idx])
    
    for i in range(tp):
        current_name = levels[-1] + f"_tp{i}"
        delattr(mod_, current_name)
    
    setattr(mod_, levels[-1], new_module)


def merge_tp_handler(model, all_tp, s, is_column):
    num_tp = len(all_tp[0])
    for _, tp in enumerate(all_tp):
        name = s.format(_)
        
        dim = 1 if is_column else 0
        
        if "qkv_proj" in name or "W_pack" in name:
            hidden_size = tp[0].qweight.shape[dim] // 3
            q_qweight = torch.cat([x.qweight[:, :hidden_size] for x in tp], dim=dim)  
            k_qweight = torch.cat([x.qweight[:, hidden_size : hidden_size * 2] for x in tp], dim=dim)  
            v_qweight = torch.cat([x.qweight[:, hidden_size * 2: ] for x in tp], dim=dim)  
            
            
            q_scales = torch.cat([x.weight_scales[:hidden_size] for x in tp])        
            k_scales = torch.cat([x.weight_scales[hidden_size: hidden_size * 2] for x in tp])
            v_scales = torch.cat([x.weight_scales[hidden_size * 2: ] for x in tp])

            qweight = [q_qweight, k_qweight, v_qweight]
            weight_scales = [q_scales, k_scales, v_scales]
            
        elif "gateup_proj" in name:
            hidden_size = tp[0].qweight.shape[dim] // 2
            gate_qweight = torch.cat([x.qweight[:, :hidden_size] for x in tp], dim=dim)              
            up_qweight = torch.cat([x.qweight[:, hidden_size: hidden_size * 2] for x in tp], dim=dim)  
            
            gate_scales = torch.cat([x.weight_scales[:hidden_size] for x in tp])    
            up_scales = torch.cat([x.weight_scales[hidden_size: hidden_size * 2] for x in tp])

            qweight = [gate_qweight, up_qweight]
            weight_scales = [gate_scales, up_scales]
            
        elif "o_proj" in name or "down_proj" in name:
            qweight = [x.qweight for x in tp]
            weight_scales = [x.weight_scales for x in tp]
        else:
            raise ValueError("Incorrect layer to merge")
        
        qweight = torch.cat(qweight, dim=dim)
        weight_scales = torch.cat(weight_scales)

        merged_weight = W8A16Linear(qweight.shape[0], qweight.shape[1])
        merged_weight.qweight = qweight
        merged_weight.weight_scales = weight_scales
        
        replace_tp(model, name, merged_weight, num_tp)



def get_op_name(module, op):
    # get the name of the op relative to the module
    for name, m in module.named_modules():
        if m is op:
            return name
    raise ValueError(f"Cannot find op {op} in module {module}")


def append_str_prefix(x, prefix):
    if isinstance(x, str):
        return prefix + x
    elif isinstance(x, tuple):
        return tuple([append_str_prefix(y, prefix) for y in x])
    elif isinstance(x, list):
        return [append_str_prefix(y, prefix) for y in x]
    else:
        return x


def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear) and "lm_head" not in name}


def get_named_layers(module, layers=[nn.Linear]):
    return {name: m for name, m in module.named_modules() if type(m) in layers}

def find_layers(module, include=[nn.Linear], exclude=["lm_head"]):
    res = {}
    for name, m in module.named_modules():
        if type(m) in include and not any([e in name for e in exclude]):
            res.update({name: m})
    return res

def exclude_layers_to_not_quantize(linear_layers, modules_to_not_convert):
    if modules_to_not_convert is None:
        return linear_layers

    filtered_layers = {}
    for name, linear_layer in linear_layers.items():
        if not any(key in name for key in modules_to_not_convert):
            filtered_layers[name] = linear_layer
    return filtered_layers