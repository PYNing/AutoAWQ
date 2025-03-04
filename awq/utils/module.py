import torch.nn as nn
import importlib
import logging
from awq.models._config import VisualQuantConfig

def try_import(module_name):
    try:
        module = importlib.import_module(module_name)
        return module, ""
    except Exception as ex:
        return None, str(ex)

def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}


def get_op_by_name(module, op_name):
    # get the op by its name relative to the module
    for name, m in module.named_modules():
        if name == op_name:
            return m
    raise ValueError(f"Cannot find op {op_name} in module {module}")


def set_op_by_name(layer, name, new_module):
    levels = name.split(".")
    if len(levels) > 1:
        mod_ = layer
        for l_idx in range(len(levels) - 1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], new_module)
    else:
        setattr(layer, name, new_module)


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


def exclude_layers_to_not_quantize(linear_layers, modules_to_not_convert):
    if modules_to_not_convert is None:
        return linear_layers

    filtered_layers = {}
    for name, linear_layer in linear_layers.items():
        if not any(key in name for key in modules_to_not_convert):
            filtered_layers[name] = linear_layer
    return filtered_layers

def get_visual_per_layer_quant_strategy(model,
                                        visual_layers_prefix,
                                        visual_quant_config,
                                       ):
    if isinstance(visual_quant_config, VisualQuantConfig):
        visual_quant_config = visual_quant_config.layer_configs
        
    quant_strategy = {
        "w8a8o8": {"weight_quant_bit": 8, "act_quant_bit": 8, "gemm_out_requant_bit": 8},
        "w8a16o8": {"weight_quant_bit": 8, "act_quant_bit": 16, "gemm_out_requant_bit": 8},
        "w8a8o16": {"weight_quant_bit": 8, "act_quant_bit": 8, "gemm_out_requant_bit": 16},
        "w8a16o16": {"weight_quant_bit": 8, "act_quant_bit": 16, "gemm_out_requant_bit": 16},
    }
    
    per_layer_quant_strategy = dict()
    for name, module in model.named_modules():
        if not (name.startswith(visual_layers_prefix) and isinstance(module, nn.Linear)):
            continue
        if name not in visual_quant_config:
            name = "common"
        linear_quant_strategy = visual_quant_config[name]
        linear_quant_strategy = linear_quant_strategy.lower()        
        if linear_quant_strategy in ["fp16", "bf16"]:
            logging.info(f"Quantization for `{name}` is set to disabled, skipping its quantization.")
            continue
        if linear_quant_strategy not in quant_strategy:
            raise RuntimeError(f"Unspport Linear Quantization Strategy: {linear_quant_strategy}")
        per_layer_quant_strategy[name] = linear_quant_strategy
        
    return per_layer_quant_strategy
