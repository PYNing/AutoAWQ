import torch
from torch import nn
import functools
from tqdm import tqdm
from collections import defaultdict
from awq.modules.linear import Fused_StaticQuant_IGEMM_Dequant_AddBias_Linear
import numpy as np
import logging
from datasets import load_dataset
from awq.utils.module import set_op_by_name, get_op_by_name
from awq.utils.utils import get_best_device

def get_cali_data(calib_dataset_name, calib_subset, calib_split, image_column, processor, max_calib_samples):
    visual_calib_dataset = load_dataset(path=calib_dataset_name, 
                                        name=calib_subset, 
                                        split=calib_split)

    if len(visual_calib_dataset) < max_calib_samples:
        logging.warning(f"`max_calib_samples` is set to {max_calib_samples}, \
                        but the dataset contains only {len(visual_calib_dataset)} samples. \
                        `max_calib_samples` is adjusted to {len(visual_calib_dataset)}.")
        max_calib_samples = len(visual_calib_dataset)
    
    cali_data = list()
    pbar = tqdm(range(max_calib_samples))
    for i in pbar:
        data = visual_calib_dataset[i]
        image = data[image_column]
        preprocessed_data = processor(text=[""], images=image, padding=True, return_tensors="pt")
        cali_data.append(preprocessed_data)
        
    return cali_data

def get_act_scales(model_wapper,
                   model,
                   per_layer_quant_strategy,
                   cali_data,
                   processor_input_map,
                   ):
    visual_model = model_wapper.get_visual_model(model)
    ori_device = next(visual_model.parameters()).device
    device = get_best_device()
    visual_model.to(device)
    visual_model.eval()
    act_scales = {}

    def stat_tensor(name, tensor):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).abs().detach()
        comming_max = torch.max(tensor, dim=0)[0].float().cpu()
        if name in act_scales:
            act_scales[name] = torch.max(act_scales[name], comming_max)
        else:
            act_scales[name] = comming_max

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)

    hooks = []
    for name, m in model.named_modules():
        if name not in per_layer_quant_strategy:
            continue
        assert isinstance(m, nn.Linear)
        logging.info(f"Add stat_input_hook for {name}")
        hooks.append(m.register_forward_hook(functools.partial(stat_input_hook, name=name)))

    num_samples = len(cali_data)
    pbar = tqdm(range(num_samples))
    for i in pbar:
        preprocessed_data = cali_data[i]
        input_dict = dict()
        for name_in_processor, data in preprocessed_data.items():
            if name_in_processor not in processor_input_map: 
                continue
            if isinstance(data, torch.Tensor):
                data = data.to(device)
            name_in_input = processor_input_map[name_in_processor]
            input_dict[name_in_input] = data
        with torch.inference_mode():
            im_emb = visual_model(**input_dict)

    for h in hooks:
        h.remove()

    visual_model.to(ori_device)
    return act_scales

def get_related_fcs_ln(model, moudle_name, ln_linear_map, quant_config):
    target_group = None
    target_suffix = None
    for group in ln_linear_map:
        for suffix in group:
            if moudle_name.endswith(suffix):
                target_group = group
                target_suffix = suffix
                break
    if target_group is None:
        return None, None, None, None
    
    target_fc_names = list()
    target_fcs = list()
    target_ln_name = list()
    target_ln = list()
    for suffix in target_group:
        target_moudle_name = moudle_name.replace(target_suffix, suffix)
        target_moudle = get_op_by_name(model, target_moudle_name)
        
        if isinstance(target_moudle, nn.Linear):
            target_fc_names.append(target_moudle_name)
            target_fcs.append(target_moudle)
        else:
            target_ln_name.append(target_moudle_name)
            target_ln.append(target_moudle)
    
    no_fc_to_quant = True
    for fc_name in target_fc_names:
        if fc_name in quant_config and quant_config[fc_name]["quant"]:
            no_fc_to_quant = False
            break
    
    if no_fc_to_quant:
        return None, None, None, None
    
    return target_fc_names, target_fcs, target_ln_name, target_ln
        
        

@torch.no_grad()
def smooth_ln_fcs(ln, fcs, act_scales, alpha=0.5):
    if not isinstance(fcs, list):
        fcs = [fcs]
    if not isinstance(ln, nn.LayerNorm):
        logging.warning(f"ln's type is {type(ln)}, not nn.LayerNorm. Confirm if this is as expected.")
    for fc in fcs:
        assert isinstance(fc, nn.Linear)
        assert ln.weight.numel() == fc.in_features == act_scales.numel()

    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)
    weight_scales = torch.cat(
        [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0
    )
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)

    scales = (
        (act_scales.pow(alpha) / weight_scales.pow(1 - alpha))
        .clamp(min=1e-5)
        .to(device)
        .to(dtype)
    )

    ln.weight.div_(scales)
    if hasattr(ln, 'bias') and ln.bias is not None:
        ln.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))

@torch.no_grad()
def smooth_model(model,
                 per_layer_quant_strategy,
                 ln_linear_map, 
                 scales, 
                 alpha):
    
    processed_fc_names = list()
    for moudle_name, module in model.named_modules():
        if moudle_name not in per_layer_quant_strategy:
            continue
        if processed_fc_names in processed_fc_names:
            continue
        assert isinstance(module, nn.Linear)
        target_fc_names, target_fcs, target_ln_name, target_ln = get_related_fcs_ln(moudle_name, ln_linear_map)
        if target_fc_names is None:
            continue
        linear_input_scale = smooth_ln_fcs[moudle_name]
        logging.info(f"Smoothing the following FCs:\n{', '.join(target_fc_names)}\nwith Norm: {target_ln_name}")
        smooth_ln_fcs(target_ln, target_fcs, linear_input_scale, alpha)


@torch.no_grad()
def get_static_decoder_layer_scales(model_wapper,
                                    model,
                                    cali_data,
                                    per_layer_quant_strategy,
                                    processor_input_map,
                                    ):
    
    visual_model = model_wapper.get_visual_model(model)
    ori_device = next(visual_model.parameters()).device
    device = get_best_device()
    visual_model.to(device)
    visual_model.eval()
    
    act_dict = defaultdict(dict)

    def stat_io_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        if name not in act_dict or "input" not in act_dict[name]:
            act_dict[name]["input"] = x.detach().abs().max().item()
        else:
            act_dict[name]["input"] = max(
                act_dict[name]["input"], x.detach().abs().max().item()
            )
        if isinstance(y, tuple):
            y = y[0]
        if name not in act_dict or "output" not in act_dict[name]:
            act_dict[name]["output"] = y.detach().abs().max().item()
        else:
            act_dict[name]["output"] = max(
                act_dict[name]["output"], y.detach().abs().max().item()
            )

    hooks = []
    for name, m in model.named_modules():
        if name not in per_layer_quant_strategy:
            continue
        
        if isinstance(m, torch.nn.Linear):
            hooks.append(m.register_forward_hook(functools.partial(stat_io_hook, name=name)))

    pbar = tqdm(range(len(cali_data)))
    for i in pbar:
        preprocessed_data = cali_data[i]
        input_dict = dict()
        for name_in_processor, data in preprocessed_data.items():
            if name_in_processor not in processor_input_map: 
                continue
            if isinstance(data, torch.Tensor):
                data = data.to(device)
            name_in_input = processor_input_map[name_in_processor]
            input_dict[name_in_input] = data
        with torch.inference_mode():
            im_emb = visual_model(**input_dict)
        
        mean_scale = np.mean([v["input"] for v in act_dict.values()])
        pbar.set_description(f"Mean input scale: {mean_scale:.2f}")
    
    for hook in hooks:
        hook.remove()

    visual_model.to(ori_device)
    return act_dict

def quant_linear_layers(model,
                        per_layer_quant_strategy,
                        act_io_range,
                        ):    
    
    pbar = tqdm(per_layer_quant_strategy.keys())
    for linear_name in pbar:
        quant_strategy = per_layer_quant_strategy[linear_name]
        linear = get_op_by_name(model, linear_name)
        quanted_linear = Fused_StaticQuant_IGEMM_Dequant_AddBias_Linear.from_float(linear,
                                                                                   act_io_range[linear_name]["input"],
                                                                                   act_io_range[linear_name]["output"],
                                                                                   quant_strategy["act_quant_bit"],
                                                                                   quant_strategy["gemm_out_requant_bit"])
        set_op_by_name(model, linear_name, quanted_linear)
        