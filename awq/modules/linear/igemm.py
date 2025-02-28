import torch
from torch import nn

# NOTE(ningpeiyang): `maxM` and `maxFL` is defiend by chip design
def find_best_approximation(fp, maxM=37767, maxFL=38):
    min_diff = float('inf')
    best_m = None
    best_fl = None

    for fl in range(maxFL+1):
        m = round(fp * (2 ** fl))
        if 0 <= m <= 32767:
            approximation = m / (2 ** fl)
            diff = abs(fp - approximation)

            if diff < min_diff:
                min_diff = diff
                best_m = m
                best_fl = fl

    assert best_m is not None and best_fl is not None
    return best_m, best_fl

def quantize_activation(tensor: torch.Tensor, act_scale: torch.Tensor, quant_bit: int) -> torch.Tensor:
    if tensor.dtype != torch.float16 and tensor.dtype != torch.bfloat16:
        raise ValueError("Input tensor must be of type torch.float16 or torch.bfloat16")

    quantized = torch.round(tensor / act_scale).clamp(-2**(quant_bit - 1), 2**(quant_bit - 1) - 1)

    if quant_bit == 8:
        quantized = quantized.to(torch.int8)
    elif quant_bit == 16:
        quantized = quantized.to(torch.int16)
    else:
        pass # ignore

    return quantized

def dequant_activation(tensor: torch.Tensor, dequant_scale: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    if tensor.dtype not in [torch.int8, torch.int16, torch.int32]:
        raise ValueError("Input tensor must be of type int8, int16, or int32")
    
    if out_dtype not in [torch.float16, torch.bfloat16]:
        raise ValueError("Unsupported output data type. Choose either float16 or bfloat16.")
    
    return tensor.to(out_dtype) * dequant_scale


def igemm(activation: torch.Tensor, 
          weight: torch.Tensor, 
          bias: torch.Tensor,
          gemm_out_dtype: str, 
          M: torch.Tensor,
          FL: torch.Tensor) -> torch.Tensor:
    if activation.dtype != torch.int8 and activation.dtype != torch.int16:
        raise ValueError("Activation tensor must be of type torch.int8 or torch.int16")
    
    if weight.dtype != torch.int8:
        raise ValueError("Weight tensor must be of type torch.int8")
    
    # NOTE(ningpeiyang): Up to torch 2.5.1, Int Linear in GPU has NOT been implemented yet,
    # we have to use sgemm to simulate igemm
    activation_f = activation.float()
    weight_f = weight.float().T
    out_f = torch.round(activation_f @ weight_f)
    out_int32 = out_f.to(torch.int32)
    if bias is not None:
        out_int32 += bias
    
    # NOTE(ningpeiyang): `out_int32 * M` may overflow in int32
    # the chip support up to 47bit for the result of `out_int32 * M`
    # we simulate this design
    out_requant = out_int32.to(torch.int64) * M
    out_requant = torch.clamp(out_requant, -2**46, 2**46 - 1) >> FL 
    
    if gemm_out_dtype == "int32":
        # NOTE(ningpeiyang): to avoid inf or nan in fp16 dequant
        # When quantizing, I have tried to controlling the value 
        # within the fp16 value range by requant, but clamp() is 
        # still necessary
        out = torch.clamp(out_requant, min=-65504, max=65504).to(torch.int32)
    elif gemm_out_dtype == "int16":
        out = torch.clamp(out_requant, min=-32768, max=32767).to(torch.int16)
    elif gemm_out_dtype == "int8":
        out = torch.clamp(out_requant, min=-128, max=127).to(torch.int8)
    else:
        raise RuntimeError(f"Unknown gemm_out_dtype:{gemm_out_dtype}")
    return out
    

class Fused_StaticQuant_IGEMM_Dequant_AddBias_Linear(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool,
                 act_quant_bit: int,
                 gemm_out_dtype: str,
                 ) -> None:
        super().__init__()
        
        assert gemm_out_dtype in ["int32", "int16", "int8"]
        
        self.in_features = in_features
        self.out_features = out_features
        self.act_quant_bit = act_quant_bit
        self.gemm_out_dtype = gemm_out_dtype
        
        self.register_buffer(
            "weight",
            torch.empty(
                self.out_features,
                self.in_features,
                dtype=torch.int8,
                requires_grad=False,
            ),
        )
        if bias:
            self.register_buffer(
                "bias",
                torch.empty(
                    (self.out_features,), dtype=torch.int32, requires_grad=False
                ),
            )
        else:
            self.register_buffer(
                "bias",
                None,
            )
        
        self.register_buffer(
            "act_scale",
            torch.empty(
                (1,), dtype=torch.float16, requires_grad=False
            ),
        )
        
        self.register_buffer(
            "M",
            torch.empty(
                (1,), dtype=torch.int16, requires_grad=False
            ),
        )
        
        self.register_buffer(
            "FL",
            torch.empty(
                (1,), dtype=torch.int8, requires_grad=False
            ),
        )
        
        self.register_buffer(
            "dequant_scale",
            torch.empty(
                (1,), dtype=torch.float16, requires_grad=False
            ),
        )


    def forward(self, activation: torch.Tensor) -> torch.Tensor:
        # if torch.isnan(activation).any() or torch.isinf(activation).any():
        #     import pdb
        #     pdb.set_trace()
        #     raise ValueError(f"NaN or Inf detected in layer {self.__class__.__name__}, activation")
        
        activation_quanted = quantize_activation(activation, self.act_scale, self.act_quant_bit)
        gemm_out = igemm(activation_quanted, self.weight, self.bias, self.gemm_out_dtype, self.M, self.FL)
        out = dequant_activation(gemm_out, self.dequant_scale, activation.dtype)
        
        # if torch.isnan(out).any() or torch.isinf(out).any():
        #     import pdb
        #     pdb.set_trace()
        #     raise ValueError(f"NaN or Inf detected in layer {self.__class__.__name__}, out")
        
        return out
    
    
    @staticmethod
    def from_float(
        fp16_linear,
        act_in_range,
        act_out_range,
        act_quant_bit,
        gemm_out_dtype
    ):
        assert isinstance(fp16_linear, torch.nn.Linear)
        assert act_quant_bit in [8, 16]
        assert gemm_out_dtype in ["int32", "int16", "int8"]
        quanted_linear = Fused_StaticQuant_IGEMM_Dequant_AddBias_Linear(in_features=fp16_linear.in_features,
                                                                       out_features=fp16_linear.out_features,
                                                                       bias=fp16_linear.bias is not None,
                                                                       act_quant_bit=act_quant_bit,
                                                                       gemm_out_dtype=gemm_out_dtype)
        
        # NOTE(ningpeiyang）: currently support W8A8 or W8A16 only
        weight_quant_bit = 8
        weight_max_int = 2**(weight_quant_bit - 1) - 1
        abs_max = torch.max(torch.abs(fp16_linear.weight)).float().item()
        weight_scale = abs_max / weight_max_int
        quantized_weight = torch.clamp(torch.round(fp16_linear.weight / weight_scale), -weight_max_int, weight_max_int).to(torch.int8)
        quanted_linear.weight = quantized_weight

        act_max_int = 2**(act_quant_bit - 1) - 1
        act_scale = act_in_range / act_max_int
        quanted_linear.act_scale = torch.tensor([act_scale], dtype=torch.half)
        
        qunatized_bias = torch.round(fp16_linear.bias.float() / (weight_scale * act_scale)).to(torch.int32)
        quanted_linear.bias = qunatized_bias
        
        max_act_out_int = int(round(act_out_range / (act_scale * weight_scale)))
        if gemm_out_dtype == "int32":
            HALF_MAX = 65504
            if max_act_out_int <= HALF_MAX:
                M = 1
                FL = 0
                requant_scale = 1.0
            else:
                requant_scale_fp = HALF_MAX / max_act_out_int
                M, FL = find_best_approximation(requant_scale_fp)
                requant_scale = M / 2**FL
        elif gemm_out_dtype == "int16":
            INT16_MAX = 32767
            if max_act_out_int <= INT16_MAX:
                M = 1
                FL = 0
                requant_scale = 1.0
            else:
                requant_scale_fp = INT16_MAX / max_act_out_int
                M, FL = find_best_approximation(requant_scale_fp)
                requant_scale = M / 2**FL
        elif gemm_out_dtype == "int8":
            INT8_MAX = 127
            if max_act_out_int <= INT8_MAX:
                M = 1
                FL = 0
            else:
                requant_scale_fp = INT8_MAX / max_act_out_int
                requant_scale_fp = INT8_MAX / max_act_out_int
                M, FL = find_best_approximation(requant_scale_fp)
                requant_scale = M / 2**FL
        else:
            raise RuntimeError(f"Unsupport requant type {gemm_out_dtype}")
        
        quanted_linear.M = torch.tensor([M], dtype=torch.int16)
        quanted_linear.FL = torch.tensor([FL], dtype=torch.int8)
        
        dequant_scale = act_scale * weight_scale / requant_scale
        quanted_linear.dequant_scale = torch.tensor([dequant_scale], dtype=torch.half)
        
        quanted_linear.to(fp16_linear.weight.device)
              
        return quanted_linear
