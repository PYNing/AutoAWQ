from awq.quantize.visual_quant_utils import (get_cali_data, 
                                             get_act_scales, 
                                             smooth_model,
                                             get_static_decoder_layer_scales,
                                             quant_linear_layers)
from awq.utils.module import get_visual_per_layer_quant_strategy

class VisualQuantizer:
    def __init__(
        self,
        model_warpper,
        model,
        processor,
        visual_calib_data,
        visual_calib_subset,
        visual_calib_split,
        visual_image_column,
        visual_quant_config,
        visual_smooth_quant_alpha,
        visual_max_calib_samples,
        visual_dataset_shuffle_seed,
        export_compatible,
    ) -> None:
        self.model_warpper = model_warpper
        self.model = model
        self.quant_config = visual_quant_config
        self.processor = processor
        self.calib_dataset_name = visual_calib_data
        self.calib_subset = visual_calib_subset
        self.calib_split = visual_calib_split
        self.image_column = visual_image_column
        self.max_calib_samples = visual_max_calib_samples
        self.smooth_quant_alpha = visual_smooth_quant_alpha
        self.dataset_shuffle_seed = visual_dataset_shuffle_seed
        self.export_compatible = export_compatible
        
    def quantize(
        self,
    ) -> None:
        # Prepare
        visual_layers_prefix = self.model_warpper.get_visual_layers_prefix()
        processor_input_map = self.model_warpper.get_processor_input_map()
        fcs_ln_group_map = self.model_warpper.get_fcs_ln_group_map()
        per_layer_quant_strategy = get_visual_per_layer_quant_strategy(self.model, visual_layers_prefix, self.quant_config)
        
        cali_data = get_cali_data(self.calib_dataset_name, 
                                  self.calib_subset, 
                                  self.calib_split, 
                                  self.image_column,
                                  self.processor,
                                  self.max_calib_samples,
                                  self.dataset_shuffle_seed)
        
        # Step1: get max values of activation per channel 
        act_scales_for_smooth = get_act_scales(self.model_warpper,
                                               self.model,
                                               per_layer_quant_strategy, 
                                               cali_data, 
                                               processor_input_map)
        
        # Step2: smooth
        smooth_model(self.model,
                     per_layer_quant_strategy,
                     fcs_ln_group_map,
                     act_scales_for_smooth,
                     self.smooth_quant_alpha)

        # Step3: calibration
        act_io_range = get_static_decoder_layer_scales(self.model_warpper,
                                                       self.model,
                                                       cali_data,
                                                       per_layer_quant_strategy,
                                                       processor_input_map)

        # Step4: generate quanted model
        quant_linear_layers(self.model,
                            per_layer_quant_strategy,
                            act_io_range,
                            )