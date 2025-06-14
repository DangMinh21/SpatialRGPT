from transformers import PretrainedConfig


class LlavaConfig(PretrainedConfig):
    model_type = "llava"

    def __init__(
        self,
        llm_cfg=None,
        vision_tower_cfg=None,
        mm_projector_cfg=None,
        region_extractor_cfg=None,
        architectures=None,
        enable_region=None,
        enable_depth=None,
        resume_path=None,
        hidden_size=None,
        mm_hidden_size=None,
        image_aspect_ratio=None,
        num_video_frames=None,
        fps=None,
        mm_vision_select_layer=None,
        mm_vision_select_feature=None,
        mm_use_im_start_end=False,
        mm_use_im_patch_token=True,
        mm_projector_lr=None,
        vision_resolution=None,
        interpolate_mode=None,
        s2=None,
        s2_scales=None,
        s2_max_split_size=None,
        # add config for region_feature_extractor
        
        enable_region_enhancer=None,
        region_enhancer_cfg=None,
        enable_region_classifier=None,
        region_classifier_cfg=None,
        **kwargs
    ):
        super().__init__()
        self.architectures = architectures
        self.llm_cfg = llm_cfg
        self.vision_tower_cfg = vision_tower_cfg
        self.mm_projector_cfg = mm_projector_cfg
        self.region_extractor_cfg = region_extractor_cfg
        self.resume_path = resume_path

        self.enable_region = enable_region
        self.enable_depth = enable_depth
        self.hidden_size = hidden_size
        self.mm_hidden_size = mm_hidden_size
        self.image_aspect_ratio = image_aspect_ratio
        self.num_video_frames = num_video_frames
        self.fps = fps
        self.mm_vision_select_layer = mm_vision_select_layer
        self.mm_vision_select_feature = mm_vision_select_feature
        self.mm_use_im_start_end = mm_use_im_start_end
        self.mm_use_im_start_end = mm_use_im_start_end
        self.mm_use_im_patch_token = mm_use_im_patch_token
        self.mm_projector_lr = mm_projector_lr
        self.vision_resolution = vision_resolution
        self.interpolate_mode = interpolate_mode
        self.s2 = s2
        self.s2_scales = s2_scales
        self.s2_max_split_size = s2_max_split_size

        # add config for region_feature_extractor
        self.enable_region_enhencer = enable_region_enhancer
        self.region_enhancer_cfg = region_enhancer_cfg
        self.enable_region_classifier =enable_region_classifier
        self.region_classifier_cfg = region_classifier_cfg