# -- coding:UTF-8 --
# ---------------------------------------------------------------------------
# DeepPruner: Learning Efficient Stereo Matching via Differentiable PatchMatch
#
# Copyright (c) 2019 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project. 
#
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Written by Shivam Duggal
# ---------------------------------------------------------------------------

from __future__ import print_function
from models.submodules3d import MinDisparityPredictor, MaxDisparityPredictor, CostAggregator
from models.submodules2d import RefinementNet
from models.submodules import SubModule, conv_relu, convbn_2d_lrelu, convbn_3d_lrelu
from models.utils import SpatialTransformer, UniformSampler
from models.patch_match import PatchMatch
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.config import config as args


def loadNetWorks(args=None):
    return DeepPruner()


class DeepPruner(SubModule):
    def __init__(self):
        super(DeepPruner, self).__init__()
        self.scale = args.cost_aggregator_scale
        self.max_disp = args.max_disp // self.scale
        self.mode = args.mode

        self.patch_match_args = args.patch_match_args
        self.patch_match_sample_count = self.patch_match_args.sample_count
        self.patch_match_iteration_count = self.patch_match_args.iteration_count
        self.patch_match_propagation_filter_size = self.patch_match_args.propagation_filter_size
        
        self.post_CRP_sample_count = args.post_CRP_sample_count
        self.post_CRP_sampler_type = args.post_CRP_sampler_type
        hourglass_inplanes = args.hourglass_inplanes

        #   refinement input features are composed of:
        #                                       left image low level features +
        #                                       CA output features + CA output disparity

        if self.scale == 8:
            from models.feature_extractor_fast import feature_extraction
            refinement_inplanes_1 = args.feature_extractor_refinement_level_1_outplanes + 1
            self.refinement_net1 = RefinementNet(refinement_inplanes_1)
        else:
            from models.feature_extractor_best import feature_extraction

        refinement_inplanes = args.feature_extractor_refinement_level_outplanes + self.post_CRP_sample_count + 2 + 1
        self.refinement_net = RefinementNet(refinement_inplanes)

        # cost_aggregator_inplanes are composed of:  
        #                            left and right image features from feature_extractor (ca_level) + 
        #                            features from min/max predictors + 
        #                            min_disparity + max_disparity + disparity_samples

        cost_aggregator_inplanes = 2 * (args.feature_extractor_ca_level_outplanes +
                                        self.patch_match_sample_count + 2) + 1
        self.cost_aggregator = CostAggregator(cost_aggregator_inplanes, hourglass_inplanes)

        self.feature_extraction = feature_extraction()
        self.min_disparity_predictor = MinDisparityPredictor(hourglass_inplanes)  # hourglass_inplanes=16
        self.max_disparity_predictor = MaxDisparityPredictor(hourglass_inplanes)
        self.spatial_transformer = SpatialTransformer()
        self.patch_match = PatchMatch(self.patch_match_propagation_filter_size)
        self.uniform_sampler = UniformSampler()

        # Confidence Range Predictor(CRP) input features are composed of:  
        #                            left and right image features from feature_extractor (ca_level) + 
        #                            disparity_samples

        CRP_feature_count = 2 * args.feature_extractor_ca_level_outplanes + 1
        self.dres0 = nn.Sequential(convbn_3d_lrelu(CRP_feature_count, 64, 3, 1, 1),
                                   convbn_3d_lrelu(64, 32, 3, 1, 1))

        self.dres1 = nn.Sequential(convbn_3d_lrelu(32, 32, 3, 1, 1),
                                   convbn_3d_lrelu(32, hourglass_inplanes, 3, 1, 1))

        self.min_disparity_conv = conv_relu(1, 1, 5, 1, 2)  # in_planes, out_planes, kernel_size, stride, pad, bias=True
        self.max_disparity_conv = conv_relu(1, 1, 5, 1, 2)
        self.ca_disparity_conv = conv_relu(1, 1, 5, 1, 2)

        self.ca_features_conv = convbn_2d_lrelu(self.post_CRP_sample_count + 2,
                                          self. post_CRP_sample_count + 2, 5, 1, 2, dilation=1, bias=True)
        self.min_disparity_features_conv = convbn_2d_lrelu(self.patch_match_sample_count + 2,
                                                     self.patch_match_sample_count + 2, 5, 1, 2, dilation=1, bias=True)
        self.max_disparity_features_conv = convbn_2d_lrelu(self.patch_match_sample_count + 2,
                                                     self.patch_match_sample_count + 2, 5, 1, 2, dilation=1, bias=True)

        self.weight_init()

    def generate_search_range(self, left_input, sample_count, stage,
                              input_min_disparity=None, input_max_disparity=None):
        """
        Description:    Generates the disparity search range depending upon the stage it is called.
                    If stage is "pre" (Pre-PatchMatch and Pre-ConfidenceRangePredictor), the search range is
                    the entire disparity search range.
                    If stage is "post" (Post-ConfidenceRangePredictor), then the ConfidenceRangePredictor search range
                    is adjusted for maximum efficiency.
        Args:
            :left_input: Left Image Features
            :sample_count: number of samples to be generated from the search range. Used to adjust the search range.
            :stage: "pre"(Pre-PatchMatch) or "post"(Post-ConfidenceRangePredictor)
            :input_min_disparity (default:None): ConfidenceRangePredictor disparity lowerbound (for stage=="post")
            :input_max_disparity (default:None): ConfidenceRangePredictor disparity upperbound (for stage=="post")

        Returns:
            :min_disparity: Lower bound of disparity search range
            :max_disparity: Upper bound of disaprity search range.
        """

        device = left_input.get_device()
        if stage == "pre":
            min_disparity = torch.zeros((left_input.size()[0], 1, left_input.size()[2], left_input.size()[3]),
                                        device=device)
            max_disparity = torch.zeros((left_input.size()[0], 1, left_input.size()[2], left_input.size()[3]),
                                        device=device) + self.max_disp

        else:
            min_disparity1 = torch.min(input_min_disparity, input_max_disparity)
            max_disparity1 = torch.max(input_min_disparity, input_max_disparity)

            # if (max_disparity1 - min_disparity1) > sample_count:
            #     sample uniformly "sample_count" number of samples from (min_disparity1, max_disparity1)
            # else:
            #     stretch min_disparity1 and max_disparity1 such that (max_disparity1 - min_disparity1) == sample_count

            # 这里的操作，实际上是让：min_disparity<-->max_disparity之间相差sample_count个视差。分两种情形：
            # 1. min_disparity1<-->max_disparity1之间大于sample_count个视差，则维持不变（后续会均匀采样吧）；
            # 2. min_disparity1<-->max_disparity1之间小于sample_count个视差，则减小min_disparity1，增大max_disparity1。使之
            # 相差sample_count个视差。
            min_disparity = torch.clamp(min_disparity1 - torch.clamp((
                sample_count - max_disparity1 + min_disparity1), min=0) / 2.0, min=0, max=self.max_disp)
            max_disparity = torch.clamp(max_disparity1 + torch.clamp(
                sample_count - max_disparity1 + min_disparity, min=0), min=0, max=self.max_disp)

        return min_disparity, max_disparity

    def generate_disparity_samples(self, left_input, right_input, min_disparity,
                                   max_disparity, sample_count=12, sampler_type="patch_match"):
        """
        Description:    Generates "sample_count" number of disparity samples from the
                                                            search range (min_disparity, max_disparity)
                        Samples are generated either uniformly from the search range
                                                            or are generated using PatchMatch.

        Args:
            :left_input: Left Image features.
            :right_input: Right Image features.
            :min_disparity: LowerBound of the disaprity search range.
            :max_disparity: UpperBound of the disparity search range.
            :sample_count (default:12): Number of samples to be generated from the input search range.
            :sampler_type (default:"patch_match"): samples are generated either using
                                                                    "patch_match" or "uniform" sampler.
        Returns:
            :disparity_samples:
        """
        if sampler_type == "patch_match":
            disparity_samples = self.patch_match(left_input, right_input, min_disparity,
                                                 max_disparity, sample_count, self.patch_match_iteration_count)
        else:
            disparity_samples = self.uniform_sampler(min_disparity, max_disparity, sample_count)  # [B, number_of_samples, H, W]

        disparity_samples = torch.cat((torch.floor(min_disparity), disparity_samples, torch.ceil(max_disparity)),
                                      dim=1).long()
        return disparity_samples

    def cost_volume_generator(self, left_input, right_input, disparity_samples):
        """
        Description: Generates cost-volume using left image features, disaprity samples
                                                            and warped right image features.
        Args:
            :left_input: Left Image fetaures
            :right_input: Right Image features
            :disparity_samples: Disaprity samples   # [B, n, H, W]

        Returns:
            :cost_volume:
            :disaprity_samples:
            :left_feature_map:
        """
        # warped右图[B, C, n, H, W]。左图[B, C, n, H, W]
        right_feature_map, left_feature_map = self.spatial_transformer(left_input,
                                                                       right_input, disparity_samples)
        # [B, n, H, W] -> [B, 1, n, H, W]
        disparity_samples = disparity_samples.unsqueeze(1).float()
        # 左特征图、warped的右特征图、视差样本：[B, C, n, H, W][B, C, n, H, W][B, 1, n, H, W] = [B, 2C+1, n, H, W]
        cost_volume = torch.cat((left_feature_map, right_feature_map, disparity_samples), dim=1)

        return cost_volume, disparity_samples, left_feature_map  # [B, 2C+1, n, H, W], [B, 1, n, H, W], [B, C, n, H, W]

    def confidence_range_predictor(self, cost_volume, disparity_samples):
        """
        Description:    The original search space for all pixels is identical. However, in practice, for each
                        pixel, the highly probable disparities lie in a narrow region. Using the small subset
                        of disparities estimated from the PatchMatch stage, we have sufficient information to
                        predict the range in which the true disparity lies. We thus exploit a confidence range
                        prediction network to adjust the search space for each pixel.

        Args:
            :cost_volume: Input Cost-Volume # [B, 2C+1, n, H, W]: 左特征图、warped的右特征图、视差样本
            :disparity_samples: Initial Disparity samples. [B, 1, n, H, W]

        Returns:
            :min_disparity: ConfidenceRangePredictor disparity lowerbound
            :max_disparity: ConfidenceRangePredictor disparity upperbound
            :min_disparity_features: features from ConfidenceRangePredictor-Min
            :max_disparity_features: features from ConfidenceRangePredictor-Max
        """
        # cost-volume bottleneck layers
        cost_volume = self.dres0(cost_volume)
        cost_volume = self.dres1(cost_volume)

        min_disparity, min_disparity_features = self.min_disparity_predictor(cost_volume,  # [B, 2C+1, n, H, W]
                                                                             disparity_samples.squeeze(1))  # [B, n, H, W]

        max_disparity, max_disparity_features = self.max_disparity_predictor(cost_volume,
                                                                             disparity_samples.squeeze(1))

        min_disparity = self.min_disparity_conv(min_disparity)
        max_disparity = self.max_disparity_conv(max_disparity)
        min_disparity_features = self.min_disparity_features_conv(min_disparity_features)
        max_disparity_features = self.max_disparity_features_conv(max_disparity_features)

        return min_disparity, max_disparity, min_disparity_features, max_disparity_features

    def forward(self, left_input, right_input):
        """
        DeepPruner
        Description: DeepPruner: Learning Efficient Stereo Matching via Differentiable PatchMatch

        Args:
            :left_input: Left Stereo Image
            :right_input: Right Stereo Image
        Returns:
            outputs depend of args.mode ("evaluation or "training"), and
            also on args.cost_aggregator_scale (8 or 4)

            All possible outputs can be:
            :refined_disparity_1: DeepPruner disparity output after Refinement1 stage.
                                                                s (only when args.cost_aggregator_scale==8)
            :refined_disparity: DeepPruner disparity output after Refinement stage.
            :ca_disparity: DeepPruner disparity output after 3D-Cost Aggregation stage.
            :max_disparity: DeepPruner disparity by Confidence Range Predictor (Max)
            :min_disparity: DeepPruner disparity by Confidence Range Predictor (Min)

        """

        # 1. 特征提取===================================================================================================
        if self.scale == 8:
            left_spp_features, left_low_level_features, left_low_level_features_1 = self.feature_extraction(left_input)
            right_spp_features, right_low_level_features, _ = self.feature_extraction(right_input)
        else:
            left_spp_features, left_low_level_features = self.feature_extraction(left_input)
            right_spp_features, right_low_level_features = self.feature_extraction(right_input)

        # 2. 第一次 Patch Match========================================================================================
        # ##2.1初始的search range包含整个视差范围：min_disparity[]=0, max_disparity[]=args.max_disp // self.scale
        min_disparity, max_disparity = self.generate_search_range(  # [B, 1, H, W]
            left_spp_features,
            sample_count=self.patch_match_sample_count, stage="pre")

        disparity_samples = self.generate_disparity_samples(  # 这里面已经进行了iteration_count=2次迭代取最优样本的操作
            left_spp_features,
            right_spp_features, min_disparity, max_disparity,
            sample_count=self.patch_match_sample_count, sampler_type="patch_match")
        # [B, 2C+1, n, H, W], [B, 1, n, H, W], [B, C, n, H, W]
        cost_volume, disparity_samples, _ = self.cost_volume_generator(left_spp_features,
                                                                       right_spp_features,
                                                                       disparity_samples)

        min_disparity, max_disparity, min_disparity_features, max_disparity_features = \
            self.confidence_range_predictor(cost_volume, disparity_samples)  # [B, 2C+1, n, H, W], [B, 1, n, H, W]

        # 3. 第二次 Patch Match=========================================================================================
        stretched_min_disparity, stretched_max_disparity = self.generate_search_range(
            left_spp_features,
            sample_count=self.post_CRP_sample_count, stage='post',
            input_min_disparity=min_disparity, input_max_disparity=max_disparity)

        disparity_samples = self.generate_disparity_samples(  # [B, n, H, W]
            left_spp_features,
            right_spp_features, stretched_min_disparity, stretched_max_disparity,
            sample_count=self.post_CRP_sample_count, sampler_type=self.post_CRP_sampler_type)
        # [B, 2C+1, n, H, W], [B, 1, n, H, W], [B, C, n, H, W]
        cost_volume, disparity_samples, expanded_left_feature_map = self.cost_volume_generator(
            left_spp_features,
            right_spp_features,
            disparity_samples)

        # 4. 3D代价聚合 ================================================================================================
        min_disparity_features = min_disparity_features.unsqueeze(2).expand(-1, -1,
                                                                            expanded_left_feature_map.size()[2], -1, -1)
        max_disparity_features = max_disparity_features.unsqueeze(2).expand(-1, -1,
                                                                            expanded_left_feature_map.size()[2], -1, -1)

        cost_volume = torch.cat((cost_volume, min_disparity_features, max_disparity_features), dim=1)
        ca_disparity, ca_features = self.cost_aggregator(cost_volume, disparity_samples.squeeze(1))  # [B, 2C+1, n, H, W], [B, 1, n, H, W]->[B, n, H, W]

        # 4. Refinement ===============================================================================================
        ca_disparity = F.interpolate(ca_disparity * 2, scale_factor=(2, 2), mode='bilinear')
        ca_features = F.interpolate(ca_features, scale_factor=(2, 2), mode='bilinear')
        ca_disparity = self.ca_disparity_conv(ca_disparity)
        ca_features = self.ca_features_conv(ca_features)

        refinement_net_input = torch.cat((left_low_level_features, ca_features, ca_disparity), dim=1)
        refined_disparity = self.refinement_net(refinement_net_input, ca_disparity)

        # 我设置这一句的align_corners=True，EPE取得了0.017正向提升
        refined_disparity = F.interpolate(refined_disparity * 2, scale_factor=(2, 2), mode='bilinear')

        if self.scale == 8:
            refinement_net_input = torch.cat((left_low_level_features_1, refined_disparity), dim=1)
            refined_disparity_1 = self.refinement_net1(refinement_net_input, refined_disparity)

        if self.mode == 'evaluation':
            if self.scale == 8:
                refined_disparity_1 = F.interpolate(refined_disparity_1 * 2, scale_factor=(2, 2),
                                                    mode='bilinear').squeeze(1)
                return refined_disparity_1
            return refined_disparity.squeeze(1)  # [B, H, W]

        min_disparity = F.interpolate(min_disparity * self.scale, scale_factor=(self.scale, self.scale),
                                      mode='bilinear').squeeze(1)
        max_disparity = F.interpolate(max_disparity * self.scale, scale_factor=(self.scale, self.scale),
                                      mode='bilinear').squeeze(1)
        ca_disparity = F.interpolate(ca_disparity * (self.scale // 2),
                                     scale_factor=((self.scale // 2), (self.scale // 2)), mode='bilinear').squeeze(1)

        if self.scale == 8:
            refined_disparity = F.interpolate(refined_disparity * 2, scale_factor=(2, 2), mode='bilinear').squeeze(1)
            refined_disparity_1 = F.interpolate(refined_disparity_1 * 2,
                                                scale_factor=(2, 2), mode='bilinear').squeeze(1)

            return refined_disparity_1, refined_disparity, ca_disparity, max_disparity, min_disparity

        return refined_disparity.squeeze(1), ca_disparity, max_disparity, min_disparity
