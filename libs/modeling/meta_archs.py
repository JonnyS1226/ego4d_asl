import math
import copy

import torch
from torch import nn
from torch.nn import functional as F

from .models import register_meta_arch, make_backbone, make_neck, make_generator
from .blocks import MaskedConv1D, Scale, LayerNorm
from .losses import ctr_diou_loss_1d, sigmoid_focal_loss, ctr_giou_loss_1d
from .utils import calc_ious, calc_cls_scores, DeformConv1d, PackedDeformConv1d

from ..utils import batched_nms
from .ml_gcn import LabelGCN, LabelTransformer
# from .roi_align import ROIAlign

def normal_distribution(x, mu=0, sigma=1):
    return (-(x - mu)**2 / (2 * sigma**2)).exp()

def multiple_normal_distribution(x, mu, sigma):
    return ((-(x - mu[:1])**2 / (2 * sigma[:1]**2)).exp() + (-(x - mu[1:])**2 / (2 * sigma[1:]**2)).exp()) / 2.0

class PtTransformerClsHead(nn.Module):
    """
    1D Conv heads for classification
    """

    def __init__(
            self,
            input_dim,
            feat_dim,
            num_classes,
            prior_prob=0.01,
            num_layers=3,
            kernel_size=3,
            act_layer=nn.ReLU,
            with_ln=False,
            empty_cls=[],
            detach_feat=False
    ):
        super().__init__()
        self.act = act_layer()
        self.detach_feat = detach_feat

        # build the head
        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()
        for idx in range(num_layers - 1):
            if idx == 0:
                in_dim = input_dim
                out_dim = feat_dim
            else:
                in_dim = feat_dim
                out_dim = feat_dim
            self.head.append(
                MaskedConv1D(
                    in_dim, out_dim, kernel_size,
                    stride=1,
                    padding=kernel_size // 2,
                    bias=(not with_ln)
                )
            )
            if with_ln:
                self.norm.append(
                    LayerNorm(out_dim)
                )
            else:
                self.norm.append(nn.Identity())

        # classifier
        self.cls_head = MaskedConv1D(
            feat_dim, num_classes, kernel_size,
            stride=1, padding=kernel_size // 2
        )

        # use prior in model initialization to improve stability
        # this will overwrite other weight init
        bias_value = -(math.log((1 - prior_prob) / prior_prob))
        torch.nn.init.constant_(self.cls_head.conv.bias, bias_value)

        # a quick fix to empty categories:
        # the weights assocaited with these categories will remain unchanged
        # we set their bias to a large negative value to prevent their outputs
        if len(empty_cls) > 0:
            bias_value = -(math.log((1 - 1e-6) / 1e-6))
            for idx in empty_cls:
                torch.nn.init.constant_(self.cls_head.conv.bias[idx], bias_value)

    def forward(self, fpn_feats, fpn_masks):
        assert len(fpn_feats) == len(fpn_masks)

        # apply the classifier for each pyramid level
        out_logits = tuple()
        for _, (cur_feat, cur_mask) in enumerate(zip(fpn_feats, fpn_masks)):
            if self.detach_feat:
                cur_out = cur_feat.detach()
            else:
                cur_out = cur_feat
            for idx in range(len(self.head)):
                cur_out, _ = self.head[idx](cur_out, cur_mask)
                cur_out = self.act(self.norm[idx](cur_out))
            cur_logits, _ = self.cls_head(cur_out, cur_mask)
            out_logits += (cur_logits,)

        # fpn_masks remains the same
        return out_logits


class PtTransformerRegHead(nn.Module):
    """
    Shared 1D Conv heads for regression
    Simlar logic as PtTransformerClsHead with separated implementation for clarity
    """

    def __init__(
            self,
            input_dim,
            feat_dim,
            fpn_levels,
            num_layers=3,
            kernel_size=3,
            act_layer=nn.ReLU,
            with_ln=False,
            num_bins=16
    ):
        super().__init__()
        self.fpn_levels = fpn_levels
        self.act = act_layer()

        # build the conv head
        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()
        for idx in range(num_layers - 1):
            if idx == 0:
                in_dim = input_dim
                out_dim = feat_dim
            else:
                in_dim = feat_dim
                out_dim = feat_dim
            self.head.append(
                MaskedConv1D(
                    in_dim, out_dim, kernel_size,
                    stride=1,
                    padding=kernel_size // 2,
                    bias=(not with_ln)
                )
            )
            if with_ln:
                self.norm.append(
                    LayerNorm(out_dim)
                )
            else:
                self.norm.append(nn.Identity())

        self.scale = nn.ModuleList()
        for idx in range(fpn_levels):
            self.scale.append(Scale())

        self.offset_head = MaskedConv1D(
            feat_dim, 2 * (num_bins + 1), kernel_size,
            stride=1, padding=kernel_size // 2
        )

    def forward(self, fpn_feats, fpn_masks):
        assert len(fpn_feats) == len(fpn_masks)
        assert len(fpn_feats) == self.fpn_levels

        # apply the classifier for each pyramid level
        out_offsets = tuple()
        for l, (cur_feat, cur_mask) in enumerate(zip(fpn_feats, fpn_masks)):
            cur_out = cur_feat
            for idx in range(len(self.head)):
                cur_out, _ = self.head[idx](cur_out, cur_mask)
                cur_out = self.act(self.norm[idx](cur_out))
            cur_offsets, _ = self.offset_head(cur_out, cur_mask)
            out_offsets += (F.relu(self.scale[l](cur_offsets)),)

        # fpn_masks remains the same
        return out_offsets

@register_meta_arch("LocPointTransformer")
class PtTransformer(nn.Module):
    """
        Transformer based model for single stage action localization
    """
    def __init__(
        self,
        backbone_type,         # a string defines which backbone we use
        fpn_type,              # a string defines which fpn we use
        use_xl,
        backbone_arch,         # a tuple defines # layers in embed / stem / branch
        scale_factor,          # scale factor between branch layers
        input_dim,             # input feat dim
        max_seq_len,           # max sequence length (used for training)
        max_buffer_len_factor, # max buffer size (defined a factor of max_seq_len)
        n_head,                # number of heads for self-attention in transformer
        n_mha_win_size,        # window size for self attention; -1 to use full seq
        embd_kernel_size,      # kernel size of the embedding network
        embd_dim,              # output feat channel of the embedding network
        embd_with_ln,          # attach layernorm to embedding network
        fpn_dim,               # feature dim on FPN
        fpn_with_ln,           # if to apply layer norm at the end of fpn
        fpn_start_level,       # start level of fpn
        head_dim,              # feature dim for head
        regression_range,      # regression range on each level of FPN
        head_num_layers,       # number of layers in the head (including the classifier)
        head_kernel_size,      # kernel size for reg/cls heads
        head_with_ln,          # attache layernorm to reg/cls heads
        use_abs_pe,            # if to use abs position encoding
        use_rel_pe,            # if to use rel position encoding
        num_classes,           # number of action classes
        train_cfg,             # other cfg for training
        test_cfg,               # other cfg for testing
    ):
        super().__init__()
        # re-distribute params to backbone / neck / head
        self.fpn_strides = [scale_factor**i for i in range(
            fpn_start_level, backbone_arch[-1]+1
        )]
        self.reg_range = regression_range
        assert len(self.fpn_strides) == len(self.reg_range)
        self.scale_factor = scale_factor
        # #classes = num_classes + 1 (background) with last category as background
        # e.g., num_classes = 10 -> 0, 1, ..., 9 as actions, 10 as background
        self.num_classes = num_classes

        # check the feature pyramid and local attention window size
        self.max_seq_len = max_seq_len
        if isinstance(n_mha_win_size, int):
            self.mha_win_size = [n_mha_win_size]*(1 + backbone_arch[-1])
        else:
            assert len(n_mha_win_size) == (1 + backbone_arch[-1])
            self.mha_win_size = n_mha_win_size
        max_div_factor = 1
        for l, (s, w) in enumerate(zip(self.fpn_strides, self.mha_win_size)):
            stride = s * (w // 2) * 2 if w > 1 else s
            if not max_seq_len % stride ==0:
                import ipdb;ipdb.set_trace()
            assert max_seq_len % stride == 0, "max_seq_len must be divisible by fpn stride and window size"
            if max_div_factor < stride:
                max_div_factor = stride
        self.max_div_factor = max_div_factor
        self.use_xl = use_xl

        # training time config
        self.train_center_sample = train_cfg['center_sample']
        assert self.train_center_sample in ['radius', 'none']
        self.train_center_sample_radius = train_cfg['center_sample_radius']
        self.train_loss_weight = train_cfg['loss_weight']
        self.train_cls_prior_prob = train_cfg['cls_prior_prob']
        self.train_dropout = train_cfg['dropout']
        self.train_droppath = train_cfg['droppath']
        self.train_label_smoothing = train_cfg['label_smoothing']
        # ============ dev# ============ dev# ============ dev# ============ dev
        self.t_c_alpha = train_cfg['t_c_alpha']
        self.al_loss_weight = train_cfg['al_loss_weight']
        self.cont_loss_weight = train_cfg['cont_loss_weight']
        self.seg_loss_weight = train_cfg['seg_loss_weight']
        self.queue_size = train_cfg['queue_size']
        self.temperature = train_cfg['temperature']
        self.use_dcn = train_cfg['use_dcn']
        self.dcn_start_layer = train_cfg['dcn_start_layer']
        self.use_us_fpn = train_cfg['use_us_fpn']
        self.length_theta = train_cfg['length_theta']
        self.num_bins = train_cfg["num_bins"]
        self.iou_weight_power = train_cfg["iou_weight_power"]
        # ============ dev# ============ dev# ============ dev# ============ dev

        # test time config
        self.test_pre_nms_thresh = test_cfg['pre_nms_thresh']
        self.test_pre_nms_topk = test_cfg['pre_nms_topk']
        self.test_iou_threshold = test_cfg['iou_threshold']
        self.test_min_score = test_cfg['min_score']
        self.test_max_seg_num = test_cfg['max_seg_num']
        self.test_nms_method = test_cfg['nms_method']
        assert self.test_nms_method in ['soft', 'hard', 'none']
        self.test_duration_thresh = test_cfg['duration_thresh']
        self.test_multiclass_nms = test_cfg['multiclass_nms']
        self.test_nms_sigma = test_cfg['nms_sigma']
        self.test_voting_thresh = test_cfg['voting_thresh']

        # ========= if add ml-gcn
        # self.gcn_t = train_cfg['gcn_t']

        # we will need a better way to dispatch the params to backbones / necks
        # backbone network: conv + transformer
        assert backbone_type in ['convTransformer', 'conv']
        if backbone_type == 'convTransformer':
            self.backbone = make_backbone(
                'convTransformer',
                **{
                    'n_in' : input_dim,
                    'n_embd' : embd_dim,
                    'n_head': n_head,
                    'n_embd_ks': embd_kernel_size,
                    'max_len': max_seq_len,
                    'use_xl': use_xl,
                    'arch' : backbone_arch,
                    't_c_alpha': self.t_c_alpha,
                    'scale_factor' : scale_factor,
                    'with_ln' : embd_with_ln,
                    'attn_pdrop' : 0.0,
                    'proj_pdrop' : self.train_dropout,
                    'path_pdrop' : self.train_droppath,
                    'use_abs_pe' : use_abs_pe,
                    'use_rel_pe' : use_rel_pe,
                    'use_dcn': self.use_dcn,
                    'dcn_start_layer': self.dcn_start_layer,
                }
            )
            self.backbone_m = make_backbone(
                'convTransformer',
                **{
                    'n_in' : input_dim,
                    'n_embd' : embd_dim,
                    'n_head': n_head,
                    'n_embd_ks': embd_kernel_size,
                    'max_len': max_seq_len,
                    'use_xl': use_xl,
                    'arch' : backbone_arch,
                    't_c_alpha': self.t_c_alpha,
                    'scale_factor' : scale_factor,
                    'with_ln' : embd_with_ln,
                    'attn_pdrop' : 0.0,
                    'proj_pdrop' : self.train_dropout,
                    'path_pdrop' : self.train_droppath,
                    'use_abs_pe' : use_abs_pe,
                    'use_rel_pe' : use_rel_pe,
                    'use_dcn': self.use_dcn,
                    'dcn_start_layer': self.dcn_start_layer,
                }
            )
        else:
            self.backbone = make_backbone(
                'conv',
                **{
                    'n_in': input_dim,
                    'n_embd': embd_dim,
                    'n_embd_ks': embd_kernel_size,
                    'arch': backbone_arch,
                    'scale_factor': scale_factor,
                    'with_ln' : embd_with_ln
                }
            )
            self.backbone_m = make_backbone(
                'conv',
                **{
                    'n_in': input_dim,
                    'n_embd': embd_dim,
                    'n_embd_ks': embd_kernel_size,
                    'arch': backbone_arch,
                    'scale_factor': scale_factor,
                    'with_ln' : embd_with_ln
                }
            )
        if isinstance(embd_dim, (list, tuple)):
            embd_dim = sum(embd_dim)

        # fpn network: convs
        assert fpn_type in ['fpn', 'identity']
        self.neck = make_neck(
            fpn_type,
            **{
                'in_channels' : [embd_dim] * (backbone_arch[-1] + 1),
                'out_channel' : fpn_dim,
                'scale_factor' : scale_factor,
                'start_level' : fpn_start_level,
                'with_ln' : fpn_with_ln,
                'use_us_fpn': self.use_us_fpn
            }
        )
        self.neck_m = make_neck(
            fpn_type,
            **{
                'in_channels' : [embd_dim] * (backbone_arch[-1] + 1),
                'out_channel' : fpn_dim,
                'scale_factor' : scale_factor,
                'start_level' : fpn_start_level,
                'with_ln' : fpn_with_ln,
                'use_us_fpn': self.use_us_fpn
            }
        )
        # location generator: points
        self.point_generator = make_generator(
            'point',
            **{
                'max_seq_len' : max_seq_len * max_buffer_len_factor,
                'fpn_strides' : self.fpn_strides,
                'regression_range' : self.reg_range,
                'use_us_fpn': self.use_us_fpn
            }
        )

        # classfication and regerssion heads
        self.cls_head = PtTransformerClsHead(
            fpn_dim, head_dim, self.num_classes,
            kernel_size=head_kernel_size,
            prior_prob=self.train_cls_prior_prob,
            with_ln=head_with_ln,
            num_layers=head_num_layers,
            empty_cls=train_cfg['head_empty_cls']
        )

        self.reg_head = PtTransformerRegHead(
            fpn_dim, head_dim, len(self.fpn_strides),
            kernel_size=head_kernel_size,
            num_layers=head_num_layers,
            with_ln=head_with_ln,
            num_bins=0
        )
        
        # ====== add seg
        self.seg_conv = PtTransformerClsHead(
            fpn_dim, head_dim, self.num_classes,
            kernel_size=head_kernel_size,
            prior_prob=self.train_cls_prior_prob,
            with_ln=head_with_ln,
            num_layers=1,
            empty_cls=train_cfg['head_empty_cls']
        )
        # ====== add seg
        
        self.mu = nn.Parameter(torch.zeros(self.num_classes, 1), requires_grad=True)
        self.sigma = nn.Parameter(torch.ones(self.num_classes, 1), requires_grad=True)
        self.mu_reg_left = nn.Parameter(-torch.ones(self.num_classes, 1)*0.5, requires_grad=True)
        self.sigma_reg_left = nn.Parameter(torch.ones(self.num_classes, 1), requires_grad=True)
        self.mu_reg_right = nn.Parameter(torch.ones(self.num_classes, 1)*0.5, requires_grad=True)
        self.sigma_reg_right = nn.Parameter(torch.ones(self.num_classes, 1), requires_grad=True)
        # self.roi_extractor = ROIAlign(16, 0)

        # maintain an EMA of #foreground to stabilize the loss normalizer
        # useful for small mini-batch training
        self.loss_normalizer = train_cfg['init_loss_norm']
        self.loss_normalizer_momentum = 0.9

    @property
    def device(self):
        # a hacky way to get the device type
        # will throw an error if parameters are on different devices
        return list(set(p.device for p in self.parameters()))[0]

    def forward(self, video_list, ensemble=False, hidden_state=False, is_training=True):
        # batch the video list into feats (B, C, T) and masks (B, 1, T)
        batched_inputs, batched_masks, segmentation_labels = self.preprocessing(video_list, is_training)      # pad to fixed length

        # forward the network (backbone -> neck -> heads)
        feats, masks = self.backbone(batched_inputs, batched_masks)     # List [b,c,ti].  List [b,1,ti]

        fpn_feats, fpn_masks = self.neck(feats, masks)

        # points: List[T x 4] with length = # fpn levels
        points = self.point_generator(fpn_feats)    # 4 means [t, reg_range_left, reg_range_right, fpn_stride]

        # out_offset: List[B, 2, T_i]
        out_offsets = self.reg_head(fpn_feats, fpn_masks)

        # out_cls: List[B, #cls + 1, T_i]       
        out_cls_logits = self.cls_head(fpn_feats, fpn_masks)   

        out_lb_logits = None
        out_rb_logits = None

        # ======= add seg loss ==== 
        out_seg_logits = self.seg_conv(fpn_feats, fpn_masks)
        out_seg_logits = [x.permute(0, 2, 1) for x in out_seg_logits]
        # ======= add seg loss ==== 

        # permute the outputs
        # out_cls: F List[B, #cls, T_i] -> F List[B, T_i, #cls]
        out_cls_logits = [x.permute(0, 2, 1) for x in out_cls_logits]      
        # out_offset: F List[B, 2 (xC), T_i] -> F List[B, T_i, 2 (xC)]      
        out_offsets = [x.permute(0, 2, 1) for x in out_offsets]   
        # fpn_masks: F list[B, 1, T_i] -> F List[B, T_i]
        fpn_masks = [x.squeeze(1) for x in fpn_masks]

        # return loss during training
        if is_training:
            # generate segment/lable List[N x 2] / List[N] with length = B
            assert video_list[0]['segments'] is not None, "GT action labels does not exist"
            assert video_list[0]['labels'] is not None, "GT action labels does not exist"
            gt_segments = [x['segments'].to(self.device) for x in video_list if len(x['labels']) > 0]
            gt_labels = [x['labels'].to(self.device) for x in video_list if len(x['labels']) > 0]            

            # compute the gt labels for cls & reg
            # list of prediction targets

            # ============= get segment level contrastive loss
            # cont_loss = self.get_segment_level_contra_loss(feats[0], gt_segments, gt_labels)
            # cont_loss /= self.loss_normalizer
            # ============= get segment level contrastive loss

            gt_cls_labels, gt_offsets, normal_probs_cls, normal_probs_reg = self.label_points(points, gt_segments, gt_labels)
            # segmentation_labels, _, _, _ = self.label_points(points, gt_segments, gt_labels, for_seg=True)

            # ============= process prompt and fpn_feats 

            # ============= segmentation loss ================ #
            # valid_mask = torch.cat(fpn_masks, dim=1)        # [b, all_points]
            # segmentation_labels = torch.stack(segmentation_labels)             # [b, all_points, C]
            # pos_mask = torch.logical_and((segmentation_labels.sum(-1) > 0), valid_mask)      # [b, all_points]
            # segmentation_target = segmentation_labels[valid_mask]          # [b*allpoints, 1]
            # segmentation_target *= 1 - self.train_label_smoothing
            # segmentation_target += self.train_label_smoothing / (self.num_classes + 1)
            # segmentation_loss = sigmoid_focal_loss(
            #     torch.cat(out_seg_logits, dim=1)[valid_mask],           # [#pos, c]
            #     segmentation_target,
            #     reduction='sum'
            # )
            # segmentation_loss /= self.loss_normalizer
            # segmentation_loss = F.multilabel_soft_margin_loss(
            #     torch.cat(out_cls_logits, dim=1)[valid_mask].sigmoid(),  segmentation_target, reduction='sum'
            # )
            # ============= segmentation loss ================ #

            # compute the cls and reg loss
            losses = self.losses(
                fpn_masks,                      # [b,192\96\48\...] for each layer in fpn
                out_cls_logits, out_offsets,    # 
                gt_cls_labels, gt_offsets,       # 相当于fpn那么多层，每个点有一个gt边界和gt类，同时每一个点有预测的gt边界和gt类，算loss
                label_list=gt_labels,                       # for label involved loss
                normal_probs_cls=normal_probs_cls,
                normal_probs_reg=normal_probs_reg,
                out_importances=None,
                out_start=out_lb_logits, out_end=out_rb_logits
            )
            # ============= other loss
            # loss_ita = loss_ita / (self.loss_normalizer / 2)
            # losses.update({"cont_loss": loss_ita})
            # losses['final_loss'] += loss_ita * self.cont_loss_weight

            # losses.update({"seg loss": segmentation_loss})
            # losses['final_loss'] += segmentation_loss * self.seg_loss_weight

            # losses.update({"contra loss": cont_loss})
            # losses['final_loss'] += cont_loss * self.cont_loss_weight
            # ============= other loss

            return losses
        else:
            # decode the actions (sigmoid / stride, etc)
            results = self.inference(
                video_list, points, fpn_masks,
                out_cls_logits, out_offsets,
                out_lb_logits, out_rb_logits
            )
            if ensemble:
                return video_list, points, fpn_masks, out_cls_logits, out_offsets
            return results

    @torch.no_grad()
    def preprocessing(self, video_list, is_training=True, padding_val=0.0):
        """
            Generate batched features and masks from a list of dict items
        """
        feats = [x['feats'] for x in video_list if len(x['labels']) > 0]
        segmentation_labels = [x['segmentation_labels'] for x in video_list if len(x['labels']) > 0]
            
        feats_lens = torch.as_tensor([feat.shape[-1] for feat in feats])
        max_len = feats_lens.max(0).values.item()

        if is_training:
            if max_len > self.max_seq_len:
                import ipdb;ipdb.set_trace()
            assert max_len <= self.max_seq_len, "Input length must be smaller than max_seq_len during training"
            # set max_len to self.max_seq_len
            max_len = self.max_seq_len
            # batch input shape B, C, T
            batch_shape = [len(feats), feats[0].shape[0], max_len]
            batched_inputs = feats[0].new_full(batch_shape, padding_val)
            for feat, pad_feat in zip(feats, batched_inputs):
                pad_feat[..., :feat.shape[-1]].copy_(feat)
            segmentation_labels_shape = [len(segmentation_labels), max_len, segmentation_labels[0].shape[-1]]
            batched_segmentation_labels = segmentation_labels[0].new_full(segmentation_labels_shape, 0.0)
            for labels, pad_labels in zip(segmentation_labels, batched_segmentation_labels):
                pad_labels[:labels.shape[0], ...].copy_(labels)
            
        else:
            assert len(video_list) == 1, "Only support batch_size = 1 during inference"
            # input length < self.max_seq_len, pad to max_seq_len
            if max_len <= self.max_seq_len:
                max_len = self.max_seq_len
            else:
                # pad the input to the next divisible size
                stride = self.max_div_factor
                max_len = (max_len + (stride - 1)) // stride * stride
            padding_size = [0, max_len - feats_lens[0]]
            batched_inputs = F.pad(
                feats[0], padding_size, value=padding_val).unsqueeze(0)
            batched_segmentation_labels = None

        # generate the mask
        batched_masks = torch.arange(max_len)[None, :] < feats_lens[:, None]

        # push to device
        batched_inputs = batched_inputs.to(self.device)
        batched_masks = batched_masks.unsqueeze(1).to(self.device)

        return batched_inputs, batched_masks, batched_segmentation_labels

    # @torch.no_grad()
    def label_points(self, points, gt_segments, gt_labels, for_seg=False):
        # concat points on all fpn levels List[T x 4] -> F T x 4
        # This is shared for all samples in the mini-batch
        num_levels = len(points)
        concat_points = torch.cat(points, dim=0)        # [2046, 4]
        gt_cls, gt_offset = [], []
        normal_probs_cls, normal_probs_reg = [], []
        # loop over each video sample
        for gt_segment, gt_label in zip(gt_segments, gt_labels):
            if len(gt_label) == 0:
                import ipdb;ipdb.set_trace()
            if not for_seg:
                cls_targets, reg_targets, (normal_prob_cls, normal_prob_reg_left, normal_prob_reg_right) = self.label_points_single_video(
                    concat_points, gt_segment, gt_label
                )
            else:
                cls_targets = self.label_points_single_video_for_seg(
                    concat_points, gt_segment, gt_label
                )
            # append to list (len = # images, each of size FT x C)
            gt_cls.append(cls_targets.detach())
            if not for_seg:
                gt_offset.append(reg_targets.detach())
                normal_probs_cls.append(normal_prob_cls)
                normal_probs_reg.append([normal_prob_reg_left, normal_prob_reg_right])

        return gt_cls, gt_offset, normal_probs_cls, normal_probs_reg    # [378,1] [378,2] for each item in batch

    # @torch.no_grad()
    def label_points_single_video(self, concat_points, gt_segment, gt_label):
        # concat_points : F T x 4 (t, regressoin range, stride)
        # gt_segment : N (#Events) x 2
        # gt_label : N (#Events) x 1
        num_pts = concat_points.shape[0]    # all fpn level points   2046 = 1024+ 512....  [numpts,4]
        num_gts = gt_segment.shape[0]       # 1

        # corner case where current sample does not have actions
        if num_gts == 0:
            cls_targets = gt_segment.new_full((num_pts, self.num_classes), 0)
            reg_targets = gt_segment.new_zeros((num_pts, 2))
            return cls_targets, reg_targets

        # compute the lengths of all segments -> F T x N
        lens = gt_segment[:, 1] - gt_segment[:, 0]      # 每个segment的lens
        lens = lens[None, :].repeat(num_pts, 1)         # 扩充到num_pts，num_pts=2046, [numsegs, numpts]

        # compute the distance of every point to each segment boundary
        # auto broadcasting for all reg target-> F T x N x2
        gt_segs = gt_segment[None].expand(num_pts, num_gts, 2)      # [numpts, num_segs, 2]
        left = concat_points[:, 0, None] - gt_segs[:, :, 0]         # timestamp - gt_left
        right = gt_segs[:, :, 1] - concat_points[:, 0, None]        # timestamp - gt_right
        dist2center = (right - left) / 2.0
        normal_prob_cls = normal_distribution(dist2center / (concat_points[:, 3, None] * lens), self.mu[gt_label].permute(1,0), self.sigma[gt_label].permute(1,0))    # [num_pts, num_segs]
        normal_prob_reg_left = normal_distribution(dist2center / (concat_points[:, 3, None] * lens), self.mu_reg_left[gt_label].permute(1,0), self.sigma_reg_left[gt_label].permute(1,0))    # [num_pts, num_segs]
        normal_prob_reg_right = normal_distribution(dist2center / (concat_points[:, 3, None] * lens), self.mu_reg_right[gt_label].permute(1,0), self.sigma_reg_right[gt_label].permute(1,0))    # [num_pts, num_segs]
        reg_targets = torch.stack((left, right), dim=-1)            # [numpts, num_segs, 2]， 每个当前点距离gt left和right的距离

        if self.train_center_sample == 'radius':
            # center of all segments F T x N
            center_pts = 0.5 * (gt_segs[:, :, 0] + gt_segs[:, :, 1])        # gt seg center [numpts, 1]
            t_mins = \
                center_pts - concat_points[:, 3, None] * self.train_center_sample_radius
            t_maxs = \
                center_pts + concat_points[:, 3, None] * self.train_center_sample_radius

            # prevent t_mins / maxs from over-running the action boundary
            # left: torch.maximum(t_mins, gt_segs[:, :, 0])
            # right: torch.minimum(t_maxs, gt_segs[:, :, 1])
            # F T x N (distance to the new boundary)
            cb_dist_left = concat_points[:, 0, None] \
                           - torch.maximum(t_mins, gt_segs[:, :, 0])
            cb_dist_right = torch.minimum(t_maxs, gt_segs[:, :, 1]) \
                            - concat_points[:, 0, None]
            # F T x N x 2
            center_seg = torch.stack(
                (cb_dist_left, cb_dist_right), -1)
            # F T x N
            inside_gt_seg_mask = center_seg.min(-1)[0] > 0      # 判断action是否在这个proposal里 [num_pts, num_segs]
        else:
            # inside an gt action
            inside_gt_seg_mask = reg_targets.min(-1)[0] > 0         # [num_pts, num_segs]

        # limit the regression range for each location
        max_regress_distance = reg_targets.max(-1)[0]               # [num_pts, num_segs]
        # F T x N
        inside_regress_range = torch.logical_and(                   # [num_pts, num_segs] 判定符合regress range的点
            (max_regress_distance >= concat_points[:, 1, None]),
            (max_regress_distance <= concat_points[:, 2, None])
        )

        # if there are still more than one actions for one moment
        # pick the one with the shortest duration (easiest to regress)
        lens.masked_fill_(inside_gt_seg_mask==0, float('inf'))
        lens.masked_fill_(inside_regress_range==0, float('inf'))
        
        # F T x N -> F T
        min_len, min_len_inds = lens.min(dim=1)         # [numpts, num_segs] -> [num_pts]
        # corner case: multiple actions with very similar durations (e.g., THUMOS14)
        min_len_mask = torch.logical_and(                           # [num_pts, num_segs]
            (lens <= (min_len[:, None] + 1e-3)), (lens < float('inf'))
        ).to(reg_targets.dtype)

        # cls_targets: F T x C; reg_targets F T x 2
        gt_label_one_hot = F.one_hot(                               # [num_segs, num_classes]
            gt_label, self.num_classes
        ).to(reg_targets.dtype)

        cls_targets = min_len_mask @ gt_label_one_hot
        # cls_targets = max_len_mask @ gt_label_one_hot
        # to prevent multiple GT actions with the same label and boundaries
        cls_targets.clamp_(min=0.0, max=1.0)

        # OK to use min_len_inds
        reg_targets = reg_targets[range(num_pts), min_len_inds]
        normal_prob_cls = normal_prob_cls[range(num_pts), min_len_inds] # [numpts]
        normal_prob_reg_left = normal_prob_reg_left[range(num_pts), min_len_inds] # [numpts]
        normal_prob_reg_right = normal_prob_reg_right[range(num_pts), min_len_inds] # [numpts]
        # normalization based on stride
        reg_targets /= concat_points[:, 3, None]

        return cls_targets, reg_targets, (normal_prob_cls, normal_prob_reg_left, normal_prob_reg_right)        

    def decode_offset(self, out_offsets, pred_start_neighbours, pred_end_neighbours):
        if self.training:
            out_offsets = torch.cat(out_offsets, dim=1)
        return out_offsets

    def losses(
        self, fpn_masks,
        out_cls_logits, out_offsets,
        gt_cls_labels, gt_offsets,
        label_list=None,
        normal_probs_cls=None,
        normal_probs_reg=None,
        out_importances=None,
        out_start=None, out_end=None
    ):
        # fpn_masks, out_*: F (List) [B, T_i, C]
        # gt_* : B (list) [F T, C]
        # fpn_masks -> (B, FT)
        valid_mask = torch.cat(fpn_masks, dim=1)        # [b, all_points]

        out_start_logits, out_end_logits = None, None       # ignore

        # 1. classification loss
        # stack the list -> (B, FT) -> (# Valid, )
        gt_cls = torch.stack(gt_cls_labels)             # [b, all_points, C]
        normal_probs_cls = torch.stack(normal_probs_cls)        # [b, all_points]
        normal_probs_reg_left = torch.stack([x[0] for x in normal_probs_reg])   # [b, all_points]
        normal_probs_reg_right = torch.stack([x[1] for x in normal_probs_reg])
        
        pos_mask = torch.logical_and((gt_cls.sum(-1) > 0), valid_mask)      # [b, all_points]

        # cat the predicted offsets -> (B, FT, 2 (xC)) -> # (#Pos, 2 (xC))
        pred_offsets = torch.cat(out_offsets, dim=1)[pos_mask]          # [numpos, 2]
        gt_offsets = torch.stack(gt_offsets)[pos_mask]                  # [numpos, 2]

        # update the loss normalizer
        num_pos = pos_mask.sum().item()
        self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer + (
            1 - self.loss_normalizer_momentum
        ) * max(num_pos, 1)

        # gt_cls is already one hots encoded now, simply masking out
        gt_target = gt_cls[valid_mask]          # [b*allpoints, 1]

        # optinal label smoothing
        gt_target *= 1 - self.train_label_smoothing

        gt_target += self.train_label_smoothing / (self.num_classes + 1)

        # calculate iou and cls score

        # focal loss
        cls_loss = sigmoid_focal_loss(
            torch.cat(out_cls_logits, dim=1)[valid_mask],           # [#pos, c]
            gt_target,
            reduction='None'
        )

        normal_probs_cls[~pos_mask] = 1.0       # negative weight is 1.0
        cls_loss = cls_loss.sum(-1) # [#pos]
        cls_loss *= normal_probs_cls[valid_mask]
        cls_loss = cls_loss.sum()
        cls_loss /= self.loss_normalizer

        # ================= label-involved loss =================
        if label_list is not None:
            max_per_class_score = torch.cat(out_cls_logits, dim=1)
            max_per_class_score.masked_fill_(valid_mask.unsqueeze(-1) == False, -1e7)
            max_per_class_score = torch.max(max_per_class_score.softmax(-1), dim=1)[0]   # [b, c]
            # max_per_class_score = torch.max(max_per_class_score.sigmoid(), dim=1)[0]   # [b, c]
            involved_identify = torch.zeros_like(max_per_class_score)        # [b, c]
            for i, item in enumerate(involved_identify):
                involved_identify[i, label_list[i]] = 1         
            al_loss = -involved_identify * max_per_class_score.log() - (1-involved_identify) * (1-max_per_class_score).log()
            al_loss = al_loss.sum()
            al_loss /= self.loss_normalizer
        # ================= label-involved loss =================

        # 2. regression using IoU/GIoU loss (defined on positive samples)
        if num_pos == 0:
            reg_loss = 0 * pred_offsets.sum()
        else:
            # giou loss defined on positive samples
            reg_loss = ctr_diou_loss_1d(
                pred_offsets,
                gt_offsets,
                reduction='None'
            )
            normal_probs_reg_left[~pos_mask] = 1.0
            normal_probs_reg_right[~pos_mask] = 1.0
            reg_loss *= (normal_probs_reg_left[pos_mask] + normal_probs_reg_right[pos_mask]) / 2.0
            reg_loss *= normal_probs_cls[pos_mask]              # for one gaussian
            

            reg_loss = reg_loss.sum()
            reg_loss /= self.loss_normalizer
        if self.train_loss_weight > 0:
            loss_weight = self.train_loss_weight
        else:
            loss_weight = cls_loss.detach() / max(reg_loss.item(), 0.01)

        # return a dict of losses
        # final_loss = cls_loss + reg_loss * loss_weight
        final_loss = cls_loss + reg_loss * loss_weight + al_loss * self.al_loss_weight

        return {'cls_loss'   : cls_loss,
                'reg_loss'   : reg_loss,
                'al_loss'    : al_loss,
                'final_loss' : final_loss}

    @torch.no_grad()
    def inference(
        self,
        video_list,
        points, fpn_masks,
        out_cls_logits, out_offsets,
        out_lb_logits, out_rb_logits,
    ):
        # video_list B (list) [dict]
        # points F (list) [T_i, 4]
        # fpn_masks, out_*: F (List) [B, T_i, C]
        results = []

        # 1: gather video meta information
        vid_idxs = [x['video_id'] for x in video_list]
        vid_fps = [x['fps'] for x in video_list]
        vid_lens = [x['duration'] for x in video_list]
        vid_ft_stride = [x['feat_stride'] for x in video_list]
        vid_ft_nframes = [x['feat_num_frames'] for x in video_list]

        # ====== test upper bound =========
        # import pickle
        # candidate_val_label = pickle.load(open("../candidate_val_label.pkl", "rb"))
        # ====== test upper bound =========

        # 2: inference on each single video and gather the results
        # upto this point, all results use timestamps defined on feature grids
        for idx, (vidx, fps, vlen, stride, nframes) in enumerate(
            zip(vid_idxs, vid_fps, vid_lens, vid_ft_stride, vid_ft_nframes)
        ):
            # gather per-video outputs
            cls_logits_per_vid = [x[idx] for x in out_cls_logits]   # [192\96\...,1]
            offsets_per_vid = [x[idx] for x in out_offsets]         # [192\96\...,2]
            fpn_masks_per_vid = [x[idx] for x in fpn_masks]         # [192\96\...]


            lb_logits_per_vid = [None for x in range(len(out_cls_logits))]
            rb_logits_per_vid = [None for x in range(len(out_cls_logits))]


            # inference on a single video (should always be the case)
            results_per_vid = self.inference_single_video(
                points, fpn_masks_per_vid,
                cls_logits_per_vid, offsets_per_vid,
                lb_logits_per_vid, rb_logits_per_vid
            )
            # pass through video meta info
            results_per_vid['video_id'] = vidx
            results_per_vid['fps'] = fps
            results_per_vid['duration'] = vlen
            results_per_vid['feat_stride'] = stride
            results_per_vid['feat_num_frames'] = nframes
            results.append(results_per_vid)
        # step 3: postprocssing
        results = self.postprocessing(results)

        return results

    @torch.no_grad()
    def inference_single_video(
        self,
        points,
        fpn_masks,
        out_cls_logits,
        out_offsets,
        lb_logits_per_vid, rb_logits_per_vid,
        candidate_label=None
    ):
        # points F (list) [T_i, 4]
        # fpn_masks, out_*: F (List) [T_i, C]
        segs_all = []
        scores_all = []
        cls_idxs_all = []

        # loop over fpn levels
        # Normal
        # for cls_i, offsets_i, pts_i, mask_i in zip(
        #         out_cls_logits, out_offsets, points, fpn_masks
        #     ):
        
        # Trident
        for cls_i, offsets_i, pts_i, mask_i, sb_cls_i, eb_cls_i in zip(
                out_cls_logits, out_offsets, points, fpn_masks, lb_logits_per_vid, rb_logits_per_vid
        ):
            # ========== if test upper bound
            # labels = [x for x in candidate_label.keys()]
            # non_labels = list(set([x for x in range(110)]) - set(labels))[:20]
            # cls_i[:, non_labels] = -1e7
            # ========== if test upper bound

            # sigmoid normalization for output logits
            pred_prob = (cls_i.sigmoid() * mask_i.unsqueeze(-1)).flatten()      

            # Apply filtering to make NMS faster following detectron2
            # 1. Keep seg with confidence score > a threshold
            keep_idxs1 = (pred_prob > self.test_pre_nms_thresh)
            pred_prob = pred_prob[keep_idxs1]
            topk_idxs = keep_idxs1.nonzero(as_tuple=True)[0]

            # 2. Keep top k top scoring boxes only
            num_topk = min(self.test_pre_nms_topk, topk_idxs.size(0))
            pred_prob, idxs = pred_prob.sort(descending=True)
            pred_prob = pred_prob[:num_topk].clone()
            topk_idxs = topk_idxs[idxs[:num_topk]].clone()

            # fix a warning in pytorch 1.9
            pt_idxs =  torch.div(
                topk_idxs, self.num_classes, rounding_mode='floor'
            )
            cls_idxs = torch.fmod(topk_idxs, self.num_classes)

            # 3. gather predicted offsets
            offsets = offsets_i[pt_idxs]
            pts = pts_i[pt_idxs]

            # 4. compute predicted segments (denorm by stride for output offsets)
            seg_left = pts[:, 0] - offsets[:, 0] * pts[:, 3]
            seg_right = pts[:, 0] + offsets[:, 1] * pts[:, 3]
            pred_segs = torch.stack((seg_left, seg_right), -1)

            # 5. Keep seg with duration > a threshold (relative to feature grids)
            seg_areas = seg_right - seg_left
            keep_idxs2 = seg_areas > self.test_duration_thresh

            # *_all : N (filtered # of segments) x 2 / 1
            segs_all.append(pred_segs[keep_idxs2])
            scores_all.append(pred_prob[keep_idxs2])
            cls_idxs_all.append(cls_idxs[keep_idxs2])

        # cat along the FPN levels (F N_i, C)
        segs_all, scores_all, cls_idxs_all = [
            torch.cat(x) for x in [segs_all, scores_all, cls_idxs_all]
        ]
        results = {'segments' : segs_all,
                   'scores'   : scores_all,
                   'labels'   : cls_idxs_all}

        return results

    @torch.no_grad()
    def postprocessing(self, results):
        # input : list of dictionary items
        # (1) push to CPU; (2) NMS; (3) convert to actual time stamps
        processed_results = []
        for results_per_vid in results:
            # unpack the meta info
            vidx = results_per_vid['video_id']
            fps = results_per_vid['fps']
            vlen = results_per_vid['duration']
            stride = results_per_vid['feat_stride']
            nframes = results_per_vid['feat_num_frames']
            # 1: unpack the results and move to CPU
            segs = results_per_vid['segments'].detach().cpu()
            scores = results_per_vid['scores'].detach().cpu()
            labels = results_per_vid['labels'].detach().cpu()
            if self.test_nms_method != 'none':
                # 2: batched nms (only implemented on CPU)
                segs, scores, labels = batched_nms(
                    segs, scores, labels,
                    self.test_iou_threshold,
                    self.test_min_score,
                    self.test_max_seg_num,
                    use_soft_nms = (self.test_nms_method == 'soft'),
                    multiclass = self.test_multiclass_nms,
                    sigma = self.test_nms_sigma,
                    voting_thresh = self.test_voting_thresh
                )
            # 3: convert from feature grids to seconds
            if segs.shape[0] > 0:
                segs = (segs * stride + 0.5 * nframes) / fps
                # truncate all boundaries within [0, duration]
                segs[segs<=0.0] *= 0.0
                segs[segs>=vlen] = segs[segs>=vlen] * 0.0 + vlen
            # 4: repack the results
            processed_results.append(
                {'video_id' : vidx,
                 'segments' : segs,
                 'scores'   : scores,
                 'labels'   : labels}
            )

        return processed_results

    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
            
    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feats, text_feats):
        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr 

    # @torch.no_grad()
    def label_points_single_video_for_seg(self, concat_points, gt_segment, gt_label):
        # concat_points : F T x 4 (t, regressoin range, stride)
        # gt_segment : N (#Events) x 2
        # gt_label : N (#Events) x 1
        num_pts = concat_points.shape[0]    # 192 + 96 + 49 + 24 + 12 + 6   2046 = 1024+ 512....  [numpts,4]
        num_gts = gt_segment.shape[0]       # 1

        # corner case where current sample does not have actions
        if num_gts == 0:
            cls_targets = gt_segment.new_full((num_pts, self.num_classes), 0)
            return cls_targets

        # compute the lengths of all segments -> F T x N
        lens = gt_segment[:, 1] - gt_segment[:, 0]      # 每个segment的lens
        lens = lens[None, :].repeat(num_pts, 1)         # 扩充到num_pts，num_pts=2046, [numsegs, numpts]

        # compute the distance of every point to each segment boundary
        # auto broadcasting for all reg target-> F T x N x2
        gt_segs = gt_segment[None].expand(num_pts, num_gts, 2)      # [numpts, num_segs, 2]
        left = concat_points[:, 0, None] - gt_segs[:, :, 0]         # timestamp - gt_left
        right = gt_segs[:, :, 1] - concat_points[:, 0, None]        # timestamp - gt_right
        reg_targets = torch.stack((left, right), dim=-1)            # [numpts, num_segs, 2]， 每个当前点距离gt left和right的距离

        # inside an gt action
        inside_gt_seg_mask = reg_targets.min(-1)[0] > 0         # [num_pts, num_segs]

        # limit the regression range for each location
        max_regress_distance = reg_targets.max(-1)[0]               # [num_pts, num_segs]
        # F T x N
        inside_regress_range = torch.logical_and(                   # [num_pts, num_segs] 判定符合regress range的点
            (max_regress_distance >= concat_points[:, 1, None]),
            (max_regress_distance <= concat_points[:, 2, None])
        )

        # if there are still more than one actions for one moment
        # pick the one with the shortest duration (easiest to regress)
        lens.masked_fill_(inside_gt_seg_mask==0, float('inf'))
        lens.masked_fill_(inside_regress_range==0, float('inf'))
        # F T x N -> F T
        min_len, min_len_inds = lens.min(dim=1)         # [numpts, num_segs] -> [num_pts]
        # corner case: multiple actions with very similar durations (e.g., THUMOS14)
        # ==================== 只回归最小的
        min_len_mask = torch.logical_and(                           # [num_pts, num_segs]
            (lens <= (min_len[:, None] + 1e-3)), (lens < float('inf'))
        ).to(reg_targets.dtype)


        # cls_targets: F T x C; reg_targets F T x 2
        gt_label_one_hot = F.one_hot(                               # [num_segs, num_classes]
            gt_label, self.num_classes
        ).to(reg_targets.dtype)
        cls_targets = min_len_mask @ gt_label_one_hot
        # cls_targets = max_len_mask @ gt_label_one_hot
        # to prevent multiple GT actions with the same label and boundaries
        cls_targets.clamp_(min=0.0, max=1.0)

        return cls_targets


























    def _to_roi_align_format(self, rois, T, k=4, scale_factor=1.):
        '''Convert RoIs to RoIAlign format.
        Params:
            RoIs: normalized segments coordinates, shape (batch_size, num_segments, 4)
            T: length of the video feature sequence
        '''
        # transform to absolute axis
        B, N = rois.shape[:2]
        rois_len = rois[..., 1] - rois[..., 0]
        scale_len = (scale_factor - 1) / 2 * rois_len
        rois[..., 0] -= scale_len
        rois[..., 1] += scale_len
        rois_abs = rois
        # expand the RoIs
        rois_abs = torch.clamp(rois_abs, min=0, max=T)  # (N, T, 2)
        # add batch index
        batch_ind = torch.arange(0, B).view((B, 1, 1)).to(rois_abs.device)
        batch_ind = batch_ind.repeat(1, N, 1)
        # batch_ind[:, k + 1] = batch_ind[:, k + 1] + B
        # batch_ind[:, k + 2] = batch_ind[:, k + 2] + B * 2
        rois_abs = torch.cat((batch_ind, rois_abs.float()), dim=-1)
        # NOTE: stop gradient here to stablize training
        return rois_abs.view((-1, 3)).detach()
    

    def get_segment_level_contra_loss(self, video_features, segments, labels):
        # [b,c,t] list[#, 2] list[#]
        # roi pooling feature
        # self.length_theta = 0.2
        self.min_len = 1
        self.gap = 1.0
        # self.gap = 0.0
        cand_features = []
        b, c, t = video_features.shape
        num_seg = segments[1].shape[0]
        for i in range(video_features.shape[0]):
            segment = segments[i]
            theta_lens = ((segment[:, 1] - segment[:, 0]) * self.length_theta).clamp(min=1.0)  # [#]
            center_pts = 0.5 * (segment[:, 0] + segment[:, 1])
            # center feature
            center_segs = center_pts.unsqueeze(-1).repeat(1,2)
            center_segs[:,0] -= theta_lens
            center_segs[:,1] += theta_lens
            center_rois = self._to_roi_align_format(center_segs.unsqueeze(0), video_features[i].shape[-1])
            center_roi_feature = self.roi_extractor(video_features[i].unsqueeze(0), center_rois).mean(-1)  # [#, d, 16]
            center_roi_feature = F.normalize(center_roi_feature, dim=-1)      
            cand_features.append(center_roi_feature)
            # boundary feature
            # left 
            left_segs = copy.deepcopy(segment)
            left_segs[:,1] = left_segs[:,0] + theta_lens
            left_rois = self._to_roi_align_format(left_segs.unsqueeze(0), video_features[i].shape[-1])
            left_roi_feature = self.roi_extractor(video_features[i].unsqueeze(0), left_rois).mean(-1)
            # right
            right_segs = copy.deepcopy(segment)
            right_segs[:,0] = right_segs[:,1] - theta_lens
            right_rois = self._to_roi_align_format(right_segs.unsqueeze(0), video_features[i].shape[-1])
            right_roi_feature = self.roi_extractor(video_features[i].unsqueeze(0), right_rois).mean(-1)
            boundary_roi_feature = (left_roi_feature + right_roi_feature) / 2.0
            boundary_roi_feature = F.normalize(boundary_roi_feature, dim=-1)
            cand_features.append(boundary_roi_feature)
            # outer feature
            # left
            left_out_segs = copy.deepcopy(segment)
            left_out_segs[:,1] = (left_out_segs[:,0] - self.gap).clamp(min=0.0)
            left_out_segs[:,0] = (left_out_segs[:,1] - theta_lens).clamp(min=0.0)
            left_out_rois = self._to_roi_align_format(left_out_segs.unsqueeze(0), video_features[i].shape[-1])
            left_out_roi_feature = self.roi_extractor(video_features[i].unsqueeze(0), left_out_rois).mean(-1)
            # right
            right_out_segs = copy.deepcopy(segment)
            right_out_segs[:,0] = (right_out_segs[:,1] + self.gap).clamp(max=video_features[i].shape[-1])
            right_out_segs[:,1] = (right_out_segs[:,0] + theta_lens).clamp(max=video_features[i].shape[-1])
            right_out_rois = self._to_roi_align_format(right_out_segs.unsqueeze(0), video_features[i].shape[-1])
            right_out_roi_feature = self.roi_extractor(video_features[i].unsqueeze(0), right_out_rois).mean(-1)
            out_roi_feature = (left_out_roi_feature + right_out_roi_feature) / 2.0
            out_roi_feature = F.normalize(out_roi_feature, dim=-1)
            cand_features.append(out_roi_feature)

        cand_features = torch.cat(cand_features)    # [#*3, d]
        # get pos/neg mask
        labels = torch.cat(labels)      # [#]
        num = labels.shape[0]
        m1 = labels.unsqueeze(0).repeat(num, 1) # [#,#]
        m2 = labels.unsqueeze(-1).repeat(1, num)    #[#,#]
        pn_mask = (m1 == m2).float().unsqueeze(0).unsqueeze(0)
        pn_mask = F.interpolate(pn_mask, size=(num*3, num*3), mode='nearest').squeeze(0).squeeze(0) # [3#,3#]
        # pn_mask[torch.arange(-1,num*3,3), :] = 0.0
        pn_mask[:, torch.arange(-1,num*3,3)] = 0.0
        ignore_mask = torch.ones((num*3, num*3), dtype=torch.float).to(labels.device)
        ignore_mask[torch.arange(num*3), torch.arange(num*3)] = 0.0 # [3#, 3#]
        # get loss
        logits = cand_features @ cand_features.T    # [3#,3#]
        logits /= self.temperature
        logits_valid = logits * ignore_mask         # [3#,3#]
        contra_loss = -torch.log((F.softmax(logits_valid, dim=1) * pn_mask).sum(1))
        final_mask = torch.ones(3*num, dtype=torch.float).to(contra_loss.device)
        final_mask[torch.arange(-1,num*3,3)] = 0.0
        contra_loss *= final_mask
        contra_loss = contra_loss.sum()
        return contra_loss


class FeatureAlign(nn.Module):
    def __init__(self, d_in, d_out, num_layers=2, kernel_size=3, deformable_groups=1):
        super().__init__()
        # offset_channels = kernel_size * kernel_size * 2
        # self.convs_offset = nn.ModuleList()
        self.convs_adapt = nn.ModuleList()
        for i in range(num_layers):
            # conv_offset = nn.Conv1d(2, deformable_groups * offset_channels,
            #                             1, bias=False)
            conv_adapt = PackedDeformConv1d(d_in, d_out, kernel_size=kernel_size,
                                        padding='same')
            # self.convs_offset.append(conv_offset)
            self.convs_adapt.append(conv_adapt)
        self.relu = nn.ReLU(inplace=True)
        # self.init_weights()
    def init_weights(self):
        def normal_init(module, mean=0, std=1, bias=0):
            nn.init.normal_(module.weight, mean, std)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, bias)
        for i in range(len(self.convs_adapt)):
            normal_init(self.convs_adapt[i].offset_pconv_tal, std=0.1)
            normal_init(self.convs_adapt[i], std=0.01)
    def forward(self, fpn_feats, fpn_masks, out_offsets):
        # out_offset: List[B, 2, T_i]
        out_fpn_feats = tuple()
        new_fpn_masks = tuple()
        for l, (cur_feat, cur_mask) in enumerate(zip(fpn_feats, fpn_masks)):
            cur_offset = out_offsets[l]          # [b,2,t]
            # residual_feat = fpn_feats[l]
            cur_feat, cur_mask = fpn_feats[l], fpn_masks[l]     # [b,c,t] [b,1,t]
            for i in range(len(self.convs_adapt)):
                # import ipdb;ipdb.set_trace()
                # cur_offset = self.convs_offset[i](cur_offset)       # [b,dg*oc,t]
                cur_feat, cur_offset = self.convs_adapt[i](cur_feat, offsets=cur_offset, with_offsets=True)
                cur_feat = self.relu(cur_feat)
                cur_feat = cur_feat * cur_mask.detach()
                cur_mask = cur_mask.bool()
            # cur_feat = residual_feat + cur_feat
            out_fpn_feats += (cur_feat, )
            new_fpn_masks += (cur_mask, )
        return fpn_feats, new_fpn_masks