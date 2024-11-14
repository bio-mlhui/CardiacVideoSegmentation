from copy import deepcopy as dcopy
import numpy as np
d_model = 64
attention_defaults = {
    'attn': {
        'dropout': 0.1,
        'nheads': 8,
        'dim_feedforward': 2048,
        'activation': 'relu',
        'normalize_before': False,
        'enforce_input_proj': True, # try 每个module对input进行proj
    },
    'deform_attn':{
        'nheads': 8,
        'dim_feedforward': 1024,
        'activation': 'relu',
        'dropout': 0.,
        'enc_n_points': 4
    },
}
ckpt_iter = 1366 // 4
trainer_configs = {
    'eval_seed': 2024, 'model_schedule_seed': 2024, 'stream_idx_seed': 2024,
    'initckpt':{'path': '', 'load_schedule': False, 'load_model': True, 'load_optimizer': False, 'load_random': False,
                'eval_init_ckpt': False,},
    'data':{
        'evaluate': {
            'card_labeled_test': {
                'mapper': {'name': 'Card_EvalMapper', 'step_size': 11, 'res': 384 },
                'evaluator': {'name': 'Card_Evaluator',},
            },
        }, 
        'train': {
            'card_labeled_train': {
                'mapper': { # 1366 // 4 = 
                    'name': 'Card_TrainMapper', 
                    'frame_sampler': {
                        'name': 'ReferenceFrame_FrameSampler',
                        'clip_sizes': [7],
                        'clip_distribute': 'dense',
                        'clip_position': 'center'}, 
                    'res': 336, 'clip_global_targets_map_to_local_targets': False,
                },
            },
        },
    },
    'optim': {
        'splits': [0, None],
        'batch_sizes': [4], 
        'ckpted_iters': ckpt_iter, 
        'one_batch_two_epoch': 'just_use', 
        'scheduler': { 'name': 'multistep_lr', 'milestones':[ckpt_iter*3,
                                                             ckpt_iter*6, ckpt_iter*9, 
                                                             ckpt_iter*12], 
                      'gamma': 0.5, 'verbose': False},
        'name': 'AdamW',
        'base_lr': 1e-3, 'backbone_lr_multiplier': 0.1, 'weight_decay': 1e-4,
        'weight_decay_embed': 0.0,
        'weight_decay_norm': 0.0,
        'clip_gradients': {
            'clip_type': 'full_model', # NORM/VALUE # grad.data.clamp_
            'clip_value': 0.01,
            'enabled': True,
            'norm_type': 2.0
        },
    },
    'model': {
        'name': 'card_model',
        'input_aux':{'video_auxes':[{'name': 'HilbertCurve_FrameQuery','frame_query_number': 20}],  'targets_auxes': [],}, 

        "video_backbone":{
            'name': 'Video2D_PVT_V2',
            'freeze': False,
        },  
        'fusion': {
            'name': 'Video_Deform2D_DividedTemporal_MultiscaleEncoder_localGlobal',
            'd_model': d_model,
            'video_projs':{
                'name': 'VideoConv_MultiscaleProj',
                'projs':{
                    'res3': {'kernel_size': 1, 'bias': False, 'norm': 'gn_32'},
                    'res4': {'kernel_size': 1, 'bias': False, 'norm': 'gn_32'},
                    'res5': {'kernel_size': 1, 'bias': False, 'norm': 'gn_32'},
                },
            },
            'nlayers': 3,
            'encoded_scales': ['res3', 'res4', 'res5'],
            'fpn_norm': 'GN',
            'deform_attn': dcopy(attention_defaults['deform_attn']),
            'frame_nqueries': 20,
            'add_local': True,
            'local_configs': {'d_model': d_model, 'num_heads': 8, 
                              'kernel_size': 5, 'dilation': 1, 'dropout': 0.0,'num_steps': 1},
            'add_global': True,
            'global_configs': {'d_model': d_model, 'dim_feedforward': 2048, 'dropout': 0.0,
                               'scan_order': 'hilbert', 'd_state': 16, 'd_conv': 3, 'nlayers': 3,
                               'add_attn_mask': False}
        },            
        'decoder':{
            'name': 'Card_Decoder',
            'd_model': d_model,
            'attn': dcopy(attention_defaults['attn']),
            'video_nqueries': 10,
            'inputs_projs': None,
            'nlayers': 3,
            'memory_scales': ['res5','res4','res3'],
            'mask_scale': 'res2',
            'num_classes': 4,
            'head_outputs': ['mask', 'class'], # polygon
            'temporal_self_layer': {
                'name': 'FrameQuery_SS2DLayer_v2',
                'd_model': d_model,
                'nlayers': 3,
                'dropout': 0.0,
                'd_state': 16,
                'd_conv': 3,
                'dim_feedforward': 2048,
            },
            'temporal_cross_layer': {
                'name': 'TemporalQuery_CrossSelf',
                'd_model': d_model,
                'attn': dcopy(attention_defaults['attn']),
            },

            'loss':{
                'losses': {
                    'point_mask_dice_ce': {'num_points': 12544, 'oversample_ratio':3.0, 'importance_sample_ratio': 0.75},
                    'class_ce': {},
                }, 
                'matching_metrics': {
                    'class_prob': {'prob': 2},
                    # 'mask_dice_ce': {'ce': 2, 'dice': 5},
                    'point_mask_dice_ce': {'ce':2, 'dice':2, 'num_points':12544}
                },
                'aux_layer_weights': 1., # int/list
                'background_cls_eos': 0.1,
            }, 
        },
        'loss_weight': {'mask_dice': 5, 
                        'mask_ce': 5, 
                        'class_ce':2},
    },
     
}

