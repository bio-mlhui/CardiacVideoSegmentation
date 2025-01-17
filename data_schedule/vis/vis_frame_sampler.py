
from detectron2.utils.registry import Registry
import random
import numpy as np
import torch
import logging
from detectron2.utils import comm
""" 
frame_idx, all_frames -> frames
包含怎么抽其他帧, 
    method, position,
合法的定义,
    naive sampler是指不借助任何model, data 选择其他帧, 没有任何其他知识, 
    比如要用到optical flow确定动作的快慢, 动作快的抽的scale大一些, 动作小的丑的scale小一些
    比如要用到其他model, 或者model内部的一些知识, 则需要把sample放到model里, model的接口也要变
frame_sampler 训练和测试
"""
VIS_FRAMES_SAMPLER_REGISTRY = Registry('VIS_FRAMES_SAMPLER')

import random
class Frames_Sampler:
    def __init__(self) -> None:
        pass

@VIS_FRAMES_SAMPLER_REGISTRY.register()
class Naive_ReferenceFrame_FrameSampler:
    # naive: 没有使用外部模型或者数据, # 没有考虑每一帧的情况, 只是按照下标进行抽样
    def __init__(self, sampler_configs, dataset_meta, **kwargs):            
        self.reference_frame_step_size = dataset_meta.get('step_size')

        self.clip_sizes = list(sampler_configs['clip_sizes']) # list[int]
        self.clip_distribute = sampler_configs['clip_distribute'] # dense, sparse, local_global
        self.clip_position = sampler_configs['clip_position'] # former, center, latter

        if max(self.clip_sizes) > self.reference_frame_step_size:
            if comm.is_main_process():
                logging.warning('训练的clip大小大于数据集的step size,可能会造成训练时每个sample之间帧重复')

    def __call__(self, 
                 frame_idx=None,
                 all_frames=None, # list[str]
                 **kwargs):
        random_clip_size = random.choice(self.clip_sizes)
        video_len = len(all_frames)
        sample_indx = [frame_idx]
        if (self.clip_position == 'center') and (self.clip_distribute == 'local_global'):
            if random_clip_size != 1:
                # local sample
                sample_id_before = random.randint(1, 3)
                sample_id_after = random.randint(1, 3)
                local_indx = [max(0, frame_idx - sample_id_before), min(video_len - 1, frame_idx + sample_id_after)]
                sample_indx.extend(local_indx)
                # global sampling
                if random_clip_size > 3:
                    all_inds = list(range(video_len))
                    global_inds = all_inds[:min(sample_indx)] + all_inds[max(sample_indx):]
                    global_n = random_clip_size - len(sample_indx)
                    if len(global_inds) > global_n:
                        select_id = random.sample(range(len(global_inds)), global_n)
                        for s_id in select_id:
                            sample_indx.append(global_inds[s_id])
                    elif video_len >= global_n:  # sample long range global frames
                        select_id = random.sample(range(video_len), global_n)
                        for s_id in select_id:
                            sample_indx.append(all_inds[s_id])
                    else:
                        select_id = random.sample(range(video_len), global_n - video_len) + list(range(video_len))           
                        for s_id in select_id:                                                                   
                            sample_indx.append(all_inds[s_id])
        
        elif (self.clip_position == 'center') and (self.clip_distribute == 'dense'):
            half_size = (random_clip_size - 1) // 2
            # 把负的换成0, 大于最大的换成最后一帧
            sample_indx += list(range(frame_idx - half_size, frame_idx))
            sample_indx += list(range(frame_idx+1, half_size + frame_idx + 1))

            if len(sample_indx) < random_clip_size: # 向前补一个
                sample_indx = [min(sample_indx)] + sample_indx
            assert len(sample_indx) == random_clip_size
            sample_indx = torch.tensor(sample_indx)
            sample_indx = sample_indx.clamp_(min=0, max=video_len-1)
            sample_indx = sample_indx.tolist()
        else:
            raise ValueError()
        sample_indx.sort()
        sampled_frames = [all_frames[idx] for idx in sample_indx]
        return sampled_frames


@VIS_FRAMES_SAMPLER_REGISTRY.register()
class ReferenceFrame_FrameSampler:
    # naive: 没有使用外部模型或者数据, # 没有考虑每一帧的情况, 只是按照下标进行抽样
    def __init__(self, sampler_configs, dataset_meta, **kwargs):            
        reference_frame_step_size = dataset_meta.get('step_size')

        self.clip_sizes = list(sampler_configs['clip_sizes']) # list[int]
        self.clip_distribute = sampler_configs['clip_distribute'] # dense, sparse, local_global
        self.clip_position = sampler_configs['clip_position'] # former, center, latter

        if  reference_frame_step_size is not None:
            if max(self.clip_sizes) > reference_frame_step_size:
                if comm.is_main_process():
                    logging.warning('训练的clip大小大于数据集的step size,可能会造成训练时每个sample之间帧重复')

    def __call__(self, 
                 frame_idx=None,
                 all_frames=None, # list[str]
                 **kwargs):
        random_clip_size = random.choice(self.clip_sizes)
        ann_frame_idx = all_frames.index(frame_idx)
        left_len = random_clip_size // 2 
        right_len = (random_clip_size - 1) // 2
        if (self.clip_position == 'center') and (self.clip_distribute == 'local_global'):
            video_len = len(all_frames)
            assert random_clip_size >= 3          
            # local sample, 从附近3个帧抽
            before_idx = max(ann_frame_idx - random.randint(1, 3), 0)
            after_idx = min(ann_frame_idx + random.randint(1, 3), video_len - 1)
            sample_indx = [before_idx, ann_frame_idx, after_idx]
            # global sample, 从比较远的地方抽
            if random_clip_size > 3:
                remain_left_idxs = torch.tensor(list(range(0, min(sample_indx)+1))).numpy()
                remain_right_idxs = torch.tensor(list(range(max(sample_indx), video_len))).numpy()
                sampled_lefts = np.random.choice(remain_left_idxs, left_len - 1).tolist()
                sampled_rights = np.random.choice(remain_right_idxs, right_len - 1).tolist()
                sample_indx = sampled_lefts + sample_indx + sampled_rights

            sampled_frames = [all_frames[idx] for idx in sample_indx]
            return sampled_frames 
          
        elif (self.clip_position == 'center') and (self.clip_distribute == 'dense'):
            ann_frame_idx = all_frames.index(frame_idx)
            all_idxs = list(range(ann_frame_idx - random_clip_size// 2, ann_frame_idx + (random_clip_size+1)//2))
            all_idxs = torch.tensor(all_idxs).clamp(0, max=len(all_frames)-1).int() # annotate_frame_idx = T//2
            frames = [all_frames[idx] for idx in all_idxs]
            return frames

        else:
            raise ValueError()


@VIS_FRAMES_SAMPLER_REGISTRY.register()
class Naive_Hybrid_Temporal_Scales:
    # naive: 没有使用外部模型或者数据
    # hybrid temporal scale: 对于一个video, 随机选一帧, 然后随机选一帧, 进行local-global
    def __init__(self, 
                 sampler_config,
                 dataset_meta,
                 **kwargs):
        super().__init__()
        self.clip_sizes = np.array(sampler_config['clip_sizes'])
        self.no_global = sampler_config['no_global']

    def __call__(self,
                 all_frames = None, # list[str]
                 **kwargs):
        video_len = len(all_frames)
        rand_clip_size = np.random.choice(self.clip_sizes, 1)[0]
        if self.no_global:
            frame_idx = random.randint(0, video_len - rand_clip_size) # 之后的几帧
            sample_index = list(range(frame_idx, frame_idx + rand_clip_size))
            assert len(sample_index) == rand_clip_size
            
        else:
            frame_idx = random.randint(0, video_len-1) # center_frame
            sample_indx = [frame_idx]
            if rand_clip_size != 1:
                # local sample
                sample_id_before = random.randint(1, 3)
                sample_id_after = random.randint(1, 3)
                local_indx = [max(0, frame_idx - sample_id_before), min(video_len - 1, frame_idx + sample_id_after)]
                sample_indx.extend(local_indx)
                # global sampling
                if rand_clip_size > 3:
                    all_inds = list(range(video_len))
                    global_inds = all_inds[:min(sample_indx)] + all_inds[max(sample_indx):]
                    global_n = rand_clip_size - len(sample_indx)
                    if len(global_inds) > global_n:
                        select_id = random.sample(range(len(global_inds)), global_n)
                        for s_id in select_id:
                            sample_indx.append(global_inds[s_id])
                    elif video_len >= global_n:  # sample long range global frames
                        select_id = random.sample(range(video_len), global_n)
                        for s_id in select_id:
                            sample_indx.append(all_inds[s_id])
                    else:
                        select_id = random.sample(range(video_len), global_n - video_len) + list(range(video_len))           
                        for s_id in select_id:                                                                   
                            sample_indx.append(all_inds[s_id])

        sample_indx.sort()
        sampled_frames = [all_frames[idx] for idx in sample_indx]
        return sampled_frames
    