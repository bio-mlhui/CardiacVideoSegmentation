import torch
import numpy as np
import random
import math
import logging
import time
import os
import sys
from utils.misc import reduce_dict, to_device, reduce_scalar, is_dist_avail_and_initialized
# from models.optimization.utils import get_total_grad_norm
import gc
import wandb
from utils.misc import  SmoothedValue, MetricLogger
from torch.nn.parallel import DistributedDataParallel as DDP
import detectron2.utils.comm as comm
import datetime
import torch.distributed as dist
from data_schedule import build_schedule
from models import model_entrypoint
from utils.misc import to_device

__all__ = ['Trainer']
class Trainer:
    def __init__(self, configs):
        torch.autograd.set_detect_anomaly(False)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        seed = configs['model_schedule_seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.random.manual_seed(seed)
        with torch.cuda.device(self.device):
            torch.cuda.manual_seed(seed)        

        # model and data
        create_model_schedule = model_entrypoint(configs['model']['name'])
        self.model, self.train_samplers, self.train_loaders, self.eval_function = create_model_schedule(configs, device=self.device,)
        logging.debug(configs)
        if comm.is_main_process():
            self.metric_logger = TrainerLogger(delimiter='\t')
        # trainer
        self.eval_seed = configs['eval_seed']
        self.out_dir = configs['out_dir']
        self.ckpted_iters = configs['optim']['ckpted_iters'] # list[int]
        self.num_iterations = 0 
        self.num_samples = 0 # wandb的横坐标, 模型见过(forward-backward成功)的samples的数量
        assert self.train_samplers[0].start_idx == self.num_samples

        random.seed(seed + comm.get_rank())
        np.random.seed(seed + comm.get_rank())
        torch.random.manual_seed(seed + comm.get_rank())
        with torch.cuda.device(self.device):
            torch.cuda.manual_seed(seed + comm.get_rank()) # 每个进程的seed不一样

        # init_ckpt 和 train_resume 不同, 更改之前的起始点和不同的线
        if configs['initckpt']['path'] != '':
            self.load_ckpt(configs['initckpt']['path'], 
                           load_random=configs['initckpt']['load_random'], # 随机状态
                           load_model=configs['initckpt']['load_model'], 
                           load_schedule=configs['initckpt']['load_schedule'], # 数据流idx
                           load_optimize=configs['initckpt']['load_optimizer'])
        self.save_ckpt() # 存储初始点
        if configs['initckpt']['eval_init_ckpt']:
            self.evaluate() # 测试的随机状态独立于训练的状态
            self.load_ckpt(os.path.join(self.iteration_dir, 'ckpt.pth.tar'),  # 训练的随机状态
                       load_random=True, load_schedule=False, load_model=False, load_optimize=False,)

    def train(self):   
        manual_stop_train = False
        for loader in self.train_loaders:
            for idx, batch_dict in enumerate(loader):
                if manual_stop_train:
                    self.save_ckpt()
                self.model.train()
                meta_idxs = batch_dict.pop('meta_idxs')
                visualize = batch_dict.pop('visualize')
                batch_dict = to_device(batch_dict, self.device)
                batch_dict['visualize_paths'] = self.visualize_path(meta_idxs=meta_idxs, 
                                                                    visualize=visualize) # visualize model训练过程
                batch_dict['num_iterations'] = self.num_iterations
                iteration_time = time.time()

                loss_dict_unscaled, loss_weight, optimize_dict = self.model(batch_dict)   
                # backward, step, zero_grad, lr_scheduler  
                model_without_ddp = self.model.module if isinstance(self.model, DDP) else self.model
                model_without_ddp.optimize(loss_dict_unscaled=loss_dict_unscaled,
                                    loss_weight=loss_weight,
                                    num_iterations=self.num_iterations,
                                    optimize_dict=optimize_dict,
                                    num_samples=self.num_samples,
                                    )
                iteration_time = time.time() - iteration_time
                sample_idxs = comm.all_gather(meta_idxs) 
                sample_idxs = [taylor for cardib in sample_idxs for taylor in cardib]
                self.num_samples += len(sample_idxs)
                self.num_iterations += 1
                loss_dict_unscaled_item = {key: torch.tensor(value.detach().item() if isinstance(value, torch.Tensor) else value , 
                                                             device=self.device) for key, value in loss_dict_unscaled.items()}
                del loss_dict_unscaled # delete graph
                self._log(loss_dict_unscaled=loss_dict_unscaled_item,
                          loss_weight=loss_weight,
                          sample_idxs=sample_idxs,
                          iteration_time=iteration_time,
                          lr_group_dicts=model_without_ddp.get_lr_group_dicts())
    def save_ckpt(self):
        """
        存储模型的状态, 模型的优化状态, 随机状态, 训练的当前节点, 
        模型的metrics设置为空
        """
        rng_state_dict = {comm.get_rank(): {
            'cpu_rng_state': torch.get_rng_state(),
            'gpu_rng_state': torch.cuda.get_rng_state(self.device),
            'numpy_rng_state': np.random.get_state(),
            'py_rng_state': random.getstate()
        }}

        rng_state_dict_by_rank = comm.gather(rng_state_dict, dst=0)

        if comm.is_main_process():
            rng_state_dict_by_rank = {key : value for rs in rng_state_dict_by_rank for key,value in rs.items()}
            model_without_ddp = self.model.module if isinstance(self.model, DDP) else self.model
            checkpoint_dict = {
                'model': model_without_ddp.state_dict(),
                'optimizer': model_without_ddp.optimize_state_dict(),
                'num_samples': self.num_samples,
                'num_iterations': self.num_iterations,
                'rng_state_dict_by_rank': rng_state_dict_by_rank, 
                'metrics': {},
            }
            os.makedirs(self.iteration_dir, exist_ok=True)
            torch.save(checkpoint_dict, os.path.join(self.iteration_dir, 'ckpt.pth.tar'))
            del checkpoint_dict
        if is_dist_avail_and_initialized():
            dist.barrier()
        del rng_state_dict_by_rank

    @torch.no_grad()
    def evaluate(self):
        """
        TODO: 怎么保证测试的model 和ckpt里的Model是一样的 ?
        使用统一的 evaluate随机状态 对model进行评估, 要求存在ckpt
        得到的metrics写进 当前状态的ckpt里
            如果当前状态ckpt的指标 和新计算的指标差别非常大, 那么把新值写进去, 但是wandb如果已经log了的话不能更改
            log新metric, 存metric到当前的ckpt里
        """
        random.seed(self.eval_seed)
        np.random.seed(self.eval_seed)
        torch.random.manual_seed(self.eval_seed)
        with torch.cuda.device(self.device):
            torch.cuda.manual_seed(self.eval_seed)     
        self.model.eval()
        eval_model = self.model.module if isinstance(self.model, DDP) else self.model
        ckpt_file = os.path.join(self.iteration_dir, 'ckpt.pth.tar')
        assert os.path.exists(ckpt_file), 'evaluate之前必须保存ckpt'
        evaluate_metrics = self.eval_function(model = eval_model,  output_dir = self.iteration_dir)
        if is_dist_avail_and_initialized():
            dist.barrier()
        if comm.is_main_process():
            checkpoint_dict = torch.load(ckpt_file, map_location='cpu')
            ckpt_metrics = checkpoint_dict['metrics']
            to_update_metrics = {} # wandb 新log的value
            for metric_key in evaluate_metrics.keys():
                metric_value = evaluate_metrics[metric_key]
                if metric_key in ckpt_metrics:
                    saved_value = ckpt_metrics[metric_key]
                    if (metric_value - saved_value) > 1e-2:
                        logging.error(f'{metric_key} 存储值 {saved_value} 和 新值{metric_value} 相差太大, 存储了新值, 但是wandb无法更新')
                        to_update_metrics[metric_key] = metric_value
                else:
                    to_update_metrics[metric_key] = metric_value
            checkpoint_dict['metrics'] = evaluate_metrics
            wandb.log(to_update_metrics, step=self.num_samples)
            metric_string = ' '.join([f'{key} : {value:.6f}' for key, value in evaluate_metrics.items()])
            logging.debug(metric_string)
            torch.save(checkpoint_dict, ckpt_file)
            del checkpoint_dict

        if is_dist_avail_and_initialized():
            dist.barrier()

    def load_ckpt(self, 
                  ckpt_path=None, 
                  load_schedule=False, # num_samples, dataload, sampler
                  load_optimize=False,  
                  load_model=False,
                  load_random=False, # 随机状态
                  ):
        """
        load其中一个状态:
        load_model: 模型的状态(load_model, model.load_state_dict), 
        load_optimize: 模型的优化状态(load_optimize, model.load_optimize_state_dict)
        load_random: 代码的随机状态
        load_schedule: 训练的当前节点

        lrsm:     model恢复, random要恢复, optimize不恢复, schedule要恢复
        imseg_pt: model恢复, random不恢复, optimize不恢复, schedule不恢复, 
        resume:   model恢复, random恢复,, optimize恢复,, schedule恢复,, 
        """
        assert os.path.exists(ckpt_path)
        model_without_ddp = self.model.module if isinstance(self.model, DDP) else self.model
        checkpoint = torch.load(ckpt_path, map_location='cpu')

        if load_model:
            model_without_ddp.load_state_dict(checkpoint['model'], strict=True)
            
        if load_optimize:
            model_without_ddp.load_optimize_state_dict(checkpoint['optimizer'])

        if load_schedule:
            self.num_samples = checkpoint['num_samples'] # 已经见过的sample的数量/下一次iteration的第一个sample的下标
            self.num_iterations = checkpoint['num_iterations'] # model更新次数

            sampler = self.train_samplers[0]
            while (sampler.end_idx != None) and (self.num_samples > (sampler.end_idx - 1)):
                self.train_samplers.pop(0)
                self.train_loaders.pop(0)
                sampler = self.train_samplers[0]
            self.train_samplers[0].set_iter_first_sample_idx(self.num_samples)

        if load_random:
            rng_state_dict_by_rank = checkpoint['rng_state_dict_by_rank']
            torch.set_rng_state(rng_state_dict_by_rank[comm.get_rank()]['cpu_rng_state'])
            torch.cuda.set_rng_state(rng_state_dict_by_rank[comm.get_rank()]['gpu_rng_state'], device=self.device)
            np.random.set_state(rng_state_dict_by_rank[comm.get_rank()]['numpy_rng_state'])
            random.setstate(rng_state_dict_by_rank[comm.get_rank()]['py_rng_state'])

        del checkpoint

    def _log(self, 
             loss_dict_unscaled,
             loss_weight,
             lr_group_dicts,
             sample_idxs, 
             iteration_time,):
        """
        每个iteration存储 loss值, 运行时间, sample_idx, 学习率状态
        如果当前iteration要进行evaluate:
            先调用save_ckpt (存储训练的random状态)
            再调用evaluate  (把metric写到ckpt里)
            再调用load_ckpt (回到当时的训练random状态)
        """
        loss_dict_unscaled_reduced = reduce_dict(loss_dict_unscaled) 
        loss_value = sum([loss_dict_unscaled_reduced[key] * loss_weight[key] for key in loss_weight.keys()])

        # logging
        if comm.is_main_process():
            for idx, sp_idx in enumerate(sample_idxs):
                wandb.log({'sample_idx': sp_idx}, step=self.num_samples - len(sample_idxs) + idx)
            
            logger_updates = lr_group_dicts
            logger_updates.update(loss_dict_unscaled_reduced)
            logger_updates.update({
                'loss_value': loss_value,
                'iteration_time': iteration_time,
            })
            self.metric_logger.update(**logger_updates)
            log_string = self.log_header(iteration_time, sample_idxs) + f'\n{str(self.metric_logger)}'
            wandb_log = self.metric_logger.to_dict()
            logging.debug(log_string)
            wandb.log(wandb_log, step=self.num_samples)

        if is_dist_avail_and_initialized():
            dist.barrier()

        # ckpting
        if type(self.ckpted_iters) == int:
            do_ckpt = (self.num_iterations % self.ckpted_iters) == 0
        elif type(self.ckpted_iters) == list:
            do_ckpt = self.num_iterations in self.ckpted_iters
        else:
            raise ValueError()
        if (self.num_iterations % 2000 == 0) or do_ckpt:
            gc.collect()
            torch.cuda.empty_cache()
        if do_ckpt:
            # try: 
            self.save_ckpt()
            self.evaluate()
            self.load_ckpt(os.path.join(self.iteration_dir, 'ckpt.pth.tar'),  # 训练的随机状态
                           load_random=True, load_schedule=False, load_model=False, load_optimize=False,)
            # except:
            #     if comm.is_main_process():
            #         logging.error(f'Iteration {self.num_iterations} evaluate错误')
        if is_dist_avail_and_initialized():
            dist.barrier()


    @property
    def device(self):
        return torch.device(comm.get_local_rank())
    
    # 这些东西一个iteration之后不会变
    @property
    def iteration_dir(self):
        return os.path.join(self.out_dir, f'epc[{self.epoch[-1]}]_iter[{self.num_iterations}]_sap[{self.num_samples}]')
        # num_iteration是见过多少iter, num_samples是见过多少sample; ckpt的名字代表的是见过这么多epoch之后

    @property
    def epoch(self):
        # TODO: 如果真的想要epoch clear的话
        # 当epoch为单位的时候, 0代表第一个epoch的样本, epoch0的ckpt代表这个ckpt 学习完了第一个epoch的样本
        dataset_length = len(self.train_loaders[0].dataset)
        epoch = self.num_samples / dataset_length
        int_part, dec_part = f'{epoch:.2f}'.split('.')
        return epoch, f'{int_part}_{dec_part}'

    def log_header(self, iteration_time, sample_idxs):
        # 只计算一个epoch的估计时间
        one_epoch_iterations = len(self.train_loaders[0].dataset) // len(sample_idxs)
        eta = datetime.timedelta(seconds=one_epoch_iterations * iteration_time)
        return f'Epoch_ETA: [{str(eta)}] Epoch:[{self.epoch[0]:.2f}] Iter: [{(self.num_iterations):06d}] Sample: [{self.num_samples:06d}]'
 
    def visualize_path(self, meta_idxs, visualize):
        # out_dir: config/
        # iteration_dir: config/epc1_iter500_sap8099/ ...
        # epc1_iter500_sap8099/visualize_model/train_meta_0
        # epc1_iter500_sap8099/eval_dataset1/visualize_model/meta_0
        # epc1_iter500_sap8099/eval_dataset1/metrci1/ config_web_epoch.zip, images
        # epc1_iter500_sap8099/eval_dataset1/metric2/
        return [os.path.join(self.iteration_dir, 'visualize_model', f'train_meta_{str(meta_idx)}') if vis else None for (meta_idx, vis) in zip(meta_idxs, visualize)]


class TrainerLogger(MetricLogger):
    def __init__(self, delimiter="\t"):
        super().__init__(delimiter)

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            if k not in self.meters:
                if 'lr_group' in k:
                    self.meters[k] = SmoothedValue(window_size=1,fmt='{value:.8f}', handler='value')
                elif 'iteration_time' in k:
                    self.meters[k] = SmoothedValue(window_size=1,fmt='{value:2f}',handler='value')
                else:
                    self.meters[k] = SmoothedValue()
                    
            self.meters[k].update(v)