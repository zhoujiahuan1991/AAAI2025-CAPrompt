"""
Train and eval functions used in main.py
"""
import math
import sys
import os
import datetime
import json
from typing import Iterable
from pathlib import Path
import copy
import torch
import torch.distributed as dist
import numpy as np
from torch.autograd import Variable
from timm.utils import accuracy
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from torch import optim
import utils
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F
import math
import time
 

def linear_constraint(prompt,task_id,args):
    layer_num,kv,pool_size,length,head,dim = prompt.shape
    prompt = prompt.reshape(layer_num,kv,pool_size,length,-1)
    prompt_0 = prompt[:,:,0].detach().clone()
    prompt_before = prompt[:,:,task_id-1].detach().clone()
    prompt_this = prompt[:,:,task_id]
    delta_1 = prompt_before - prompt_0
    delta_2 = prompt_this - prompt_0
    sim = F.cosine_similarity(delta_1,delta_2,dim=-1)
    loss_sim = (-sim +1) 
    return torch.mean(loss_sim)

        

def train_one_epoch(model: torch.nn.Module, 
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    set_training_mode=True, task_id=-1, class_mask=None, target_task_map=None, args=None, ):
    model.train(set_training_mode)
    class_per_task = args.nb_classes // args.num_tasks
    if args.distributed and utils.get_world_size() > 1:
        data_loader.sampler.set_epoch(epoch)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = f'Train: Epoch[{epoch + 1:{int(math.log10(args.epochs)) + 1}}/{args.epochs}]'
    loss_con = torch.zeros(1)
    #for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
    for input, target in data_loader:
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if task_id>0:
            with torch.no_grad():
                
                prompt_weight = torch.ones(input.shape[0],args.num_tasks,class_per_task).cuda()
                prompt_weight = torch.mean(prompt_weight,dim=-1)
                prompt_weight[:,task_id+1:] = 0

                prompt_weight = norm(prompt_weight)
                output = model(input, task_id=task_id, train=set_training_mode,prompt_weight=prompt_weight)
                logits = output['logits']

                prompt_weight = F.softmax(logits,dim=-1)
                prompt_weight = prompt_weight.reshape(input.shape[0],-1,class_per_task)
                prompt_weight = torch.mean(prompt_weight,dim=-1)
                prompt_weight[:,task_id+1:] = 0

                prompt_weight = norm(prompt_weight)
                
                prompt_weight_new_sum = prompt_weight[:,task_id]
                prompt_weight_old_sum = torch.sum( prompt_weight[:,:task_id] ,dim=-1)
                prompt_weight_old = prompt_weight.clone() 
                prompt_weight_old[:,task_id:] = 0
                prompt_weight_old = norm(prompt_weight_old)
                prompt_weight_new = prompt_weight.clone() 
                prompt_weight_new[:,:task_id] = 0
                prompt_weight_new[:,task_id+1:] = 0
                prompt_weight_new = norm(prompt_weight_new)
                output_old = model(input, task_id=task_id,  train=set_training_mode,prompt_weight=prompt_weight_old)
                logits_old = output_old['logits']

            output_new = model(input, task_id=task_id,prompt_weight=prompt_weight_new , train=set_training_mode)
            output = model(input, task_id=task_id,  train=set_training_mode,prompt_weight=prompt_weight)
            
            
            logits_mix = output['logits_detach']
            logits = output['logits']
            logits_new_mix = output_new['logits_detach']
            logits_new = output_new['logits']
            #print(prompt_weight_new_sum+prompt_weight_old_sum)
            logits_mix[:,:task_id*class_per_task] = logits_mix[:,:task_id*class_per_task].detach().clone()
            logits_new_mix[:,:task_id*class_per_task] = logits_new_mix[:,:task_id*class_per_task].detach().clone()
            prob = F.softmax(logits_mix,dim=-1)
            prob_old = F.softmax(logits_old,dim=-1).detach().clone()
            prob_new = F.softmax(logits_new_mix,dim=-1)#.detach().clone()
            select_index = torch.arange(prob_new.shape[0])
            delta = prob_new[select_index,target] * prompt_weight_new_sum \
                + prob_old[select_index,target] * prompt_weight_old_sum \
                - prob[select_index,target]
            delta[delta<0]=0
            loss_con = torch.mean(delta)*args.delta_weight
            # here is the trick to mask out classes of non-current tasks
            if args.train_mask and class_mask is not None:
                mask = class_mask[task_id]
                not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
                logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))
                logits_new = logits_new.index_fill(dim=1, index=not_mask, value=float('-inf'))
            
            loss = criterion(logits, target)  # base criterion (CrossEntropyLoss)

            loss = loss + loss_con

        else :
            prompt_weight = torch.ones(input.shape[0],args.num_tasks,class_per_task).cuda()
            prompt_weight = torch.mean(prompt_weight,dim=-1)
            prompt_weight[:,task_id+1:] = 0

            prompt_weight = norm(prompt_weight)
            output = model(input, task_id=task_id, train=set_training_mode,prompt_weight=prompt_weight)
            logits = output['logits']
            if args.train_mask and class_mask is not None:
                mask = class_mask[task_id]
                not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
                logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

            loss = criterion(logits, target)  # base criterion (CrossEntropyLoss)

        acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        if task_id >= 2 :
            #loss_linear =  linear_constraint(model.e_prompt.prompt,task_id,args) * args.penalty_weight
            if args.distributed:
                loss_linear =  model.module.linear_constrain(task_id,args) * args.penalty_weight
            else :
                loss_linear =  model.linear_constrain(task_id,args) * args.penalty_weight
            loss += loss_linear
        else :
            loss_linear = torch.zeros_like(loss)
        
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        torch.cuda.synchronize()
        metric_logger.update(Loss=loss.item())
        metric_logger.update(Loss_l=loss_linear.item())
        metric_logger.update(Loss_con=loss_con.item())
        metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
        metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def norm (feature):
    feature_norm = torch.sum(feature,dim=-1)
    feature = feature/feature_norm.unsqueeze(-1)
    return feature

def prompt_id_from_logit (logits,args,task_id,class_mask,device,target_task_map):
    if args.train_mask and class_mask is not None:
        mask = []
        for id in range(task_id + 1):
            mask.extend(class_mask[id])
        not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
        not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
        logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))
    prompt_id = torch.max(logits, dim=1)[1]
    # translate cls to task_id
    prompt_id = torch.tensor([target_task_map[v.item()] for v in prompt_id], device=device).unsqueeze(
        -1)
    return prompt_id



@torch.no_grad()
def evaluate(model: torch.nn.Module, data_loader,
             device, task_id=-1, class_mask=None, target_task_map=None, args=None, ):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    #header = 'Test: [Task {}]'.format(i + 1)

    # switch to evaluation mode
    model.eval()
    class_per_task = args.nb_classes // args.num_tasks
    feature_all = []
    with torch.no_grad():
        #for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
        for input, target in data_loader:
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output
            

            prompt_weight = torch.ones(input.shape[0],args.num_tasks,class_per_task).cuda()
            prompt_weight = torch.mean(prompt_weight,dim=-1)
            prompt_weight[:,task_id+1:] = 0
            prompt_weight = norm(prompt_weight)



            output = model(input,prompt_weight=prompt_weight)
            logits = output['logits']
            prompt_id2 = prompt_id_from_logit(logits,args,task_id,class_mask,device,target_task_map)
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            metric_logger.meters['Acc@cyc1'].update(acc1.item(), n=input.shape[0])
            task_inference_acc = utils.task_inference_accuracy(prompt_id2, target, target_task_map)
            metric_logger.meters['Acc@task'].update(task_inference_acc.item(), n=input.shape[0])
            
            prompt_weight2 = F.softmax(logits,dim=-1)
            prompt_weight2 = prompt_weight2.reshape(input.shape[0],-1,class_per_task)
            prompt_weight2 = torch.mean(prompt_weight2,dim=-1)
            prompt_weight2[:,task_id+1:] = 0

            prompt_weight2 = norm(prompt_weight2)
        
            output2 = model(input, prompt_weight=prompt_weight2)
            logits2 = output2['logits']
            acc1, acc5 = accuracy(logits2, target, topk=(1, 5))
            metric_logger.meters['Acc@cyc2'].update(acc1.item(), n=input.shape[0])
            if args.cycle_num > 2:
                logitsn = logits2
                for _ in range(args.cycle_num-2) :  
                    prompt_weight = F.softmax(logitsn,dim=-1)
                    prompt_weight = prompt_weight.reshape(input.shape[0],-1,class_per_task)
                    prompt_weight = torch.mean(prompt_weight,dim=-1)
                    prompt_weight[:,task_id+1:] = 0

                    prompt_weight = norm(prompt_weight)
                    outputn = model(input, prompt_weight=prompt_weight)
                    logitsn = outputn['logits']
                acc1, acc5 = accuracy(logitsn, target, topk=(1, 5))
            metric_logger.meters['Acc@cycn'].update(acc1.item(), n=input.shape[0])
    # gather the stats from all processes

    metric_logger.synchronize_between_processes()
    print(
        '* Acc@task {task.global_avg:.3f} Acc@cyc1 {cyc1.global_avg:.3f} Acc@pw2 {cyc2.global_avg:.3f} Acc@pwn {cycn.global_avg:.3f}'
        .format(task=metric_logger.meters['Acc@task'],cyc1=metric_logger.meters['Acc@cyc1'], cyc2=metric_logger.meters['Acc@cyc2'], cycn=metric_logger.meters['Acc@cycn']))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_till_now(model: torch.nn.Module, data_loader,
                      device, task_id=-1, class_mask=None, target_task_map=None, acc_matrix=None, args=None, acc_matrix_accn=None):
    stat_matrix = np.zeros((4, args.num_tasks))  # 3 for Acc@1, Acc@5, Loss
    start_time = time.time()
    for i in range(task_id + 1):
        test_stats = evaluate(model=model, data_loader=data_loader[i]['val'],
                              device=device,  task_id=task_id, class_mask=class_mask, target_task_map=target_task_map,
                              args=args)


        stat_matrix[0, i] = test_stats['Acc@cyc1']
        stat_matrix[1, i] = test_stats['Acc@cyc2']
        stat_matrix[2, i] = test_stats['Acc@task']
        stat_matrix[3, i] = test_stats['Acc@cycn']

        acc_matrix[i, task_id] = test_stats['Acc@cyc2']
        acc_matrix_accn[i, task_id] = test_stats['Acc@cycn']
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=(total_time)))
    print(f"Total test time: {total_time_str}")
    avg_stat = np.divide(np.sum(stat_matrix, axis=1), task_id + 1)

    diagonal = np.diag(acc_matrix)

    result_str = "[Average accuracy till task{}]\tAcc@task: {:.4f}\tAcc@cyc1: {:.4f}\tAcc@cyc2: {:.4f}\tAcc@cycn: {:.4f}".format(
        task_id + 1,
        avg_stat[2],
        avg_stat[0],
        avg_stat[1],
        avg_stat[3],)
    if task_id > 0:
        forgetting = np.mean((np.max(acc_matrix, axis=1) -
                              acc_matrix[:, task_id])[:task_id])
        forgetting_accn = np.mean((np.max(acc_matrix_accn, axis=1) -
                              acc_matrix_accn[:, task_id])[:task_id])
        backward = np.mean((acc_matrix[:, task_id] - diagonal)[:task_id])
        acc_all = 0 
        for t in range(task_id+1) :
            acc_all += (acc_matrix[:, t][:t+1]).mean()
        acc_all = acc_all / (task_id+1)
        result_str += "\tForgetting: {:.4f}\tAFn: {:.4f}\tBackward: {:.4f}\tAAC: {:.4f}".format(forgetting,forgetting_accn ,backward,acc_all)
    print(result_str)

    return test_stats


def train_and_evaluate(model: torch.nn.Module, model_without_ddp: torch.nn.Module, 
                       criterion, data_loader: Iterable, data_loader_per_cls: Iterable,
                       optimizer: torch.optim.Optimizer,
                       lr_scheduler,
                       device: torch.device,
                       class_mask=None, target_task_map=None, args=None, ):
    # create matrix to save end-of-task accuracies
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
    acc_matrix_accn = np.zeros((args.num_tasks, args.num_tasks))
    pre_ca_acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
    pre_ca_acc_matrix_accn = np.zeros((args.num_tasks, args.num_tasks))
    global cls_mean
    global cls_cov
    cls_mean = dict()
    cls_cov = dict()

    for task_id in range(args.num_tasks):
        # Create new optimizer for each task to clear optimizer status
        if task_id > 0 and args.reinit_optimizer:
            if args.larger_prompt_lr:
                # This is a simple yet effective trick that helps to learn task-specific prompt better.
                base_params = [p for name, p in model_without_ddp.named_parameters() if
                            'prompt' in name and p.requires_grad == True]
                base_fc_params = [p for name, p in model_without_ddp.named_parameters() if
                                'prompt' not in name and p.requires_grad == True]
                base_params = {'params': base_params, 'lr': args.lr, 'weight_decay': args.weight_decay}
                base_fc_params = {'params': base_fc_params, 'lr': args.lr * 0.1, 'weight_decay': args.weight_decay}
                network_params = [base_params, base_fc_params]
                optimizer = create_optimizer(args, network_params)
            else:
                optimizer = create_optimizer(args, model)
            
            if args.sched != 'constant':
                lr_scheduler, _ = create_scheduler(args, optimizer)
            elif args.sched == 'constant':
                lr_scheduler = None

        # if model already trained
        checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id + 1))
        if task_id < args.ckpt_num :
            if args.ckpt_num>0 and task_id < args.ckpt_num-1 :
                continue 
            resume = True
            load_path = os.path.join(args.trained_caprompt_model, 'checkpoint/task{}_checkpoint.pth'.format(task_id + 1))

            if os.path.exists(load_path):
                print('Loading checkpoint from:', load_path)
                checkpoint = torch.load(load_path, map_location=device)
                model.load_state_dict(checkpoint['model'])
            else:
                print('No checkpoint found at:', load_path)
                return
            
        else :
            resume = False

            if args.prompt_pool and args.shared_prompt_pool :
                if task_id > 0:
                    prev_start = (task_id - 1) * args.top_k
                    prev_end = task_id * args.top_k

                    cur_start = prev_end
                    cur_end = (task_id + 1) * args.top_k

                    if (prev_end > args.size) or (cur_end > args.size):
                        pass
                    else:
                        cur_idx = (
                            slice(None), slice(None), slice(cur_start, cur_end)) if args.use_prefix_tune_for_e_prompt else (
                            slice(None), slice(cur_start, cur_end))
                        prev_idx = (
                            slice(None), slice(None),
                            slice(prev_start, prev_end)) if args.use_prefix_tune_for_e_prompt else (
                            slice(None), slice(prev_start, prev_end))

                        with torch.no_grad():
                            if args.distributed:
                                model.module.e_prompt.prompt.grad.zero_()
                                model.module.e_prompt.prompt[cur_idx] = model.module.e_prompt.prompt[prev_idx]
                                # optimizer.param_groups[0]['params'] = model.module.parameters()
                            else:
                                if model.e_prompt.prompt.grad != None :
                                    model.e_prompt.prompt.grad.zero_()
                                model.e_prompt.prompt[cur_idx] = model.e_prompt.prompt[prev_idx]
                                # optimizer.param_groups[0]['params'] = model.parameters()

            for epoch in range(args.epochs):
                train_stats = train_one_epoch(model=model, criterion=criterion,
                                                data_loader=data_loader[task_id]['train'], optimizer=optimizer,
                                                device=device, epoch=epoch, max_norm=args.clip_grad,
                                                set_training_mode=True, task_id=task_id, class_mask=class_mask,
                                                target_task_map=target_task_map, args=args, )

                if lr_scheduler:
                    lr_scheduler.step(epoch)

            if args.prompt_momentum > 0 and task_id > 0:
                if args.use_prefix_tune_for_e_prompt:
                    with torch.no_grad():
                        print(model.e_prompt.prompt[:, :, task_id].shape)
                        print(
                            model.e_prompt.prompt[:, :, 0:task_id].detach().clone().mean(dim=2, keepdim=True).shape)
                        model.e_prompt.prompt[:, :, task_id].copy_(
                            (1 - args.prompt_momentum) * model.e_prompt.prompt[:, :, task_id].detach().clone()
                            + args.prompt_momentum * model.e_prompt.prompt[:, :, 0:task_id].detach().clone().mean(
                                dim=2))

            # compute mean and variance
        _compute_mean(model=model, data_loader=data_loader_per_cls, device=device, task_id=task_id,
                      class_mask=class_mask[task_id], args=args)

        if task_id > 0 and not args.not_train_ca and not resume:
            pre_ca_test_stats = evaluate_till_now(model=model, data_loader=data_loader,
                                                  device=device,
                                                  task_id=task_id, class_mask=class_mask,
                                                  target_task_map=target_task_map,
                                                  acc_matrix=pre_ca_acc_matrix, args=args,acc_matrix_accn=pre_ca_acc_matrix_accn)

            train_task_adaptive_prediction(model, args, device, class_mask, task_id)

        test_stats = evaluate_till_now(model=model, data_loader=data_loader,
                                       device=device,
                                       task_id=task_id, class_mask=class_mask, target_task_map=target_task_map,
                                       acc_matrix=acc_matrix, args=args,acc_matrix_accn=acc_matrix_accn)

        if args.output_dir and utils.is_main_process():
            Path(os.path.join(args.output_dir, 'checkpoint')).mkdir(parents=True, exist_ok=True)

            checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id + 1))
            state_dict = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'args': args,
            }
            if args.sched is not None and args.sched != 'constant':
                state_dict['lr_scheduler'] = lr_scheduler.state_dict()

            utils.save_on_master(state_dict, checkpoint_path)
        if not resume:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        }

            if args.output_dir and utils.is_main_process():
                with open(os.path.join(args.output_dir,
                                    '{}_stats.txt'.format(datetime.datetime.now().strftime('log_%Y_%m_%d_%H_%M'))),
                        'a') as f:
                    f.write(json.dumps(log_stats) + '\n')


@torch.no_grad()
def _compute_mean(model: torch.nn.Module, data_loader: Iterable, device: torch.device, task_id, class_mask=None,
                  args=None, ):
    model.eval()

    class_per_task = args.nb_classes // args.num_tasks
    

    for cls_id in class_mask:
        data_loader_cls = data_loader[cls_id]['train']
        features_per_cls = []
        for i, (inputs, targets) in enumerate(data_loader_cls):
            inputs = inputs.to(device, non_blocking=True)
            prompt_weight = torch.ones(inputs.shape[0],args.num_tasks,class_per_task).cuda()
            prompt_weight = torch.mean(prompt_weight,dim=-1)
            prompt_weight[:,:task_id] = 0
            prompt_weight[:,task_id+1:] = 0
            prompt_weight = norm(prompt_weight)
            features = model(inputs, task_id=task_id,prompt_weight=prompt_weight, train=True)['pre_logits']
            features_per_cls.append(features)
        features_per_cls = torch.cat(features_per_cls, dim=0)
        features_per_cls_list = [torch.zeros_like(features_per_cls, device=device) for _ in range(args.world_size)]
        try :
            dist.barrier()
            dist.all_gather(features_per_cls_list, features_per_cls)
        except Exception as e :
            features_per_cls_list = [features_per_cls]

        if args.ca_storage_efficient_method == 'covariance':
            features_per_cls = torch.cat(features_per_cls_list, dim=0)
            # print(features_per_cls.shape)
            cls_mean[cls_id] = features_per_cls.mean(dim=0)
            cls_cov[cls_id] = torch.cov(features_per_cls.T) + (torch.eye(cls_mean[cls_id].shape[-1]) * 1e-4).to(device)
        
        if args.ca_storage_efficient_method == 'variance':
            features_per_cls = torch.cat(features_per_cls_list, dim=0)
            # print(features_per_cls.shape)
            cls_mean[cls_id] = features_per_cls.mean(dim=0)
            cls_cov[cls_id] = torch.diag(torch.cov(features_per_cls.T) + (torch.eye(cls_mean[cls_id].shape[-1]) * 1e-4).to(device))
        if args.ca_storage_efficient_method == 'multi-centroid':
            from sklearn.cluster import KMeans
            n_clusters = args.n_centroids
            features_per_cls = torch.cat(features_per_cls_list, dim=0).cpu().numpy()
            kmeans = KMeans(n_clusters=n_clusters)
            kmeans.fit(features_per_cls)
            cluster_lables = kmeans.labels_
            cluster_means = []
            cluster_vars = []
            for i in range(n_clusters):
               cluster_data = features_per_cls[cluster_lables == i]
               cluster_mean = torch.tensor(np.mean(cluster_data, axis=0), dtype=torch.float64).to(device)
               cluster_var = torch.tensor(np.var(cluster_data, axis=0), dtype=torch.float64).to(device)
               cluster_means.append(cluster_mean)
               cluster_vars.append(cluster_var)
            
            cls_mean[cls_id] = cluster_means
            cls_cov[cls_id] = cluster_vars


def train_task_adaptive_prediction(model: torch.nn.Module, args, device, class_mask=None, task_id=-1):
    model.train()
    run_epochs = args.crct_epochs
    crct_num = 0
    param_list = [p for n, p in model.named_parameters() if p.requires_grad and 'prompt' not in n]
    network_params = [{'params': param_list, 'lr': args.ca_lr, 'weight_decay': args.weight_decay}]
    if 'mae' in args.model or 'beit' in args.model:
        optimizer = optim.AdamW(network_params, lr=args.ca_lr / 10, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(network_params, lr=args.ca_lr, momentum=0.9, weight_decay=5e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=run_epochs)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    for i in range(task_id):
        crct_num += len(class_mask[i])

    # TODO: efficiency may be improved by encapsulating sampled data into Datasets class and using distributed sampler.
    for epoch in range(run_epochs):

        sampled_data = []
        sampled_label = []
        num_sampled_pcls = args.batch_size * 5

        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

        if args.ca_storage_efficient_method in ['covariance', 'variance']:
            for i in range(task_id + 1):
                for c_id in class_mask[i]:
                    mean = torch.tensor(cls_mean[c_id], dtype=torch.float64).to(device)
                    cov = cls_cov[c_id].to(device)
                    if args.ca_storage_efficient_method == 'variance':
                        cov = torch.diag(cov)
                    m = MultivariateNormal(mean.float(), cov.float())
                    sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
                    sampled_data.append(sampled_data_single)

                    sampled_label.extend([c_id] * num_sampled_pcls)

        elif args.ca_storage_efficient_method == 'multi-centroid':
            for i in range(task_id + 1):
               for c_id in class_mask[i]:
                   for cluster in range(len(cls_mean[c_id])):
                       mean = cls_mean[c_id][cluster]
                       var = cls_cov[c_id][cluster]
                       if var.mean() == 0:
                           continue
                       m = MultivariateNormal(mean.float(), (torch.diag(var) + 1e-4 * torch.eye(mean.shape[0]).to(mean.device)).float())
                       sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
                       sampled_data.append(sampled_data_single)
                       sampled_label.extend([c_id] * num_sampled_pcls)
        else:
            raise NotImplementedError


        sampled_data = torch.cat(sampled_data, dim=0).float().to(device)
        sampled_label = torch.tensor(sampled_label).long().to(device)
        #print(sampled_data.shape)

        inputs = sampled_data
        targets = sampled_label

        sf_indexes = torch.randperm(inputs.size(0))
        inputs = inputs[sf_indexes]
        targets = targets[sf_indexes]

        for _iter in range(crct_num):
            inp = inputs[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]
            tgt = targets[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]
            outputs = model(inp, fc_only=True)
            logits = outputs['logits']

            if args.train_mask and class_mask is not None:
                mask = []
                for id in range(task_id + 1):
                    mask.extend(class_mask[id])
                # print(mask)
                not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
                logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

            loss = criterion(logits, tgt)  # base criterion (CrossEntropyLoss)
            acc1, acc5 = accuracy(logits, tgt, topk=(1, 5))

            if not math.isfinite(loss.item()):
                print("Loss is {}, stopping training".format(loss.item()))
                sys.exit(1)

            optimizer.zero_grad()
            loss.backward()
            #for name, p in model.named_parameters():
            #    if p.requires_grad and p.grad is None:
            #        print(name)
            optimizer.step()
            torch.cuda.synchronize()

            metric_logger.update(Loss=loss.item())
            metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
            metric_logger.meters['Acc@1'].update(acc1.item(), n=inp.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=inp.shape[0])

            # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        scheduler.step()


