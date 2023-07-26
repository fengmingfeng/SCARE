import argparse
import os
import ruamel_yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from models.model_vqa import ALBEF
from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer
import utils
from dataset.utils import save_result
from dataset import create_dataset, create_sampler, create_loader, vqa_collate_fn
from scheduler import create_scheduler
from optim import create_optimizer
import torch.distributed as dist
dist.init_process_group('gloo', init_method='file:///tmp/somefile', rank=0, world_size=1)
def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config):
    # train
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}')) #生成lr与后面配置的键值对
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}')) #loss与SmoothedValue类型的键值对
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50    
    step_size = 100
    warmup_iterations = warmup_steps*step_size  #400
    for i,(image, question, answer, weights, n) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        #         print(image.shape)
        #         print("image1",image)
        #         print("weights1",weights)
        #         print(question)
        #         print("answer", answer)
        #         print("weights", weights)
        #         print("n", n)
        image, weights = image.to(device,non_blocking=True), weights.to(device,non_blocking=True)
        #         print("image",image)
        #         print("weights",weights)
        question_input = tokenizer(question, padding='longest', truncation=True, max_length=25, return_tensors="pt").to(device)
        #         print(question_input)
        answer_input = tokenizer(answer, padding='longest', return_tensors="pt").to(device)
        #         print("config['warm_up']", config['warm_up']) True
        if epoch>0 or not config['warm_up']:
            alpha = config['alpha']
            #             print("alpha",alpha)未进去
        else:
            alpha = config['alpha']*min(1,i/len(data_loader))
            # print("alpha", alpha)
        loss = model(image, question_input, answer_input, train=True, alpha=alpha, k=n, weights=weights)
        optimizer.zero_grad()#清空过往梯度
        loss.backward() #反向传播，计算当前梯度
        optimizer.step()    #根据梯度更新网络参数
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if epoch==0 and i%step_size==0 and i<=warmup_iterations: 
            scheduler.step(i//step_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}
@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, config) :
    # test
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generate VQA test result:'
    print_freq = 50
    result = []
    answer_list = [answer+config['eos'] for answer in data_loader.dataset.answer_list]
    answer_input = tokenizer(answer_list, padding='longest', return_tensors='pt').to(device)
    for n, (image, question, question_id) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):        
        image = image.to(device,non_blocking=True)             
        question_input = tokenizer(question, padding='longest', return_tensors="pt").to(device)
        topk_ids, topk_probs = model(image, question_input, answer_input, train=False, k=config['k_test'])#[16,128]
        for ques_id, topk_id, topk_prob in zip(question_id, topk_ids, topk_probs):
#             ques_id = int(ques_id.item())
            ques_id = int(ques_id)
            _, pred = topk_prob.max(dim=0)
            # print("pred", pred)
            # k=topk_id[pred]
            result.append({"question_id":ques_id, "answer":data_loader.dataset.answer_list[topk_id[pred]]})
    return result
def main(args, config):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    start_epoch = 0
    #读取schedular中的epoch系数作为我们的最大epoch
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']
    #### Dataset #### 
    print("Creating vqa datasets")
    datasets = create_dataset('vqa', config)
    #datasets是包含训练集和测试集两部分处理后数据的元组
    print(type(datasets))
    print(len(datasets))
    #print("args.distributed",args.distributed) 为False
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler(datasets, [True, False], num_tasks, global_rank)         
    else:
        print("未使用分布式")
        samplers = [None, None]
    #由于显存溢出，所以将batch_size变小，设为16
    # train_loader, test_loader = create_loader(datasets,samplers,
    #                                           batch_size=[config['batch_size_train'],config['batch_size_test']],
    #                                           num_workers=[4,4],is_trains=[True, False],
    #                                           collate_fns=[vqa_collate_fn,None])
    # print(vqa_collate_fn)
    #函数名vqa_collate_fn作为参数传入，在进入函数后可以直接使用传入的参数调用
    #返回训练和测试的dataloader
    train_loader, test_loader = create_loader(datasets, samplers,
                                              batch_size=[16, 16],
                                              num_workers=[4, 4], is_trains=[True, False],
                                              collate_fns=[vqa_collate_fn, None])
    '''
    dataloader的内部属性
    loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,#shuffle用于打乱数据
            collate_fn=collate_fn,
            drop_last=drop_last,#若drop_lase为true，则每个epoch在加载数据时不会利用不能完整划分为batch_size的数据；如batch-size=20,共有213个数据，则有13个数据不能使用
        )  
    '''
    # data1 = train_loader.dataset
    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)
    #### Model #### 
    print("Creating model")
    model = ALBEF(config=config, text_encoder=args.text_encoder, text_decoder=args.text_decoder, tokenizer=tokenizer)
    print("model", model)
    model = model.to(device)
    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)
    print("args.checkpoint", args.checkpoint)
    if args.checkpoint:    
        checkpoint = torch.load(args.checkpoint, map_location='cpu') #加载模型参数
        state_dict = checkpoint['model']
        # reshape positional embedding to accomodate for image resolution change visual_encoder.pos_embed为[1,257,768],下面将其变为[1,577,768]
        pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder)         
        state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
        if not args.evaluate:
            if config['distill']:
                m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],model.visual_encoder_m)   
                state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped    #同上
            keys=state_dict.keys()
            for key in list(state_dict.keys()):
                if 'bert' in key:#bert不在里面
                    encoder_key = key.replace('bert.','')         
                    state_dict[encoder_key] = state_dict[key] 
                # intialize text decoder as multimodal encoder (last 6 layers of model.text_encoder)    ,取encoder后6层作为多模态编码器，前6层删除，后6层的层数删除
                if 'text_encoder' in key:                
                    if 'layer' in key:
                        encoder_keys = key.split('.')
                        layer_num = int(encoder_keys[4])#TCL_4M.pth范围为0-11
#                         print(type(state_dict[key]))
                        if layer_num<6:
                            del state_dict[key]  #删除layer_num为6的值
                            continue
                        else:
                            decoder_layer_num = (layer_num-6)#大于6,则layer-6
                            encoder_keys[4] = str(decoder_layer_num)
                            encoder_key = '.'.join(encoder_keys)  #encoder_key为更新完layer的值
                    else:
                        encoder_key = key
                    decoder_key = encoder_key.replace('text_encoder','text_decoder')  #如用text_decoder.bert.encoder.layer.0.attention.self.query.weight替换text_encoder.bert.encoder.layer.0.attention.self.query.weight
                    state_dict[decoder_key] = state_dict[key]
                    # print("state_dict[decoder_key]", state_dict[decoder_key])
                    # print(key)
                    del state_dict[key]
        keys=state_dict.keys()
        msg = model.load_state_dict(state_dict,strict=False) #把模型参数加载到模型中，将参数和缓冲拷贝到当前这个模块及其子模块中,设置为False时，一些缺失值不用给
        print('load checkpoint from %s'%args.checkpoint)
        print(msg)
    model_without_ddp = model
    print("args.distributed", args.distributed)#False
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    print("Start training")
    start_time = time.time()
    for epoch in range(start_epoch, max_epoch):
        if epoch>0:
            lr_scheduler.step(epoch+warmup_steps)
        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            train_stats = train(model, train_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler, config)
        if args.evaluate:
            break
        if utils.is_main_process():
            print("进入is_main_process()")
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                        }                
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")
            save_obj = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'config': config,
                'epoch': epoch,
            }
            torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_%02d.pth'%epoch))
        dist.barrier()
    vqa_result = evaluation(model, test_loader, tokenizer, device, config)        
    result_file = save_result(vqa_result, args.result_dir, 'vqa_result_epoch%d'%epoch)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/VQA.yaml') 
    parser.add_argument('--checkpoint', default='./data/pretrain/TCL_4M.pth')
    parser.add_argument('--output_dir', default='output/vqa')
    parser.add_argument('--evaluate', action='store_true', default=False)
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--text_decoder', default='bert-base-uncased')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    #os.path.join()从第一个斜杠开始的地方进行路径拼接
    args.result_dir = os.path.join(args.output_dir, 'result')
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
    #将config中的内容全部写入后面的，output/vqa/config.yaml中
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))
    main(args, config)