# -------------------------------------#
#       对数据集进行训练
# -------------------------------------#
import datetime
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.efficient_vrnet import EfficientVRNet
from nets.yolo_training import (ModelEMA, YOLOLoss, get_lr_scheduler,
                                set_optimizer_lr, weights_init)
# from loss.multitaskloss import HUncertainty
from utils.callbacks_eval import LossHistory, EvalCallback
from utils_seg.callbacks import EvalCallback as EvalCallback_seg
from utils_seg.callbacks import LossHistory as LossHistory_seg
from utils.dataloader import YoloDataset, yolo_dataset_collate
from utils.utils import get_classes, show_config
from utils.utils_fit_v4_miou import fit_one_epoch
import argparse
import wandb


if __name__ == "__main__":
    # =========== 参数解析实例 =========== #
    parser = argparse.ArgumentParser()

    # 添加参数解析
    parser.add_argument("--cuda", type=str, default="True")
    parser.add_argument("--ddp", type=str, default="False")
    parser.add_argument("--model_path", type=str, default='/root/autodl-tmp/EfficientVRNet/EfficientVRNe/weights/best_epoch_weights_ep097_mAP500.791_mAP50-950.487_mIoU0.784.pth')
    parser.add_argument("--fp16", type=str, default="True")
    parser.add_argument("--phi", type=str, default='nano')
    parser.add_argument("--resolution", type=int, default=320)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--lr_init", type=float, default=1e-2)
    parser.add_argument("--lr_decay", type=str, default="cos")
    parser.add_argument("--opt", type=str, default='adam')
    parser.add_argument("--nw", type=int, default=4)
    parser.add_argument("--dice", type=str, default="True")
    parser.add_argument("--focal", type=str, default="True")
    parser.add_argument("--data_root", type=str, default='/root/autodl-tmp/WaterScenes')
    parser.add_argument("--save_dir", type=str, default='/root/autodl-tmp/EfficientVRNet')
    parser.add_argument('--wandb_path', type=str, default='/root/autodl-tmp/EfficientVRNet/wandb', help='path of saving wandb files locally')
    parser.add_argument('--wandb_name', type=str, default='EfficientVRNe',
                        help='name of current training procedure of wandb')
    parser.add_argument('--description', type=str, default=
                        'Achelous++ with uncertainty aware cross attention for fusion(vision only), ' \
                        # 'baseline of Achelous++' \
    'cross-attention with soft gate, ' \
    'the fused features are both inputed into detection and segmentation branches. ' \
    'Introduce pixel-wise uncertainty maps into loss calculation of YOLO Loss instead of mean'
    'training from scratch, test training, ' \
    'four channels of radar features(range, elevation, velocity, and power),' \
    # 'without pier class' \
    '',
                        help='version description of the being trained model')

    args = parser.parse_args()

    # ==================================== #

    # ---------------------------------#
    #   Cuda    是否使用Cuda
    #           没有GPU可以设置成False
    # ---------------------------------#
    Cuda = True if args.cuda == 'True' else False

    # ---------------------------------------------------------------------#
    distributed = True if args.ddp == 'True' else False
    # ---------------------------------------------------------------------#
    #   sync_bn     是否使用sync_bn，DDP模式多卡可用
    # ---------------------------------------------------------------------#
    sync_bn = False
    # ---------------------------------------------------------------------#
    #   classes_path    指向model_data下的txt，与自己训练的数据集相关
    #                   训练前一定要修改classes_path，使其对应自己的数据集
    # ---------------------------------------------------------------------#
    # classes_path = 'model_data/waterscenes_benchmark_wo_pier.txt'
    classes_path = 'model_data/waterscenes_benchmark_ship_only.txt'
    model_path = args.model_path

    # ------------------------------------------------------#
    #   input_shape     all models support 320*320, all models except mobilevit support 416*416
    # ------------------------------------------------------#
    input_shape = [args.resolution, args.resolution]
    # ------------------------------------------------------#
    #   The size of model, three options: S0, S1, S2
    # ------------------------------------------------------#
    phi = args.phi
    # ------------------------------------------------------#

    # ------------------------------------------------------------------#
    #   eval_flag       是否在训练时进行评估，评估对象为验证集
    #                   安装pycocotools库后，评估体验更佳。
    #   eval_period     代表多少个epoch评估一次，不建议频繁的评估
    #                   评估需要消耗较多的时间，频繁评估会导致训练非常慢
    #   此处获得的mAP会与get_map.py获得的会有所不同，原因有二：
    #   （一）此处获得的mAP为验证集的mAP。
    #   （二）此处设置评估参数较为保守，目的是加快评估速度。
    # ------------------------------------------------------------------#
    eval_flag = True
    eval_period = 1
    # ------------------------------------------------------------------#
    #   num_workers     用于设置是否使用多线程读取数据
    #                   开启后会加快数据读取速度，但是会占用更多内存
    #                   内存较小的电脑可以设置为2或者0
    # ------------------------------------------------------------------#
    num_workers = args.nw

    # ========================================  Dataset Path =========================================== #
    # ----------------------------------------------------#
    # 雷达feature map路径
    # ----------------------------------------------------#
    radar_file_path = args.data_root + "/radar/VOCradar320_v2"

    # ----------------------------------------------------#
    #   获得目标检测图片路径和标签
    # ----------------------------------------------------#
    # train_annotation_path = args.data_root + '/MIPC_wo_pier/2007_train.txt'
    # val_annotation_path = args.data_root + '/MIPC_wo_pier/2007_val.txt'
    train_annotation_path = args.data_root + '/MIPC_shipOnly/2007_train.txt'
    val_annotation_path = args.data_root + '/MIPC_shipOnly/2007_val.txt'
    test_annotation_path = args.data_root + '/MIPC_shipOnly/2007_test.txt'

    # ----------------------------------------------------#
    #   jpg图像路径
    # ----------------------------------------------------#
    jpg_path = args.data_root + "/images"

    # ------------------------------------------------------------------#
    # 语义分割数据集路径
    # ------------------------------------------------------------------#
    se_seg_path = args.data_root + "/semantic/SegmentationClass"

    # ================================================================================================== #

    # ============================ segmentation hyperparameters ============================= #
    # -----------------------------------------------------#
    #   num_classes     训练自己的数据集必须要修改的
    #                   自己需要的分类个数+1，如2+1
    # -----------------------------------------------------#
    num_classes_seg = 9

    # ------------------------------------------------------------------#
    #   save_dir_seg        日志文件保存的文件夹
    # ------------------------------------------------------------------#
    save_dir = os.path.join(os.path.join(args.save_dir, args.wandb_name), 'logs_detection')
    save_dir_seg = os.path.join(os.path.join(args.save_dir, args.wandb_name), 'logs_seg')


    # ======================================================================================= #

    # ------------------------------------------------------#
    #   设置用到的显卡
    #   主线程的local_rank为0
    # ------------------------------------------------------#
    ngpus_per_node = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank = 0
        rank = 0

    # ----------------------------------------------------#
    #   获取classes和anchor
    # ----------------------------------------------------#
    class_names, num_classes = get_classes(classes_path)

    # ------------------------------------------------------#
    #   创建模型
    # ------------------------------------------------------#
    model = EfficientVRNet(num_classes=num_classes, num_seg_classes=num_classes_seg, phi=phi).cuda(local_rank)
    weights_init(model)
    if model_path != '':
        # ------------------------------------------------------#
        #   权值文件请看README，百度网盘下载
        # ------------------------------------------------------#
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))

        # ------------------------------------------------------#
        #   根据预训练权重的Key和模型的Key进行加载
        # ------------------------------------------------------#
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        # ------------------------------------------------------#
        #   显示没有匹配上的Key
        # ------------------------------------------------------#
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")

    # ----------------------#
    #   记录Loss
    # ----------------------#
    time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    log_dir = os.path.join(save_dir, "eval_" + str(time_str))
    log_dir_seg = os.path.join(save_dir_seg, "eval_" + str(time_str))
    loss_history = LossHistory(log_dir, model, input_shape=input_shape)
    loss_history_seg = LossHistory_seg(log_dir_seg, model, input_shape=input_shape)

    # ------------------------------------------------------------------#
    #   torch 1.2不支持amp，建议使用torch 1.7.1及以上正确使用fp16
    #   因此torch1.2这里显示"could not be resolve"
    # ------------------------------------------------------------------#

    model_train = model.train()
    # ----------------------------#
    #   多卡同步Bn
    # ----------------------------#
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:
            # ----------------------------#
            #   多卡平行运行
            # ----------------------------#
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank],
                                                                    find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.to(device)

    # ---------------------------#
    #   读取检测数据集对应的txt
    # ---------------------------#
    with open(test_annotation_path, encoding='utf-8') as f:
        val_lines = f.readlines()
    num_val = len(val_lines)


    # ----------------------#
    #   记录eval的map曲线
    # ----------------------#
    eval_callback = EvalCallback(model, input_shape, class_names, num_classes, val_lines, log_dir, Cuda, \
                                    eval_flag=eval_flag, period=eval_period, radar_path=radar_file_path,
                                    local_rank=local_rank)
    eval_callback_seg = EvalCallback_seg(model, input_shape, num_classes_seg, val_lines, se_seg_path,
                                            log_dir_seg, Cuda, eval_flag=eval_flag, period=eval_period,
                                            radar_path=radar_file_path, local_rank=local_rank, jpg_path=jpg_path)

    # ---------------------------------------#
    #   模型性能测试
    # ---------------------------------------#
    epoch = 0
    model_train_eval = model_train.eval()
    # eval_callback.on_epoch_end(epoch, model_train_eval)
    eval_callback_seg.on_epoch_end(epoch, model_train_eval)
    # eval_callback_seg_wl.on_epoch_end(epoch, model_train_eval)
    # if is_radar_pc_seg:
        # eval_callback_seg_pc.on_epoch_end(epoch, model_train_eval)
