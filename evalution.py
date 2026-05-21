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
# from nets.Achelous import *
# from nets.Achelous_gate import *
# from nets.Achelous_uncertainty_v4 import *
from nets.efficient_vrnet import EfficientVRNet
from nets.yolo_training import (ModelEMA, YOLOLoss, get_lr_scheduler,
                                set_optimizer_lr, weights_init)
from utils.callbacks import LossHistory, EvalCallback
from utils_seg.callbacks import EvalCallback as EvalCallback_seg
from utils_seg_line.callbacks import EvalCallback as EvalCallback_seg_line
from utils.dataloader import YoloDataset, yolo_dataset_collate, yolo_dataset_collate_all
from utils.utils import get_classes, show_config
from utils.utils_fit import fit_one_epoch
from utils_seg.callbacks_v2 import LossHistory as LossHistory_seg
from utils_seg_line.callbacks import LossHistory as LossHistory_seg_line
from utils_seg_pc.callbacks import LossHistory as LossHistory_seg_pc
from utils_seg_pc.callbacks import EvalCallback as EvalCallback_seg_pc
import argparse
import wandb


if __name__ == "__main__":
    # =========== 参数解析实例 =========== #
    parser = argparse.ArgumentParser()

    # 添加参数解析
    parser.add_argument("--cuda", type=str, default="True")
    parser.add_argument("--ddp", type=str, default="False")
    parser.add_argument("--is_pc", help="use pc seg", type=str, default="False")
    # parser.add_argument("--model_path", type=str, default='/data/Achelous_v3.2/weights/best_epoch_weights_ep271_valLoss0.708_mIoU0.818_mIoUw0.660.pth')
    # parser.add_argument("--model_path", type=str, default='/data/Achelous_v3.2/weights/best_epoch_weights_ep224_valLoss0.714_mIoU0.813_mIoUw0.660.pth')
    # parser.add_argument("--model_path", type=str, default='/data/Achelous_v3.2/weights/best_epoch_weights_ep221_valLoss0.719_mIoU0.812_mIoUw0.659.pth')
    # parser.add_argument("--model_path", type=str, default='/data/Achelous_v3.3/weights/best_epoch_weights_ep219_valLoss0.712_mIoU0.812_mIoUw0.658.pth')
    # parser.add_argument("--model_path", type=str, default='/data/Achelous_v3.4/Achelous_v3.4-MIPC-2/weights/best_epoch_weights_ep081_mAP500.863_mAP50-950.598_mIoU0.804_mIoUw0.671.pth')
    parser.add_argument("--model_path", type=str, default='/data/Achelous_v3.4/Achelous_v3.4-MIPC-2/weights/best_epoch_weights_ep091_mAP500.864_mAP50-950.600_mIoU0.803_mIoUw0.672.pth')
    parser.add_argument("--backbone", type=str, default='mv')
    parser.add_argument("--neck", type=str, default='gdf')
    parser.add_argument("--nd", type=str, default="True")
    parser.add_argument("--phi", type=str, default='S0')
    parser.add_argument("--resolution", type=int, default=320)
    parser.add_argument("--pc_num", type=int, default=512)
    parser.add_argument("--rader_channels", type=int, default=4)
    parser.add_argument("--pc_model", type=str, default='pn')
    parser.add_argument("--spp", type=str, default='True')
    parser.add_argument("--data_root", type=str, default='/data_ssd/datasets/WaterScenes')
    parser.add_argument("--save_dir", type=str, default='/data/Achelous_v3.4')
    parser.add_argument('--wandb_name', type=str, default='Achelous_v3.4-MIPC-2',
                        help='name of current training procedure of wandb')

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
    classes_path = 'model_data/waterscenes_benchmark_ship_only.txt'
    model_path = args.model_path

    # ------------------------------------------------------#
    #   backbone (4 options): ef (EfficientFormer), en (EdgeNeXt), ev (EdgeViT), mv (MobileViT), rv (RepViT), pf (PoolFormer), mo (MobileOne), fv (FastViT)
    # ------------------------------------------------------#
    backbone = args.backbone

    # ------------------------------------------------------#
    #   neck (2 options): gdf (Ghost-Dual-FPN), cdf (CSP-Dual-FPN)
    # ------------------------------------------------------#
    neck = args.neck

    # ------------------------------------------------------#
    #   spp: True->SPP, False->SPPF
    # ------------------------------------------------------#
    spp = True if args.spp == 'True' else False

    # ------------------------------------------------------#
    #   detection head (2 options): normal -> False, lightweight -> True
    # ------------------------------------------------------#
    lightweight = True if args.nd == 'True' else False

    # ------------------------------------------------------#
    #   input_shape     all models support 320*320, all models except mobilevit support 416*416
    # ------------------------------------------------------#
    input_shape = [args.resolution, args.resolution]
    # ------------------------------------------------------#
    #   The size of model, three options: S0, S1, S2
    # ------------------------------------------------------#
    phi = args.phi
    # ------------------------------------------------------------------#
    #   save_period     多少个epoch保存一次权值
    # ------------------------------------------------------------------#
    save_period = 5
    # ------------------------------------------------------------------#
    #   save_dir        权值与日志文件保存的文件夹
    # ------------------------------------------------------------------#
    # save_dir = os.path.join(args.save_dir, 'weights_origin')
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
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
    eval_period = 10

    # ========================================  Dataset Path =========================================== #
    # ----------------------------------------------------#
    # 雷达feature map路径
    # ----------------------------------------------------#
    radar_file_path = args.data_root + "/radar/VOCradar320_v2"

    # ----------------------------------------------------#
    #   获得目标检测图片路径和标签
    # ----------------------------------------------------#
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

    # ------------------------------------------------------------------#
    # 水岸线分割数据集路径
    # ------------------------------------------------------------------#
    wl_seg_path = args.data_root + "/waterline/SegmentationClass"

    # ------------------------------------------------------------------#
    # 是否需要训练毫米波雷达点云分割
    # ------------------------------------------------------------------#
    is_radar_pc_seg = True if args.is_pc == 'True' else False

    pc_seg_model = args.pc_model
    # ------------------------------------------------------------------#
    # 每个batch的点云数量
    # ------------------------------------------------------------------#
    radar_pc_num = args.pc_num

    # ------------------------------------------------------------------#
    # 毫米波雷达点云分割路径
    # ------------------------------------------------------------------#
    radar_pc_seg_path = args.data_root + "radar/radar_0220/radar"

    # ------------------------------------------------------------------#
    # 毫米波雷达点云分割属性, 其中label表示雷达目标的语义标签
    # ------------------------------------------------------------------#
    radar_pc_seg_features = ['x', 'y', 'z', 'comp_velocity', 'rcs']
    radar_pc_seg_label = ['label']

    radar_pc_classes = 8
    radar_pc_channels = len(radar_pc_seg_features)
    # ================================================================================================== #

    # ============================ segmentation hyperparameters ============================= #
    # -----------------------------------------------------#
    #   num_classes     训练自己的数据集必须要修改的
    #                   自己需要的分类个数+1，如2+1
    # -----------------------------------------------------#
    num_classes_seg = 9

    # ------------------------------------------------------------------#
    #   是否给不同种类赋予不同的损失权值，默认是平衡的。
    #   设置的话，注意设置成numpy形式的，长度和num_classes一样。
    #   如：
    #   num_classes = 3
    #   cls_weights = np.array([1, 2, 3], np.float32)
    # ------------------------------------------------------------------#
    cls_weights = np.ones([num_classes_seg], np.float32)
    cls_weights_wl = np.ones([2], np.float32)

    # ------------------------------------------------------------------#
    #   save_dir_seg        分割权值与日志文件保存的文件夹
    # ------------------------------------------------------------------#
    save_dir = os.path.join(os.path.join(args.save_dir, args.wandb_name), 'logs_detection')
    save_dir_seg = os.path.join(os.path.join(args.save_dir, args.wandb_name), 'logs_seg')
    save_dir_seg_wl = os.path.join(os.path.join(args.save_dir, args.wandb_name), 'logs_seg_line')
    save_dir_seg_pc = os.path.join(os.path.join(args.save_dir, args.wandb_name), 'logs_seg_pc')

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
    if is_radar_pc_seg:
        model = Achelous(resolution=input_shape[0], num_det=num_classes, num_seg=num_classes_seg, phi=phi,
                         backbone=backbone, neck=neck, nano_head=lightweight, pc_seg=pc_seg_model,
                         pc_channels=radar_pc_channels, pc_classes=radar_pc_classes, spp=spp).cuda(local_rank)
    else:
        model = Achelous3T(resolution=input_shape[0], num_det=num_classes, num_seg=num_classes_seg, phi=phi,
                           backbone=backbone, neck=neck, spp=spp,
                           nano_head=lightweight, radar_channels=args.rader_channels).cuda(local_rank)
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
    log_dir_seg_wl = os.path.join(save_dir_seg_wl, "eval_" + str(time_str))
    log_dir_seg_pc = os.path.join(save_dir_seg_pc, "eval_" + str(time_str))

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(log_dir_seg):
        os.makedirs(log_dir_seg)
    if not os.path.exists(log_dir_seg_wl):
        os.makedirs(log_dir_seg_wl)
    if not os.path.exists(log_dir_seg_pc):
        os.makedirs(log_dir_seg_pc)

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
                                    radar_pc_seg_path=radar_pc_seg_path, local_rank=local_rank, is_radar_pc_seg=is_radar_pc_seg,
                                    radar_pc_seg_features=radar_pc_seg_features, radar_pc_seg_label=radar_pc_seg_label,
                                    radar_pc_num=radar_pc_num, train_name=args.wandb_name)
    eval_callback_seg = EvalCallback_seg(model, input_shape, num_classes_seg, val_lines, se_seg_path,
                                            log_dir_seg, Cuda, eval_flag=eval_flag, period=eval_period,
                                            radar_path=radar_file_path, radar_pc_seg_path=radar_pc_seg_path,
                                            local_rank=local_rank, jpg_path=jpg_path, is_radar_pc_seg=is_radar_pc_seg,
                                            radar_pc_seg_features=radar_pc_seg_features,
                                            radar_pc_seg_label=radar_pc_seg_label, radar_pc_num=radar_pc_num, train_name=args.wandb_name)
    eval_callback_seg_wl = EvalCallback_seg_line(model, input_shape, 2, val_lines, wl_seg_path,
                                            log_dir_seg_wl, Cuda, eval_flag=eval_flag, period=eval_period,
                                            radar_path=radar_file_path, local_rank=local_rank,
                                            radar_pc_seg_path=radar_pc_seg_path, jpg_path=jpg_path, is_radar_pc_seg=is_radar_pc_seg,
                                                    radar_pc_seg_features=radar_pc_seg_features,
                                                    radar_pc_seg_label=radar_pc_seg_label, radar_pc_num=radar_pc_num, train_name=args.wandb_name)
    eval_callback_seg_pc = EvalCallback_seg_pc(model, input_shape, 2, val_lines, wl_seg_path,
                                                    log_dir_seg_wl, Cuda, eval_flag=eval_flag, period=eval_period,
                                                    radar_path=radar_file_path, local_rank=local_rank,
                                                    radar_pc_seg_path=radar_pc_seg_path, jpg_path=jpg_path,
                                                    is_radar_pc_seg=is_radar_pc_seg,
                                                    radar_pc_seg_features=radar_pc_seg_features,
                                                    radar_pc_seg_label=radar_pc_seg_label, radar_pc_num=radar_pc_num)

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
