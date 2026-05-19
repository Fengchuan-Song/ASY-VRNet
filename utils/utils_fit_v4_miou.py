import os
import torch
from tqdm import tqdm
from utils.utils import get_lr
from nets.deeplabv3_training import (CE_Loss, Dice_loss, Focal_Loss,
                                     weights_init)
from utils_seg.utils import get_lr
from utils_seg.utils_metrics import f_score, mIoU

from utils.multitaskloss import MultiTaskLossWrapper

import wandb


def fit_one_epoch(model_train, model, ema, yolo_loss, loss_history, loss_history_seg, eval_callback, optimizer, epoch, epoch_step,
                  epoch_step_val, gen, gen_val, Epoch, cuda, fp16, scaler, weight_save_dir, dice_loss, focal_loss, cls_weights, num_class_seg,  
                  local_rank=0):
    total_loss_det = 0
    total_loss_seg = 0
    total_f_score = 0
    total_miou = 0

    val_loss_det = 0
    val_loss_seg = 0
    val_f_score = 0
    val_miou = 0

    train_total_loss = 0
    val_total_loss = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            iteration -= 1
            break

        images, targets, radars, pngs, seg_labels = batch[0], batch[1], batch[2], batch[3], batch[4]

        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                images = images.cuda(local_rank)
                targets = [ann.cuda(local_rank) for ann in targets]
                radars = radars.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                seg_labels = seg_labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

        # ----------------------#
        #   清零梯度
        # ----------------------#
        optimizer.zero_grad()
        if not fp16:
            # ----------------------#
            #   前向传播
            # ----------------------#
            outputs, outputs_seg = model_train(images, radars)

            # ----------------------------------- 计算损失 ------------------------------------ #
            if focal_loss:
                loss_seg = Focal_Loss(outputs_seg, pngs, weights, num_classes=num_class_seg)
            else:
                loss_seg = CE_Loss(outputs_seg, pngs, weights, num_classes=num_class_seg)

            if dice_loss:
                main_dice = Dice_loss(outputs_seg, seg_labels)
                loss_seg = loss_seg + main_dice

            loss_det = yolo_loss(outputs, targets)

            # mtl = HUncertainty(task_num=3)
            # mgda = MGDA()
            mtl = MultiTaskLossWrapper(task_num=2)
            total_loss = mtl(loss_seg, loss_det)
            # -------------------------------------------------------------------------------- #

            with torch.no_grad():
                # train_f_score = f_score(outputs_seg, seg_labels)
                # train_f_score_w = f_score(outputs_seg_w, seg_w_labels)
                train_miou = mIoU(outputs_seg, seg_labels)

            # ----------------------#
            #   反向传播
            # ----------------------#
            total_loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                outputs, outputs_seg = model_train(images, radars)

                # ----------------------------------- 计算损失 ------------------------------------ #
                if focal_loss:
                    loss_seg = Focal_Loss(outputs_seg, pngs, weights, num_classes=num_class_seg)
                else:
                    loss_seg = CE_Loss(outputs_seg, pngs, weights, num_classes=num_class_seg)

                if dice_loss:
                    main_dice = Dice_loss(outputs_seg, seg_labels)
                    loss_seg = loss_seg + main_dice

                loss_det = yolo_loss(outputs, targets)

                # mtl = HUncertainty(task_num=3)
                # mgda = MGDA()
                total_loss = loss_det + 5 * loss_seg
                # -------------------------------------------------------------------------------- #

                with torch.no_grad():
                    # train_f_score = f_score(outputs_seg, seg_labels)
                    # train_f_score_w = f_score(outputs_seg_w, seg_w_labels)
                    train_miou = mIoU(outputs_seg, seg_labels)

            # ----------------------#
            #   back-propagation
            # ----------------------#
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        if ema:
            ema.update(model_train)

        total_loss_det += loss_det.item()
        total_loss_seg += loss_seg.item()
        train_total_loss = total_loss_det + total_loss_seg
        total_miou += train_miou.item()

        if local_rank == 0:
            pbar.set_postfix(**{'detection loss': total_loss_det / (iteration + 1),
                                'se seg loss': total_loss_seg / (iteration + 1),
                                'total loss': train_total_loss / (iteration + 1),
                                # 'f score se': total_f_score / (iteration + 1),
                                # 'f score wl': total_f_score_w / (iteration + 1),
                                'lr': get_lr(optimizer),
                                # 'img var': total_image_log_var / (iteration + 1),
                                # 'rad var': total_radar_log_var / (iteration + 1),
                                # 'rad var': total_radar_log_var / (iteration + 1),
            })
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    if ema:
        model_train_eval = ema.ema
    else:
        model_train_eval = model_train.eval()

    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            iteration -= 1
            break
        images, targets, radars, pngs, seg_labels = batch[0], batch[1], batch[2], batch[3], batch[4]

        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                targets = [ann.cuda(local_rank) for ann in targets]
                radars = radars.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                seg_labels = seg_labels.cuda(local_rank)
                weights = weights.cuda(local_rank)
            # ----------------------#
            #   清零梯度
            # ----------------------#
            optimizer.zero_grad()
            # ----------------------#
            #   前向传播
            # ----------------------#
            outputs, outputs_seg = model_train(images, radars)

            if focal_loss:
                loss_seg = Focal_Loss(outputs_seg, pngs, weights, num_classes=num_class_seg)
            else:
                loss_seg = CE_Loss(outputs_seg, pngs, weights, num_classes=num_class_seg)

            if dice_loss:
                main_dice = Dice_loss(outputs_seg, seg_labels)
                loss_seg = loss_seg + main_dice

            # -------------------------------#
            #   计算f_score
            # -------------------------------#
            # _f_score = f_score(outputs_seg, seg_labels)
            # _f_score_w = f_score(outputs_seg_w, seg_w_labels)
            _miou = mIoU(outputs_seg, seg_labels)

            # ----------------------#
            #   计算损失
            # ----------------------#
            loss_value = yolo_loss(outputs, targets)
            loss_value_seg = loss_seg
            val_miou += _miou.item()

        val_loss_det += loss_value.item()
        val_loss_seg += loss_value_seg.item()
        val_total_loss = val_loss_det + val_loss_seg

        if local_rank == 0:
            pbar.set_postfix(**{'detection val_loss': val_loss_det / (iteration + 1),
                                'se seg val_loss': val_loss_seg / (iteration + 1),
                                'val loss': val_total_loss / (iteration + 1),
                                # 'f_score se': val_f_score / (iteration + 1),
                                # 'f_score wl': val_f_score_w / (iteration + 1),
                                'miou se(val)': val_miou / (iteration + 1),
                                })
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        val_map50, val_map50_95 = eval_callback.on_epoch_end(epoch + 1, model_train_eval)
        loss_history.append_mAP50(val_map50)
        loss_history.append_mAP50_95(val_map50_95)
        loss_history_seg.append_miou(val_miou / epoch_step_val)
        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        print(
            'Total Loss: %.3f || Val Loss Det: %.3f || Val mAP50: %.3f || Val mAP50-95: %.3f || Val Loss Seg: %.3f' % (
            (train_total_loss / epoch_step,
                val_loss_det / epoch_step_val,
                val_map50,
                val_map50_95,
                val_loss_seg / epoch_step_val,
            )))
        
        wandb.log({
            'epoch': epoch,
            'detection loss': total_loss_det / epoch_step,
            'se seg loss': total_loss_seg / epoch_step,
            'total loss': train_total_loss / epoch_step,
            'mIoU se(train)': total_miou / epoch_step,
            'lr': get_lr(optimizer),
            'detection val_loss': val_loss_det / epoch_step_val,
            'mAP50(eval)': val_map50,
            'mAP50-95(eval)': val_map50_95,
            'se seg val_loss': val_loss_seg / epoch_step_val,
            'val loss': val_total_loss / epoch_step_val,
            'mIoU se(eval)': val_miou / epoch_step_val,            
        })
        # -----------------------------------------------#
        #   保存权值
        # -----------------------------------------------#
        if ema:
            save_state_dict = ema.ema.state_dict()
        else:
            save_state_dict = model.state_dict()

        # if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
        #     torch.save(save_state_dict, os.path.join(weight_save_dir,
        #                                              "ep%03d-loss%.3f-det_val_loss%.3f-seg_val_loss%.3f-seg_wl_val_loss%.3f.pth" % (
        #                                                  epoch + 1, val_total_loss / epoch_step_val,
        #                                                  val_loss_det / epoch_step_val,
        #                                                  val_loss_seg / epoch_step_val,
        #                                                  val_loss_seg_w / epoch_step_val)))

        flag = 0
        if len(loss_history.mAP50) <= 1 or val_map50 >= max(loss_history.mAP50):
            flag += 1
        if len(loss_history.mAP50_95) <= 1 or val_map50_95 >= max(loss_history.mAP50_95):
            flag += 1
        if len(loss_history_seg.miou) <= 1 or (val_miou / epoch_step_val) >= max(loss_history_seg.miou):
            flag += 1

        if flag >= 1:
            print('Save best model to best_epoch_weights.pth')
            torch.save(save_state_dict, os.path.join(weight_save_dir, "best_epoch_weights_ep%03d_mAP50%.3f_mAP50-95%.3f_mIoU%.3f.pth" % (
                epoch + 1, val_map50, val_map50_95, val_miou / epoch_step_val)))
            # torch.save(model_train, os.path.join(weight_save_dir, "best_epoch_weights_ep%03d_valLoss%.3f_mIoU%.3f_mIoUw%.3f.pt" % (
                # epoch + 1, val_loss_det / epoch_step_val, val_miou / epoch_step_val, val_miou_w / epoch_step_val)))

        torch.save(save_state_dict, os.path.join(weight_save_dir, "last_epoch_weights.pth"))
        # torch.save(model_train, os.path.join(weight_save_dir, "last_epoch_weights.pt"))