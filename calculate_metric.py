import os
import numpy as np
import json
import math
from pathlib import Path


def calculate_metric(res_fp, rmse_thresh):
    with open(res_fp) as f:
        res = json.load(f)
    
    ls_rmse = []
    tp, tn, fp, fn = 0, 0, 0, 0
    total = 0
    num_pos, num_neg = 0, 0
    for img_fp, img_result in res['img_dict'].items():
        is_true, pred_pos, true_pos = img_result['result'], img_result['pred'], img_result['true']
        have_ball = all(el > 0 for el in true_pos)

        rmse = np.sqrt((pred_pos[0] - true_pos[0])**2 + (pred_pos[1] - true_pos[1])**2)

        if have_ball:
            num_pos += 1
            if rmse < rmse_thresh:
                tp += 1
            else:
                if all(el==0 for el in pred_pos):   # neu co bong nhung ko detect duoc bong
                    fn += 1
                else:
                    fp += 1     # neu co bong nhung detect sai
        else:
            num_neg += 1
            if all(el==0 for el in pred_pos):
                tn += 1
            else:
                fp += 1

        if have_ball and not all(el==0 for el in pred_pos):
            ls_rmse.append(rmse)
        
        total += 1

    
    print(f'total: {total}')
    print(f'true_positive: {tp}')
    print(f'true_negative: {tn}')
    print(f'false_positive: {fp}')
    print(f'false_negative: {fn}')
    print('mean rmse: ', round(sum(ls_rmse)/len(ls_rmse), 3))

    res['total'] = total
    res['num_pos'] = num_pos
    res['num_neg'] = num_neg
    res['true_positive'] = tp
    res['true_negative'] = tn
    res['false_positive'] = fp
    res['false_negative'] = fn
    res['precision'] = tp / (tp + fp)
    res['recall'] = tp / (tp + fn)
    res['accuracy'] = (tp + tn) / (tp + tn + fp + fn)
    res['mean_rmse'] = round(sum(ls_rmse)/len(ls_rmse), 3)

    res_fp = Path(res_fp)
    with open(res_fp.parent / (res_fp.stem + '_add_precision_recall.json'), 'w') as f:
        json.dump(res, f)


def calculate_metric_yolo(res_fp, rmse_thresh):
    with open(res_fp) as f:
        res = json.load(f)
    
    ls_rmse = []
    tp, tn, fp, fn = 0, 0, 0, 0
    total = 0
    for img_fp, img_result in res.items():
        if '/' not in img_fp:
            continue
        input_size, pred_pos, true_pos = img_result['input_size'], img_result['pred_pos'], img_result['true_pos']
        have_ball = all(el > 0 for el in true_pos)

        rmse = np.sqrt((pred_pos[0] - true_pos[0])**2 + (pred_pos[1] - true_pos[1])**2)

        if have_ball:
            if rmse < rmse_thresh:
                tp += 1
            else:
                if all(el==0 for el in pred_pos):   # neu co bong nhung ko detect duoc bong
                    fn += 1
                else:
                    fp += 1     # neu co bong nhung detect sai
        else:
            if all(el==0 for el in pred_pos):
                tn += 1
            else:
                fp += 1

        if have_ball and not all(el==0 for el in pred_pos):
            ls_rmse.append(rmse)
        
        total += 1

    
    print(f'total: {total}')
    print(f'true_positive: {tp}')
    print(f'true_negative: {tn}')
    print(f'false_positive: {fp}')
    print(f'false_negative: {fn}')
    print('mean rmse: ', round(sum(ls_rmse)/len(ls_rmse), 3))

    res['total'] = total
    res['true_positive'] = tp
    res['true_negative'] = tn
    res['false_positive'] = fp
    res['false_negative'] = fn
    res['precision'] = tp / (tp + fp)
    res['recall'] = tp / (tp + fn)
    res['accuracy'] = (tp + tn) / (tp + tn + fp + fn)
    res['mean_rmse'] = round(sum(ls_rmse)/len(ls_rmse), 3)

    res_fp = Path(res_fp)
    with open(res_fp.parent / (res_fp.stem + '_add_precision_recall.json'), 'w') as f:
        json.dump(res, f)


if __name__ == '__main__':
    # calculate_metric_yolo(
    #     res_fp='/data2/tungtx2/datn/yolov8/results/train8_test_dict_1_add_no_ball_frames.json',
    #     # res_fp='/data2/tungtx2/datn/yolov8/results/train17_test_dict_1_add_no_ball_frames.json',
    #     rmse_thresh=3,
    # )

    calculate_metric(
        res_fp='/data2/tungtx2/datn/ball_detect/results/exp85_centernet_no_asl_3_frames_add_pos_pred_weight_add_no_ball_frame/test_add_no_ball_frames/result.json',
        rmse_thresh=3,
    )
