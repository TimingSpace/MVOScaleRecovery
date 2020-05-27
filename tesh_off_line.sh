#!/bin/bash
for num in 0 1 2 3 4 5 6 7 8 9
do
    python src/main_offline.py  evaluate_result/kitti_$2_base_01.npy $1$2$num  > result_log/log_program_$1_$2_$num
    echo  'evaluation vo'
    python script/evaluate_vo.py dataset/kitti_gt/$2.txt evaluate_result/kitti_$2_base_01_path.txt$1$2$num >result_log/log_vo_$1_$2_$num
    #echo  'evaluation scale'
    #python script/evaluate_scale.py dataset/kitti_gt/$2_speed.txt evaluate_result/kitti_$2_base_01_scales.txt$1$2$num>result_log/log_scale_$1_$2_$num

    python script/change_scale.py dataset/kitti_gt/$2.txt evaluate_result/kitti_$2_base_01_scales.txt$1$2$num
    echo  'evaluation gt+vo_scale'
    python script/evaluate_vo.py dataset/kitti_gt/$2.txt new_pose.txt$1$2$num >result_log/log_vo_s_$1_$2_$num
done
cat result_log/log_vo_$1_$2_[0123456789] > result_log/log_vo_$1_$2_all
cat result_log/log_vo_s_$1_$2_[0123456789] > result_log/log_vo_s_$1_$2_all
#cat result_log/log_scale_$1_$2_[0123456789] > result_log/log_scale_$1_$2_all
