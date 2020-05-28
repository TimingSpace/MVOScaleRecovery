#!/bin/bash
for num in 0 1
do
    python src/main.py  dataset/kitti_image_$2.txt $1$num > result/log_program_$1_$2_$num &
done
wait
#!/bin/bash
for num in 0 1
do
    echo  'evaluation vo' $1 $2 $num
    python script/evaluate_vo.py dataset/kitti_gt/$2.txt result/kitti_image_$2_path.txt$1$num > result/log_vo_$1_$2_$num
    cat result/log_vo_$1_$2_$num
done


