#python src/main.py dataset/kitti_image_$2.txt $1
echo  'evaluation vo'
python script/evaluate_vo.py dataset/kitti_gt/$2.txt result/kitti_image_$2_path.txt$1
echo  'evaluation scale'
python script/evaluate_scale.py dataset/kitti_gt/$2_speed.txt result/kitti_image_$2_scales.txt$1
python script/change_scale.py dataset/kitti_gt/$2.txt result/kitti_image_$2_scales.txt$1
echo  'evaluation gt+vo_scale'
python script/evaluate_vo.py dataset/kitti_gt/$2.txt new_pose.txttxt$1

