# IMAGE COLORIZATION BY CAPSULE NETWORKS


Instructions
-------------
1) Before start training, datasets should be generated by command below:

						./generate_patch_pairs.py

   If asked, datasets (ILSVRC 2012 and DIV2K) may be provided as npz files:
   
					ILSVRC 2012 		-> train_9_9_4_4.npz					
					DIV2K (train)		-> train_9_9_4.npz
					DIV2K (validation)	-> valid_9_9_4.npz

2) Training is performed by commands below:

					./colorizer.py --batch_size 128 --complexity 6 --datapath ./train_9_9_4_4.npz --dataset ntire --epochs 10 --loss mse --optimizer adam --routings 1 --train --save --run 1
					./colorizer.py --batch_size 128 --complexity 6 --datapath ./train_9_9_4.npz --dataset ntire --epochs 10 --loss mse --optimizer adam --routings 1 --train --save --run 2 --pretrained_model runs/1/weights-10.h5
					./colorizer.py --batch_size 128 --complexity 6 --datapath ./valid_9_9_4.npz --dataset ntire --epochs 10 --loss mse --optimizer adam --routings 1 --train --save --run 3 --pretrained_model runs/2/weights-10.h5

3) Validation/testing is performed by commands below:

					./colorizer.py --complexity 6 --epochs 10 --routings 1 --save --predict --testpath Validation_gray/ --run 3
					./colorizer.py --complexity 6 --epochs 10 --routings 1 --save --predict --testpath Test_gray/ --run 3

Here, the model file runs/3/weights-10.h5 is used to predict all png files under Validation_gray or Test_gray directories.
   

Citation
--------
@InProceedings{Ozbulak_2019_CVPR_Workshops,
author = {Ozbulak, Gokhan},
title = {Image Colorization by Capsule Networks},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
month = {June},
year = {2019}
} 
