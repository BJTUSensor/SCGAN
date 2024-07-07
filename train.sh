# noise modeling
#python train.py  \
#       --mode NoiseModeling \
#       --isTrain True \
#       --name all2 \
#       --dir_data data/5-all/train/  \
#       --dataset_noisy noisyC_patch_50 \
#       --dataset_clean cleanC_patch_50 \
#       --dataset TwoFolders \
#       --patch_size 50 \
#       --batch_size 20 \
#       --epochs 60 \
#       --lr 1e-4 \
#       --lr_decay_epochs 15 \
#
#python train.py \
#       --mode NoiseModeling \
#       --pre_train ./checkpoints/all2 \
#       --which_epoch 60 \
#       --dir_noisy data/5-all/train/noisyC_patch_50 \
#       --dir_clean data/5-all/train/cleanC_patch_50 \


#
#### denoising
python train.py  \
       --name all1-dncnn \
       --mode Denoising \
       --isTrain True \
       --dir_data data/5-all/train/  \
       --dataset_noisy noisyC_patch_50 \
       --dataset_clean cleanC_patch_50 \
       --dataset PairedFolders \
       --patch_size 50 \
       --batch_size 20 \
       --epochs 45 \
       --lr_decay_epochs 13 \
       --lr 0.00005 \

#       --pre_train ./checkpoints/all4-dncnn \
#       --which_epoch 18  \

python train.py \
      --mode Denoising \
      --dn_methods SCGAN \
      --pre_train ./checkpoints/all1-dncnn --which_epoch 45 \
      --dir_noisy ./data/4-real/test/noisyA \
#     --dir_clean ./data/5-all/train/test_clean
