python trainer.py --manualSeed 806 \
--cls_weight 0.01 --cls_weight_unseen 0.01 --nclass_all 81 --syn_num 300 --val_every 1 \
--cuda --netG_name MLP_G --netD_name MLP_D \
--nepoch 100 --nepoch_cls 15 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 \
--dataset coco --batch_size 32 --nz 300 --attSize 512 --resSize 1024 --gan_epoch_budget 38000 \
--lr 0.0001 --lr_step 30 --lr_cls 0.0001 \
--outname checkpoints/coco_48_17 \
--pretrain_classifier mmdetection/work_dirs/coco14/coco4817/epoch_12.pth \
--pretrain_classifier_unseen MSCOCO/unseen_Classifier_4817.pth \
--class_embedding MSCOCO/se-coco.npy \
--testsplit test_0.6_0.3 \
--trainsplit train_0.6_0.3 \
--classes_split 48_17 \
--lz_ratio 0.01 \
--dataroot mmdetection/feature/coco14/coco4817
# --netD checkpoints/coco_65_15_0.6_1_0_1/disc_latest.pth \
# --netG checkpoints/coco_65_15_0.6_1_0_1/gen_latest.pth
# split, th, ls, lu, lz
