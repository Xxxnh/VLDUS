python trainer.py --manualSeed 806 \
--cls_weight 0.01 --cls_weight_unseen 0.01 --nclass_all 81 --syn_num 250 --val_every 1 \
--cuda --netG_name MLP_G --netD_name MLP_D \
--nepoch 100 --nepoch_cls 15 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 \
--dataset coco --batch_size 32 --nz 300 --attSize 512 --resSize 1024 --gan_epoch_budget 38000 \
--lr 0.00005 --lr_step 30 --lr_cls 0.0001 \
--pretrain_classifier mmdetection/work_dirs/coco2014/coco6515/epoch_12.pth \
--pretrain_classifier_unseen MSCOCO/unseen_Classifier_6515.pth \
--class_embedding MSCOCO/se-coco.npy \
--dataroot mmdetection/feature/coco14/coco6515 \
--testsplit test_0.6_0.3 \
--trainsplit train_0.6_0.3 \
--classes_split 65_15 \
--lz_ratio 0.01 \
--outname checkpoints/coco_65_15 \



