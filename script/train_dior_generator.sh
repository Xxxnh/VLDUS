python trainer.py --manualSeed 806 \
--cls_weight 0.1 --cls_weight_unseen 0.1 --nclass_all 21 --syn_num 500 --val_every 1 \
--cuda --netG_name MLP_G \
--netD_name MLP_D --nepoch 150 --nepoch_cls 25 --ngh 4096 --ndh 4096 --lambda1 10 \
--critic_iter 5 \
--dataset voc --batch_size 64 --nz 512 --attSize 512 --resSize 1024 \
--lr 0.00001 \
--lr_step 20 --gan_epoch_budget 38000 --lr_cls 0.00005 \
--pretrain_classifier mmdetection/work_dirs/dior1604/epoch_14.pth \
--class_embedding DIOR/se-dior.npy \
--testsplit test_0.6_0.3 \
--trainsplit train_0.6_0.3 \
--lz_ratio 0.001 \
--outname checkpoints/DIOR \
--pretrain_classifier_unseen DIOR/unseen_Classifier.pth \
--dataroot mmdetection/feature/dior1604