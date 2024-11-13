from __future__ import print_function
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
from torch.autograd import Variable
import math
from util import *
import model
import clip
from cls_models import ClsModel, ClsUnseen
from PIL import Image
import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity

# device = "cuda" if torch.cuda.is_available() else "cpu"
# # print(clip.available_models())
# clipmodel, preprocess = clip.load("ViT-B/32", device=device)

class TrainGAN():
    def __init__(self, opt, attributes, clip_attributes, unseenAtt, unseenLabels, seen_feats_mean, gen_type='FG'):
        
        '''
        CLSWGAN trainer
        Inputs:
            opt -- arguments
            unseenAtt -- embeddings vector of unseen classes
            unseenLabels -- labels of unseen classes
            attributes -- embeddings vector of all classes
        '''
        self.opt = opt

        self.gen_type = gen_type
        self.Wu_Labels = np.array([i for i, l in enumerate(unseenLabels)])
        print(f"Wu_Labels {self.Wu_Labels}")
        self.Wu = unseenAtt

        self.unseen_classifier = ClsUnseen(unseenAtt)
        self.unseen_classifier.cuda()
        self.unseen_classifier = loadUnseenWeights(opt.pretrain_classifier_unseen, self.unseen_classifier)
        
        self.classifier = ClsModel(num_classes=opt.nclass_all)
        self.classifier.cuda()
        self.classifier = loadFasterRcnnCLSHead(opt.pretrain_classifier, self.classifier)
        
        for p in self.classifier.parameters():
            p.requires_grad = False
        
        for p in self.unseen_classifier.parameters():
            p.requires_grad = False

        self.ntrain = opt.gan_epoch_budget
        self.attributes = attributes.data.numpy()
        self.clip_attributes = clip_attributes.data.numpy()

        print(f"# of training samples: {self.ntrain}")
        # initialize generator and discriminator
        self.netG = model.MLP_G(self.opt)
        self.netD = model.MLP_CRITIC(self.opt)
        


        if self.opt.cuda and torch.cuda.is_available():
            self.netG = self.netG.cuda()
            self.netD = self.netD.cuda()

        print('\n\n#############################################################\n')
        print(self.netG, '\n')
        print(self.netD)
        print('\n#############################################################\n\n')

        # classification loss, Equation (4) of the paper
        self.cls_criterion = nn.NLLLoss()
        self.clip_loss_f1 = nn.L1Loss()

        self.one = torch.tensor(1, dtype=torch.float)
        self.mone = self.one * -1
        # self.one = 1
        # self.mone = -1

        if self.opt.cuda:
            
            self.one = self.one.cuda()
            self.mone = self.mone.cuda()
            self.cls_criterion.cuda()
            self.clip_loss_f1.cuda()


        self.optimizerD = optim.Adam(self.netD.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))

    def __call__(self, epoch, features, labels, clipmodel, prep):
        """
        Train GAN for one epoch
        Inputs:
            epoch: current epoch
            features: current epoch subset of features
            labels: ground truth labels
        """
        self.epoch = epoch
        self.features = features
        self.labels = labels
        self.ntrain = len(self.labels)
        self.clipmodel = clipmodel
        self.clipreprocess = prep
        self.trainEpoch()
    
    def load_checkpoint(self):
        checkpoint = torch.load(self.opt.netG)
        self.netG.load_state_dict(checkpoint['state_dict'])
        epoch = checkpoint['epoch']
        self.netD.load_state_dict(torch.load(self.opt.netD)['state_dict'])
        print(f"loaded weights from epoch: {epoch} \n{self.opt.netD} \n{self.opt.netG} \n")
        return epoch
    def save_checkpoint(self, state='latest'):
        torch.save({'state_dict': self.netD.state_dict(), 'epoch': self.epoch}, f'{self.opt.outname}/disc_{state}.pth')
        torch.save({'state_dict': self.netG.state_dict(), 'epoch': self.epoch}, f'{self.opt.outname}/gen_{state}.pth')

    def generate_syn_feature(self, labels, attribute, num=100, no_grad=True):
        """
        generates features
        inputs:
            labels: features labels to generate nx1 n is number of objects 
            attributes: attributes of objects to generate (nxd) d is attribute dimensions
            num: number of features to generate for each object
        returns:
            1) synthesised features 
            2) labels of synthesised  features 
        """

        nclass = labels.shape[0]
        syn_feature = torch.FloatTensor(nclass * num , self.opt.resSize)
        syn_label = torch.LongTensor(nclass*num)

        syn_att = torch.FloatTensor(num, self.opt.attSize)
        syn_noise = torch.FloatTensor(num, self.opt.nz)
        
        if self.opt.cuda:
            syn_att = syn_att.cuda()
            syn_noise = syn_noise.cuda()
        if no_grad is True:
            with torch.no_grad():
                for i in range(nclass):
                    label = labels[i]
                    iclass_att = attribute[i]
                    syn_att.copy_(iclass_att.repeat(num, 1))
                    syn_noise.normal_(0, 1)
                    # print(syn_att.shape,syn_noise.shape)
                    output = self.netG(Variable(syn_noise), Variable(syn_att))
                    
                    syn_feature.narrow(0, i*num, num).copy_(output.data.cpu())
                    syn_label.narrow(0, i*num, num).fill_(label)
                    # print(output.shape,syn_feature.shape,syn_label.shape)
        else:
            for i in range(nclass):
                label = labels[i]
                iclass_att = attribute[i]
                syn_att.copy_(iclass_att.repeat(num, 1))
                syn_noise.normal_(0, 1)
                # print(syn_att.shape,syn_noise.shape)
                output = self.netG(Variable(syn_noise), Variable(syn_att))
            
                syn_feature.narrow(0, i*num, num).copy_(output.data.cpu())
                syn_label.narrow(0, i*num, num).fill_(label)
                # print(output.shape,syn_feature.shape,syn_label.shape)

        return syn_feature, syn_label

    def sample(self):
        """
        randomaly samples one batch of data
        returns (1)real features, (2)labels (3) attributes embeddings
        """
        idx = torch.randperm(self.ntrain)[0:self.opt.batch_size]
        batch_feature = torch.from_numpy(self.features[idx])
        batch_label = torch.from_numpy(self.labels[idx])
        batch_att = torch.from_numpy(self.attributes[batch_label])
        # print(self.features.shape)
        if 'BG' == self.gen_type:
            batch_label*=0
        return batch_feature, batch_label, batch_att

    def calc_gradient_penalty(self, real_data, fake_data, input_att):
        alpha = torch.rand(self.opt.batch_size, 1)
        alpha = alpha.expand(real_data.size())
        if self.opt.cuda:
            alpha = alpha.cuda()

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        if self.opt.cuda:
            interpolates = interpolates.cuda()

        interpolates = Variable(interpolates, requires_grad=True)

        _, disc_interpolates = self.netD(interpolates, Variable(input_att))

        ones = torch.ones(disc_interpolates.size())
        if self.opt.cuda:
            ones = ones.cuda()

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                grad_outputs=ones,
                                create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.opt.lambda1
        return gradient_penalty

    def get_z_random(self):
        """
        returns normal initialized noise tensor 
        """
        z = torch.cuda.FloatTensor(self.opt.batch_size, self.opt.nz)
        z.normal_(0, 1)
        return z

    def trainEpoch(self):

        for i in range(0, self.ntrain, self.opt.batch_size):
            # import pdb; pdb.set_trace()
            input_res, input_label, input_att = self.sample()

            # all_clip_att = torch.from_numpy(self.attributes)
            all_clip_att = torch.from_numpy(self.clip_attributes)
            all_clip_att = all_clip_att.type(torch.FloatTensor).cuda()
            clip_att = Variable(all_clip_att)

            if self.opt.batch_size != input_res.shape[0]:
                continue
            input_res, input_label, input_att = input_res.type(torch.FloatTensor).cuda(), input_label.type(torch.LongTensor).cuda(), input_att.type(torch.FloatTensor).cuda()
            # print(input_res.shape, input_label.shape, input_att.shape)
            ############################
            # (1) Update D network: optimize WGAN-GP objective, Equation (2)
            ###########################
            for p in self.netD.parameters(): # reset requires_grad
                p.requires_grad = True # they are set to False below in netG update

            for iter_d in range(self.opt.critic_iter):
                self.netD.zero_grad()
                # train with realG
                # sample a mini-batch
                # sparse_real = self.opt.resSize - input_res[1].gt(0).sum()
                input_resv = Variable(input_res)
                input_attv = Variable(input_att)
                
                _, criticD_real = self.netD(input_resv, input_attv)
                criticD_real = criticD_real.mean()
                criticD_real.backward(self.mone)

                # train with fakeG
                # noise.normal_(0, 1)
                noise = self.get_z_random()

                noisev = Variable(noise)
                # print(noise.shape)
                fake = self.netG(noisev, input_attv)
                # print(fake.shape)
                _, criticD_fake = self.netD(fake.detach(), input_attv)
                # print(criticD_fake.shape)
                criticD_fake = criticD_fake.mean()
                # print(criticD_fake)
                criticD_fake.backward(self.one)

                # gradient penalty
                gradient_penalty = self.calc_gradient_penalty(input_res, fake.data, input_att)
                gradient_penalty.backward()

                # loss：CLIP<->GAN (1×512）
                input_resc = Variable(input_res)
                input_attc = Variable(input_att)
                real_feat_all, _ = self.netD(input_resc, input_attc)
                noisec = self.get_z_random()
                noisevc = Variable(noisec)
                fakec = self.netG(noisevc, input_attc)
                fake_feat_all, _ = self.netD(fakec.detach(), input_attc)
                selected_rows = np.random.choice(range(0, 64), size=1, replace=False)
                real_feat = real_feat_all[selected_rows, :]
                fake_feat = fake_feat_all[selected_rows, :]
                input_att_pair = input_attc[selected_rows, :]
                concat_real = torch.empty(0, 512).to("cuda")
                concat_fake = torch.empty(0, 512).to("cuda")
                for row in selected_rows:
                   
                    clip_input = input_resc[row].unsqueeze(0)
                    clip_fake = fakec[row].unsqueeze(0)

                    image_shape = (32, 32)
                    clip_input = clip_input.view(image_shape)
                    clip_fake = clip_fake.view(image_shape)
                    
                    reali = (clip_input - clip_input.min()) / (clip_input.max() - clip_input.min())
                    reali = reali * 255
                    reali = reali.cpu()
                    reali = reali.detach().numpy()
                    reali = Image.fromarray(reali.astype('uint8'))
                    reali = self.clipreprocess(reali).unsqueeze(0).to("cuda")
                    reali = self.clipmodel.encode_image(reali).to("cuda")
                    concat_real = torch.cat((concat_real, reali), dim=0)              
                    
                    fakei = (clip_fake - clip_fake.min()) / (clip_fake.max() - clip_fake.min())
                    fakei = fakei * 255
                    fakei = fakei.cpu()
                    fakei = fakei.detach().numpy()
                    fakei = Image.fromarray(fakei.astype('uint8'))
                    fakei = self.clipreprocess(fakei).unsqueeze(0).to("cuda")
                    fakei = self.clipmodel.encode_image(fakei).to("cuda")
                    concat_fake = torch.cat((concat_fake, fakei), dim=0)

                concat_real = concat_real.type(torch.FloatTensor).cuda()
                concat_fake = concat_fake.type(torch.FloatTensor).cuda()

                # pairwise lose
                rowwise_clip = torch.mm(concat_fake, clip_att.t())
                rowwise_fake = torch.mm(fake_feat, clip_att.t())

                norms_clip = torch.norm(rowwise_clip, dim=1)
                norms_fake = torch.norm(rowwise_fake, dim=1)

                result_clip = rowwise_clip / norms_clip.unsqueeze(1)
                result_fake = rowwise_fake / norms_fake.unsqueeze(1)

                result_clip_pair = result_clip.detach().cpu().numpy()
                similarities_clip = cosine_similarity(result_clip_pair)
                similarities_clip = torch.tensor(similarities_clip, requires_grad=True)

                sum_value = (torch.sum(similarities_clip) - torch.trace(similarities_clip))/2
                # sum_value.backward()

                loss_product = self.clip_loss_f1(result_clip, result_fake)
                # loss_product.backward()
                loss_pair = sum_value*0.001 + loss_product*0.01


                loss_rr = self.clip_loss_f1(real_feat, concat_real)
                loss_ff = self.clip_loss_f1(fake_feat,concat_fake)
                loss_rf = self.clip_loss_f1(real_feat, concat_fake)
                loss_fr = self.clip_loss_f1(fake_feat,concat_real)

                # random_number = random.uniform(0, 1)
                # if random_number < 0.8:
                #     loss_clip = loss_rr + loss_ff + loss_rf + loss_fr
                # else:
                #     loss_clip = loss_rr + loss_ff
                loss_clip = loss_rr + loss_ff + loss_rf + loss_fr
                loss_clip = loss_clip * 0.005
                loss_all = loss_pair + loss_clip
                # loss_all.backward()
                # loss_clip.backward()
                
                Wasserstein_D = criticD_real - criticD_fake
                D_cost = criticD_fake - criticD_real + gradient_penalty
                # loss_loss = loss_clip + D_cost
                # loss_loss.backward()

                self.optimizerD.step()


            ############################
            # (2) Update G network: optimize WGAN-GP objective, Equation (2)
            ###########################
            for p in self.netD.parameters(): # reset requires_grad
                p.requires_grad = False # avoid computation

            self.netG.zero_grad()
            input_attv = Variable(input_att)
            noise = self.get_z_random()
            noisev = Variable(noise)
            fake = self.netG(noisev, input_attv)
            _, criticG_fake = self.netD(fake, input_attv)
            criticG_fake = criticG_fake.mean()
            G_cost = criticG_fake

            # ---------------------
            # mode seeking loss https://github.com/HelenMao/MSGAN/blob/e386c252f059703fcf5d439258949ae03dc46519/DCGAN-Mode-Seeking/model.py#L66
            noise2 = self.get_z_random()

            noise2v = Variable(noise2)
            fake2 = self.netG(noise2v, input_attv)

            lz = torch.mean(torch.abs(fake2 - fake)) / torch.mean(
                torch.abs(noise2v - noisev))
            eps = 1 * 1e-5
            loss_lz = 1 / (lz + eps)
            loss_lz*=self.opt.lz_ratio
            # ---------------------

            # classification loss
            # seen
            c_errG = self.cls_criterion(self.classifier(feats=fake, classifier_only=True), Variable(input_label))
            c_errG = self.opt.cls_weight*c_errG
            # --------------------------------------------

            # unseen 
            fake_unseen_f, fake_unseen_l = self.generate_syn_feature(self.Wu_Labels, self.Wu, num=self.opt.batch_size//4, no_grad=False)
                
            fake_pred = self.unseen_classifier(feats=fake_unseen_f.cuda(), classifier_only=True)

            unseenc_errG = self.opt.cls_weight_unseen * self.cls_criterion(fake_pred, Variable(fake_unseen_l.cuda()))

            
            

            # ---------------------------------------------
            # Total loss 

            errG = -G_cost + c_errG + loss_lz + unseenc_errG
            errG.backward()
            self.optimizerG.step()

            print(f"{self.gen_type} [{self.epoch+1:02}/{self.opt.nepoch:02}] [{i:06}/{int(self.ntrain)}] \
            Loss: {errG.item() :0.4f} D loss: {D_cost.data.item():.4f} G loss: {G_cost.data.item():.4f}, W dist: {Wasserstein_D.data.item():.4f} \
            seen loss: {c_errG.data.item():.4f}  unseen loss: {unseenc_errG.data.item():.4f} loss div: {loss_lz.item():0.4f} \
            loss distill: {loss_clip.data.item():.4f} loss_pair: {loss_pair.data.item():.4f}")
            # print(f"{self.gen_type} [{self.epoch+1:02}/{self.opt.nepoch:02}] [{i:06}/{int(self.ntrain)}] \
            # Loss: {errG.item() :0.4f} D loss: {D_cost.data.item():.4f} G loss: {G_cost.data.item():.4f}, W dist: {Wasserstein_D.data.item():.4f} \
            # seen loss: {c_errG.data.item():.4f}  unseen loss: {unseenc_errG.data.item():.4f} loss div: {loss_lz.item():0.4f}")
            
        self.netG.eval()
