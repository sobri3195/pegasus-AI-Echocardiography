from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from discriminator_heart import _netD, _netG
import numpy as np 
import helper
import skimage.io as skio
import skimage.transform

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw | fake')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=40, help='input batch size')
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=20, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0003, help='learning rate, default=0.0003')
parser.add_argument('--beta1', type=float, default=0.7, help='beta1 for adam. default=0.7')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--split', type=float, default=0.1, help='what percentage of data to be considered unlabelled' )
parser.add_argument('--few', type=int, default = 0, help='determine whether to use a split ratio (set 0) or to specify number of labelled samples per class')
parser.add_argument('--numlabelled', type=int, default = 0, help='number of labelled samples per class to be used')
parser.add_argument('--resume', type=int, default = 0, help='Determines whether we are continuing training from a checkpoint (set 1) or staring from scratch (set 0)')
parser.add_argument('--experiment', type=float, default = 0, help='extra flag to make modifications for runnning special experiments')



args = parser.parse_args()
print(args)

try:
    os.makedirs(args.outf)
except OSError:
    pass

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
print("Random Seed: ", args.manualSeed)
rng = np.random.RandomState(args.manualSeed)
torch.manual_seed(args.manualSeed)
if args.cuda:
    torch.cuda.manual_seed_all(args.manualSeed)
    # device = 1

cudnn.benchmark = True

if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if args.dataset in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset = dset.ImageFolder(root=args.dataroot,
                               transform=transforms.Compose([
                                   transforms.Scale(args.imageSize),
                                   transforms.CenterCrop(args.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
elif args.dataset == 'lsun':
    dataset = dset.LSUN(db_path=args.dataroot, classes=['bedroom_train'],
                        transform=transforms.Compose([
                            transforms.Scale(args.imageSize),
                            transforms.CenterCrop(args.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
elif args.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=args.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Scale(args.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    testset = dset.CIFAR10(root=args.dataroot, train=False, download=True,
                           transform=transforms.Compose([
                               transforms.Scale(args.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
elif args.dataset == 'ecg':
    trainx, trainy = helper.read_dataset(args.dataroot + "/FILE.hdf5")
    testx, testy = helper.read_dataset(args.dataroot + "/FILE.hdf5")
    



    testx = torch.from_numpy(testx[:,10:,30:-20,:])
    # train = []
    # trainx = np.array(trainx)
    # for i, img in enumerate(trainx):
    #     if i%1000 == 0:
    #         print(i)
    #     img = skimage.transform.resize(img,(110,110))
    #     train.append(img)

    # # print(np.s)
    trainx = torch.from_numpy(trainx[:,10:,30:-20,:])
    trainy = torch.from_numpy(np.argmax(trainy, axis=1))
    testy = torch.from_numpy(np.argmax(testy, axis=1))

    dataset = torch.utils.data.TensorDataset(trainx, trainy)
    print("here", trainx.size())
    testset = torch.utils.data.TensorDataset(testx, testy)
    
elif args.dataset == 'lvh':
    trainx1, trainy1 = helper.read_dataset(args.dataroot + "/FILE.hdf5")
    trainx2, trainy2 = helper.read_dataset(args.dataroot + "/FILE.hdf5")
    unlabx1 = helper.read_unlab_dataset(args.dataroot + "/FILE.hdf5")
    unlabx2 = helper.read_unlab_dataset(args.dataroot + "/FILE.hdf5")
    unlabx3 = helper.read_unlab_dataset(args.dataroot + "/FILE.hdf5")
    print(np.shape(unlabx1))
    unlaby1 = np.ones(np.shape((unlabx1[0],1)))

    trainx = []
    unlabx = []
    trainy = []
    for i, img in enumerate(trainx1):
        trainx.append(skimage.transform.resize(img[:,:,0],(110,110)))
        trainy.append(trainy1[i])
    for i, img in enumerate(trainx2):
        trainx.append(skimage.transform.resize(img[:,:,0], (110,110)))
        trainy.append(trainy2[i])
    for img in unlabx1:
        img = np.reshape(img, (600,800))
        unlabx.append(skimage.transform.resize(img, (110,110)))
    for img in unlabx2:
        img = np.reshape(img, (600,800))
        unlabx.append(skimage.transform.resize(img, (110,110)))
    for img in unlabx3:
        img = np.reshape(img, (180,240))
        unlabx.append(skimage.transform.resize(img, (110,110)))




    testx, testy = helper.read_dataset(args.dataroot + "/FILE.hdf5")
    trainx = torch.from_numpy(np.array(trainx))
    testx = torch.from_numpy(testx[:,10:,30:-20,:])
    unlabx = torch.from_numpy(np.array(unlabx))
    # trainy = np.array(trainy)
    # print(trainy[0])
    # print(np.shape(trainy))

    labels = np.argmax(trainy, axis=1)
    trainy = torch.from_numpy(labels)
    # trainy = torch.from_numpy(np.hstack((labels, unlaby1)))
    # trainy = torch.from_numpy(labels)
    unlaby1 = torch.ones(unlabx.size()[0],1)
    testy = torch.from_numpy(np.argmax(testy, axis=1))
    dataset = torch.utils.data.TensorDataset(trainx, trainy)
    testset = torch.utils.data.TensorDataset(testx, testy)
    
elif args.dataset == 'mnist':
    dataset = dset.MNIST(root=args.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor()
                               ,transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), #seems like torchvision is buggy for 1 channel normalize
                           ]))
elif args.dataset == 'fake':
    dataset = dset.FakeData(image_size=(3, args.imageSize, args.imageSize),
                            transform=transforms.ToTensor())
assert dataset
ngpu = int(args.ngpu)
nz = int(args.nz)
ngf = int(args.ngf)
ndf = int(args.ndf)
if args.dataset == 'mnist':
    nc = 1 
    nb_label = 10
elif args.dataset == 'ecg':
    nc = 1
    nb_label = 15
elif args.dataset == 'lvh':
    nc = 1
    nb_label = 2
else:
    nc = 3
    nb_label = 10


dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize,
                                         shuffle=True, num_workers=int(0))

txs = []
tys = []
for data in dataloader:

    img, label = data
    txs.append(img)
    tys.append(label)
# testx = []
# for data in testset:
#     img = data
#     testx.append(img)
cats = dict()
for i in range(nb_label):
    cats[i] = []
for i,lab in enumerate(tys):
#     print(cats)
    lab = lab.numpy()[0]
#     print(lab)
    if lab in cats:
        a = cats[lab]
        a.append(i)
        cats[lab] = a
    else:
        a = [i]
        cats[lab] = a

#Making the labelled/unlabelled split
if args.few == 0:
    if args.dataset == 'lvh':
    
    # print(torch.stack(txs, dim=0).size())
    # input()
        x_unlab = torch.stack(unlabx)
        x_unlab = torch.unsqueeze(x_unlab, 1)
        # print(x_unlab.size())
        # input()
        print(unlaby1.size())
        y_unlab = torch.stack(unlaby1, dim=0)
        x_lab = torch.stack(txs)
        y_lab = torch.stack(tys,dim=0)
        for key in cats.keys():
            print(key)
            print(np.shape(cats[key]))
        print(x_lab.size(), "x_lab")
        print(x_unlab.size(), "x_unlab")
        # input()
    else:
        txs = np.array(txs)
        tys = np.array(tys)
        labelled_inds = []
        # for key in cats.keys():
        #     a = cats[key][:]
        #     splitinds = int(len(a)*args.split)
        #     print(key, np.shape(cats[key]))
            # labelled_inds.extend(cats[key][:splitinds])
        txs = torch.from_numpy(txs)
        tys = torch.from_numpy(tys)
        print(torch.stack(txs, dim=0).size())
        x_unlab = torch.squeeze(torch.stack(txs, dim=0), dim=4)
        y_unlab = torch.stack(tys, dim=0)
        x_lab = torch.squeeze(torch.stack(txs,dim=0), dim =4)
        y_lab = torch.stack(tys,dim=0)
else:
    if args.experiment == 0:
        txs = np.array(txs)
        tys = np.array(tys)
        splitinds = args.numlabelled
        labelled_inds = []
        for key in cats.keys():
            print(key)
            labelled_inds.extend(cats[key][:splitinds])
        # x_unlab = torch.squeeze(torch.stack(txs, dim=0), dim=4)
        # y_unlab = torch.stack(tys, dim=0)
        # x_lab = torch.squeeze(torch.stack(txs[labelled_inds],dim=0), dim =4)
        # y_lab = torch.stack(tys[labelled_inds],dim=0)
    else:
        # nb_label = 2
        txs = np.array(txs)
        tys = np.array(tys)
        splitinds = args.numlabelled
        labelled_inds = []
        unlabelled_inds = []

        for key in cats.keys():
            print(key, np.shape(cats[key]))
            if key == 5:
                labelled_inds.extend(cats[key][:splitinds])
                a = cats[key][:]
                index = int(0.5*args.experiment*len(a))
                print(len(a), "len 5")
                unlabelled_inds.extend(a[:index])
            elif key == 3:
                labelled_inds.extend(cats[key][:splitinds])
                a = cats[key][:]
                index  = int(len(a))
                print(len(a), "len 3")
                unlabelled_inds.extend(a[:index])


    x_unlab = torch.squeeze(torch.stack(txs[unlabelled_inds], dim=0), dim=4)
    y_unlab = torch.stack(tys[unlabelled_inds], dim=0)
    x_lab = torch.squeeze(torch.stack(txs[labelled_inds],dim=0), dim =4)
    y_lab = torch.stack(tys[labelled_inds],dim=0)

    print(x_unlab.size(), "unlab size")



labset = torch.utils.data.TensorDataset(x_lab, y_lab)
unlabset = torch.utils.data.TensorDataset(x_unlab, y_unlab)

labloader = torch.utils.data.DataLoader(labset, batch_size=args.batchSize,
                                         shuffle=True, num_workers=int(1), drop_last = False)

unlabloader = torch.utils.data.DataLoader(unlabset, batch_size=args.batchSize,
                                         shuffle=False, num_workers=int(1), drop_last = True)


# labloader = dataloader
# unlabloader = dataloader




netG = _netG(ngpu, nz, ngf, nc, args)
netD = _netD(ngpu, ndf, nc, nb_label, args)


if (args.resume == 1):
    epoch  = int(args.netD[-6:-4])



if args.netG != '':
    netG.load_state_dict(torch.load(args.netG))
print(netG)





if args.netD != '':
    netD.load_state_dict(torch.load(args.netD))
print(netD)

d_criterion = nn.BCEWithLogitsLoss()
c_criterion = nn.CrossEntropyLoss()
gen_criterion = nn.MSELoss()

input = torch.FloatTensor(args.batchSize, 3, args.imageSize, args.imageSize)
input2 = torch.FloatTensor(args.batchSize, 3, args.imageSize, args.imageSize)

noise = torch.FloatTensor(args.batchSize, nz)
fixed_noise = torch.FloatTensor(args.batchSize, nz, 1,1).normal_(0, 1)
d_label = torch.FloatTensor(args.batchSize,1)
c_label = torch.LongTensor(args.batchSize,1)


real_label = 1
fake_label = 0



fixed_noisev = Variable(fixed_noise)
d_labelv = Variable(d_label)
c_labelv = Variable(c_label)
noisev = Variable(noise)
inputv = Variable(input)
input2v = Variable(input2)

fixed_noise_ = np.random.normal(0,1, (args.batchSize, nz))
random_label = np.random.randint(0, nb_label, args.batchSize)
print('fixed label:{}'.format(random_label))
random_onehot = np.zeros((args.batchSize, nb_label))
random_onehot[np.arange(args.batchSize), random_label] = 1
fixed_noise_[np.arange(args.batchSize), :nb_label] = random_onehot[np.arange(args.batchSize)]


fixed_noise_ = (torch.from_numpy(fixed_noise_))
fixed_noise_ = fixed_noise_.resize_(args.batchSize,nz,1,1)
fixed_noise.copy_(fixed_noise_)

if args.cuda:
    netD.cuda()
    netG.cuda()
    netD = torch.nn.parallel.DataParallel(netD, device_ids=[0,1])
    netG = torch.nn.parallel.DataParallel(netG, device_ids=[0,1])
    d_criterion.cuda()
    c_criterion.cuda()
    gen_criterion.cuda()
    input, d_label = input.cuda(), d_label.cuda()
    input2 = input2.cuda()
    c_label = c_label.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()


#costs
# output

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

schedulerD = torch.optim.lr_scheduler.StepLR(optimizerD, step_size=1, gamma=0.2)
schedulerG = torch.optim.lr_scheduler.StepLR(optimizerG, step_size=1, gamma=0.2)


def test(predict, labels):
    correct = 0
    pred = predict.data.max(1)[1]
    correct = pred.eq(labels).cpu().sum()
    return correct, len(labels)
    
def feature_loss(X1, X2):

    m1 = torch.mean(X1, 0)
    m2 = torch.mean(X2, 0)
    loss = torch.mean(torch.abs(m1 - m2))
    return loss    

loss_g = list()
loss_d = list()
if args.resume == 1:
    start_iter = epoch
else:
    start_iter = 0



for epoch in range(start_iter, start_iter + args.niter):
    # schedulerG.step()
    # schedulerD.step()

    x_lab = [] 
    y_lab = []
    for data in labloader:
        img, label = data
        x_lab.append(img)
        y_lab.append(label)
    num_labs = len(x_lab)


    x_unlab = []
    y_unlab = []    
    for data in unlabloader:
        img, label = data
        x_unlab.append(img)
        y_unlab.append(label)
    
    # print(img.size())
    # input()
    total_correct_unl = 0
    total_length_unl = 0

    total_correct_lab = 0
    total_length_lab = 0

    for i, img2 in enumerate(x_unlab):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with labelled
        netD.zero_grad()
        i2 = i % num_labs
        batch_size = args.batchSize
        label = y_lab[i2]

        unl_label = y_unlab[i]
        img = x_lab[i2]

        if (img.size()[0] != batch_size):
            x_lab = []
            y_lab = []   
            for data in labloader:
                img, label = data
                x_lab.append(img)
                y_lab.append(label)


        if args.cuda:
            img = img.cuda()


        # print(img.size())
        input.resize_(img.size()).copy_(img)
        input2.resize_(img2.size()).copy_(img2)



        d_label.resize_(batch_size,1).fill_(real_label)


        c_label.resize_(label.size()).copy_(label)
        c_label = torch.squeeze(c_label)

   
        inputv = Variable(input)
        input2v = Variable(input2)
        d_labelv = Variable(d_label)
        c_labelv = Variable(c_label)
        labelv = Variable(label)

        discriminate, before, after, last = netD(inputv)

        # print(label.size(), "label size")
        # print(c_label, "c_label")
        # print(before, "before")
        c_errD_labelled = c_criterion(before, c_labelv)

        errD_real = c_errD_labelled
        errD_real.backward()
        
        input.resize_(img2.size()).copy_(img2)
        inputv = Variable(input)


        discriminate2, before2, after2, last2 = netD(inputv)

        # l_lab = Variable(before.data[torch.from_numpy(np.arange(batch_size)).cuda(),c_label])

        # l_unl = helper.log_sum_exp(before2)



        D_x = 0.5*discriminate.data.mean() + 0.5*discriminate2.data.mean()

        correct, length = test(after, c_label)
        c_label.resize_(batch_size).copy_(unl_label)

        # print(c_label.size(), "size")
        correct_unl, length_unl = test(after2, c_label)


        total_correct_unl += correct_unl
        total_length_unl += length_unl

        total_correct_lab += correct
        total_length_lab += length

        # train with fake
        noise.resize_(batch_size, nz,1,1).normal_(0, 1)

        # noise.resize_(batch_size, nz, 1, 1).normal_(0, 1)
        noisev = Variable(noise)

        label = np.random.randint(0, nb_label, batch_size)
        noise_ = np.random.normal(0,1, (batch_size, nz))
        label_onehot = np.zeros((batch_size, nb_label))
        label_onehot[np.arange(batch_size), label] = 1
        noise_[np.arange(batch_size), :nb_label] = label_onehot[np.arange(batch_size)]

        noise_ = (torch.from_numpy(noise_))
        noise_ = noise_.resize_(batch_size, nz, 1, 1)
        noise.copy_(noise_)


        noisev = Variable(noise)
        fake = netG(noisev)


        d_label = d_label.fill_(real_label)
        d_labelv = Variable(d_label)
        loss_unl = d_criterion(discriminate2, d_labelv)
        loss_unl.backward()



        d_label = d_label.fill_(fake_label)
        d_labelv = Variable(d_label)
        discriminate3, before3, after3, last3 = netD(fake.detach())
        loss_fake = d_criterion(discriminate3, d_labelv)
        loss_fake.backward()

        # z_exp_unl = helper.log_sum_exp(before2)
        # z_exp_fake = helper.log_sum_exp(before3)

        # l_gen = helper.log_sum_exp(before3)
        # softplus = torch.nn.Softplus()
  
        errD_fake = loss_unl + loss_fake


        # errD_fake.backward(retain_graph = True)
        D_G_z1 = discriminate3.data.mean()
        errD = errD_real + errD_fake
        # errD.backward(retain_graph = True)
        optimizerD.step()


        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        # print(input2v.data)
        # input.resize_(img.size()).copy_(img)
        # inputv = Variable(input)
        # discriminate, before, after = netD(inputv)

        fake = netG(noisev)
        discriminate, before, after, last  = netD(fake)
        discriminate2, before2, after2, last2  = netD(inputv.detach())
        # print(before)
        # gen_loss = feature_loss(before, before2)

        d_labelv = Variable(d_label.fill_(real_label))  # fake labels are real for generator cost
        # # d_output, c_output = netD(fake)
        # d_errG = d_criterion(discriminate2, d_labelv)
        # c_errG = c_criterion(c_output, c_label)

        # m1 = torch.mean(last, 0)
        # m2 = torch.mean(last2, 0)
        # # print(m1 - m2)
        # loss_gen = torch.mean(torch.abs(m1-m2))
        last2v = Variable(last2.data, requires_grad = False)
        gen_loss = gen_criterion(torch.mean(last, 0), torch.mean(last2v,0))
        # gen_loss = d_criterion(discriminate, d_labelv)

        # errG = loss_gen
        errG = gen_loss
        errG.backward()
        D_G_z2 = discriminate.data.mean()
        optimizerG.step()


        loss_d.append(errD.detach().data)
        loss_g.append(errG.detach().data)
        print('[%d/%d][%d/%d] Loss_D: %.4f Fake_Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f Correct_l: %.4f Lab_Length: %.4f Correct_unl: %.4f Unlab_Length: %.4f' 
              % (epoch, args.niter, i, len(x_unlab),
                 errD.data[0], errD_fake.data[0], errG.data[0], D_x, D_G_z1, D_G_z2, correct, length, correct_unl,  length_unl))
        if i % 100 == 0:
            vutils.save_image(img,
                    '%s/real_samples.png' % args.outf,
                    normalize=True)
            fake = netG(fixed_noisev)
            # print(fake.data)
            vutils.save_image(fake.data,'%s/fake_samples_epoch_%03d.png' % (args.outf, epoch),
                    normalize=True)

    lab_training_error = 1.0 - float(total_correct_lab)/float(total_length_lab)
    unlab_training_error = 1.0 - float(total_correct_unl)/float(total_length_unl)        
    print('[%d/%d] Labelled_Training_Error: %.4f Unlabelled_Training_Error: %.4f'
              % (epoch, args.niter,lab_training_error, unlab_training_error))
    # do checkpointing
    d = np.array(loss_d)
    g = np.array(loss_g)
    np.save('%s/LossG_epoch_%d.npy' % (args.outf, epoch), g)
    np.save('%s/LossD_epoch_%d.npy' % (args.outf, epoch), d)
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (args.outf, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (args.outf, epoch))
