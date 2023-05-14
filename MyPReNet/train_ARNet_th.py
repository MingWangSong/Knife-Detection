import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from DerainDataset import *
from utils import *
from torch.optim.lr_scheduler import MultiStepLR
from SSIM import SSIM
#from networks import *
from networks_m_ed_attn_arth import *


parser = argparse.ArgumentParser(description="PReNet_train")
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--batch_size", type=int, default=18, help="Training batch size")
parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=[40,80,140], help="When to decay learning rate")
parser.add_argument("--lr", type=float, default=1e-4, help="initial learning rate")
parser.add_argument("--save_path", type=str, default="logs/PReNet_test", help='path to save models and log files')
parser.add_argument("--save_freq",type=int,default=1,help='save intermediate model')
parser.add_argument("--data_path",type=str, default="datasets/train/Rain12600",help='path to training data')
parser.add_argument("--use_gpu", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument("--recurrent_iter", type=int, default=6, help='number of recursive stages')
opt = parser.parse_args()

if opt.use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

def tensor_to_np(tensor):
#    print(tensor.shape)
    img = tensor.mul(255).byte()
    img = img.detach().cpu().numpy().squeeze(0).transpose((1, 2, 0))
    return img

def toTensor(img):
    assert type(img) == np.ndarray,'the img type is {}, but ndarry expected'.format(type(img))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().div(255).unsqueeze(0)  # 255也可以改为256

def main():

    print('Loading dataset ...\n')
    dataset_train = Dataset(data_path=opt.data_path)
    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batch_size, shuffle=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))

    # Build model
    model = PReNet(recurrent_iter=opt.recurrent_iter, use_GPU=opt.use_gpu)
    print_network(model)

    # loss function
    # criterion = nn.MSELoss(size_average=False)
    criterion = SSIM()

    # Move to GPU
    if opt.use_gpu:
        model = model.cuda()
        criterion.cuda()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = MultiStepLR(optimizer, milestones=opt.milestone, gamma=0.2)  # learning rates

    # record training
    writer = SummaryWriter(opt.save_path)

    # load the lastest model
    initial_epoch = findLastCheckpoint(save_dir=opt.save_path)
    if initial_epoch > 0:
        print('resuming by loading epoch %d' % initial_epoch)
        model.load_state_dict(torch.load(os.path.join(opt.save_path, 'net_epoch%d.pth' % initial_epoch)))

    # start training
    step = 0
    for epoch in range(initial_epoch, opt.epochs):
        scheduler.step(epoch)
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])

        ## epoch training start
        for i, (input_train, target_train) in enumerate(loader_train, 0):
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            #thresholding
            #import pdb;pdb.set_trace()
            for kk in range(input_train.size()[0]):
                        
                input_train_np = tensor_to_np(input_train[kk,:,:,:].unsqueeze(0))
                blur = cv2.GaussianBlur(input_train_np, (5, 5), 0)
                ret3, th1 = cv2.threshold(blur, 235, 255, cv2.THRESH_BINARY)

#            print("logs/th/"+str(step)+"_th.png")
#            cv2.imwrite("logs/th/"+str(step)+"_th.png",np.uint8(th3))
#            cv2.imwrite("logs/th/"+str(step)+"_raw.png",np.uint8(tensor_to_np(input_train)))
                
                th1=toTensor(th1)
                if kk==0:
                    th3 = th1
                else:
                    th3 = torch.cat((th3,th1),0)
#                import pdb;pdb.set_trace()                
            th3 = Variable(th3)
#            import pdb;pdb.set_trace() 
            input_train, target_train = Variable(input_train), Variable(target_train)

            if opt.use_gpu:
                input_train, target_train = input_train.cuda(), target_train.cuda()
                th3 = th3.cuda()

            out_train, _ = model(input_train,th3)
            pixel_metric = criterion(target_train, out_train)
            loss = -pixel_metric

            loss.backward()
            optimizer.step()

            # training curve
            model.eval()
            out_train, _ = model(input_train,th3)
            out_train = torch.clamp(out_train, 0., 1.)
            psnr_train = batch_PSNR(out_train, target_train, 1.)
            print("[epoch %d][%d/%d] loss: %.4f, pixel_metric: %.4f, PSNR: %.4f" %
                  (epoch+1, i+1, len(loader_train), loss.item(), pixel_metric.item(), psnr_train))

            if step % 10 == 0:
                # Log the scalar values
                writer.add_scalar('loss', loss.item(), step)
                writer.add_scalar('PSNR on training data', psnr_train, step)
            step += 1
            
        ## epoch training end

        # log the images
        model.eval()
        out_train, _ = model(input_train,th3)
        out_train = torch.clamp(out_train, 0., 1.)
        im_target = utils.make_grid(target_train.data, nrow=8, normalize=True, scale_each=True)
        im_input = utils.make_grid(input_train.data, nrow=8, normalize=True, scale_each=True)
        im_derain = utils.make_grid(out_train.data, nrow=8, normalize=True, scale_each=True)
        writer.add_image('clean image', im_target, epoch+1)
        writer.add_image('rainy image', im_input, epoch+1)
        writer.add_image('deraining image', im_derain, epoch+1)
	    # cal test dataset
        model.eval()
        psnrTotal = 0
        SSIMTotal = 0    
        count = 0
        for img_name in os.listdir("datasets/test/clean306"):
            if is_image(img_name):
                img_path = os.path.join("datasets/test/clean306", img_name)
                img_path_noclean = os.path.join("datasets/test/noclean306", "no"+img_name)
                # input image
                y = cv2.imread(img_path)
                y_noclean = cv2.imread(img_path_noclean)	            
                blur_noclean = cv2.GaussianBlur(y_noclean, (5, 5), 0)
                ret3, th3_noclean = cv2.threshold(blur_noclean, 235, 255, cv2.THRESH_BINARY )	            
                
                b, g, r = cv2.split(y)
                b1, g1, r1 = cv2.split(y_noclean)	  
                b2, g2, r2 = cv2.split(th3_noclean)	 
                                          
                y1 = cv2.merge([r1, g1, b1])
                y = cv2.merge([r, g, b])
                y2 = cv2.merge([r2, g2, b2])                	            
                #y = cv2.resize(y, (int(500), int(500)), interpolation=cv2.INTER_CUBIC)
	
                y = normalize(np.float32(y))
                y = np.expand_dims(y.transpose(2, 0, 1), 0)
                y = Variable(torch.Tensor(y))
		       
                y1 = normalize(np.float32(y1))
                y1 = np.expand_dims(y1.transpose(2, 0, 1), 0)
                y1 = Variable(torch.Tensor(y1))

                y2 = normalize(np.float32(y2))
                y2 = np.expand_dims(y2.transpose(2, 0, 1), 0)
                y2 = Variable(torch.Tensor(y2))
	
                y = y.cuda()
                y1 = y1.cuda()
                y2 = y2.cuda()
                with torch.no_grad():                 
                    out, _ = model(y1,y2) 
                    out = torch.clamp(out, 0., 1.)                                                           	            
                SSIMTotal += criterion(y,out)
                psnrTotal += batch_PSNR(y, out, 1.)	

                count += 1     
        ssimavg = SSIMTotal/count            		  
        psnravg = psnrTotal/count
        print('SSIM test:', ssimavg.item())
        print('PSNR test:', psnrTotal/count)
        print('epoch:', epoch+1)
        writer.add_scalar('SSIMtest', ssimavg.item(), epoch+1)
        writer.add_scalar('PSNRtest', psnravg, epoch+1)

        # save model
        torch.save(model.state_dict(), os.path.join(opt.save_path, 'net_latest.pth'))
        if epoch % opt.save_freq == 0:
            torch.save(model.state_dict(), os.path.join(opt.save_path, 'net_epoch%d.pth' % (epoch+1)))


if __name__ == "__main__":
    if opt.preprocess:
        if opt.data_path.find('RainTrainH') != -1:
            print(opt.data_path.find('RainTrainH'))
            prepare_data_RainTrainH(data_path=opt.data_path, patch_size=100, stride=80)
        elif opt.data_path.find('RainTrainL') != -1:
            prepare_data_RainTrainL(data_path=opt.data_path, patch_size=100, stride=80)
        elif opt.data_path.find('Rain12600') != -1:
            prepare_data_Rain12600(data_path=opt.data_path, patch_size=100, stride=100)
        else:
            print('unkown datasets: please define prepare data function in DerainDataset.py')


    main()
