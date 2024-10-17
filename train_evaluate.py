import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
import torchvision.transforms as T
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import time

from Moving_MNIST import *
from adrnetwork import ADRnet  #advection_diffusion_reaction_network / Advection Augmented CNN
from metrics import metric

def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)        

#########################

device = get_device()
print('Device: ',device)

history = 10
prediction = 10

# Generates random movies
mm = MovingMNIST(root='/data/sid/PhyDNetMovingMNIST/data/MNIST/', is_train=True, n_frames_input=history, n_frames_output=prediction, num_objects=[2])
train_loader = torch.utils.data.DataLoader(dataset=mm, batch_size=16, shuffle=False, num_workers=4)  # 10k generated samples train

# Standard datset
mmtst = MovingMNIST(root='/data/sid/PhyDNetMovingMNIST/data/', is_train=False, n_frames_input=history, n_frames_output=prediction, num_objects=[2])
test_loader = torch.utils.data.DataLoader(dataset=mmtst, batch_size=16, shuffle=False, num_workers=0) # 10k test samples len(mm2)

in_c = history
SZ = 64

# Advection Diffusion Reaction Network
model = ADRnet(in_c=history, hid_c=256, out_c=prediction, nlayers=9, imsz=[SZ, SZ], device=device).to(device)

print('Number of model parameters = %3d'%(count_parameters(model)))

lr = 1e-3
optim = Adam(model.parameters(), lr)
epochs = 100
scheduler = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs)
s = 1.0
gamma = np.exp(np.log(0.001)/epochs)

torch.autograd.set_detect_anomaly(True) 
tqdm_epoch = tqdm(range(epochs), desc=f"Training progress")
start = time.time()
loss_history = []
for k in tqdm_epoch:
    temp_loss = 0
    s = gamma*s
    for j,data in enumerate(train_loader): 
        
        optim.zero_grad()

        tt, xx, yy = data               # index[batch_size], input[batch_size, history, 1, 64, 64], output[batch_size, prediction, 1, 64, 64]    
        
        xx = xx.to(device)
        yy = yy.to(device)
        tt = tt.to(device)

        xx = xx[:,:,0,:,:]
        yy = yy[:,:,0,:,:]

        tt = tt*0.
        
        ycomp = model(xx, tt)       
        
        loss = F.mse_loss(ycomp, yy)/F.mse_loss(yy*0, yy)     # misfit
    
        loss.backward()
        optim.step()
        scheduler.step()

        tqdm_epoch.set_description('epochs = %3d.%3d   Loss =  %3.2e'%(k,j,loss))
        
        temp_loss+=loss.item()

    loss_history.append(temp_loss/(j+1))
stop = time.time()

print('Done Training\n')

plt.plot(np.array(loss_history))
plt.xlabel('epochs')
plt.ylabel('loss (misfit)')
plt.yscale("log")
plt.title("Training loss curve")
plt.savefig("plots/Training_loss.png")

# Save model
torch.save(model, "model-full.pth")

tqdm_epoch = tqdm((test_loader), desc=f"Testing progress")

MSE = 0
MAE = 0
SSIM = 0
PNSR = 0
loss_history = []

for j,data in enumerate(tqdm_epoch):

    tt, xx, yy = data
    
    xx = xx.to(device)
    yy = yy.to(device)
    tt = tt.to(device)
    
    xx = xx[:,:,0,:,:]
    yy = yy[:,:,0,:,:]

    tt = tt*0.
    
    ycomp = model(xx, tt)
    
    loss = F.mse_loss(ycomp, yy)        # MSE
    mf_loss = F.mse_loss(ycomp, yy)/F.mse_loss(yy*0, yy)     # misfit
    
    tqdm_epoch.set_description('step# = %3d   Loss =  %3.2e'%(j,mf_loss))
    
    ycomp = ycomp.detach().cpu().numpy()
    yy = yy.detach().cpu().numpy()
    
    m1,m2,m3,m4 = metric(ycomp, yy, yy.mean(), yy.std()) # mse, mae, ssim, psnr
    
    MSE = MSE + m1
    MAE = MAE + m2
    SSIM = SSIM + m3
    PNSR = PNSR + m4    
    
    loss_history.append(mf_loss.item())
    
    # Testing plots
    if j < 5:
        fig, axs = plt.subplots(4, 2, figsize=(20, 20))
        c1 = axs[0, 0].imshow(yy[0,0], cmap='bone')
        axs[0, 0].set_title('First Reference')
        plt.colorbar(c1,ax = axs[0, 0])
        c2 = axs[0, 1].imshow(ycomp[0,0], cmap='bone')
        axs[0, 1].set_title('First Prediction')
        plt.colorbar(c2,ax = axs[0, 1])
        c3 = axs[1, 0].imshow((yy-ycomp)[0,0], cmap='bone')
        axs[1, 0].set_title('First error')
        plt.colorbar(c3,ax = axs[1, 0])   
        c4 = axs[2, 0].imshow(yy[0,-1], cmap='bone')
        axs[2, 0].set_title('Last Reference')
        plt.colorbar(c4,ax = axs[2, 0])
        c5 = axs[2, 1].imshow(ycomp[0,-1], cmap='bone')
        axs[2, 1].set_title('Last Prediction')
        plt.colorbar(c5,ax = axs[2, 1])
        c6 = axs[3, 0].imshow((yy-ycomp)[0,-1], cmap='bone')
        axs[3, 0].set_title('Last error')
        plt.colorbar(c6,ax = axs[3, 0]) 
        plt.savefig("plots/Comparison"+str(j)+".png")


MSE = MSE/len(test_loader)
MAE = MAE/len(test_loader)
SSIM = SSIM/len(test_loader)
PNSR = PNSR/len(test_loader)

print("\nMSE: ",MSE,"\tMAE: ",MAE,"\tSSIM: ",SSIM,"\tPNSR: ",PNSR)

print('\nDone Testing')