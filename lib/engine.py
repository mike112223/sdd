import time
import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from .utils import Meter


class Trainer(object):
    '''This class takes care of training and validation of our model'''
    def __init__(self, model, criterion, dataloaders, phases, batch_size, lr, num_epochs, device, restart_epoch, save_frequency):

        self.device = device
        self.batch_size = batch_size 
        self.accumulation_steps = 32 // self.batch_size['train']
        self.lr = lr 
        self.num_epochs = num_epochs
        self.best_loss = float('inf')
        self.best_score = float('-inf')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        self.net = model
        self.criterion = criterion 
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=3, verbose=True)
        # self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-4)
        # self.scheduler = StepLR(self.optimizer, step_size=self.num_epochs//3, gamma=0.1)

        self.net = self.net.to(self.device)
        self.restart_epoch = restart_epoch
        self.save_frequency = save_frequency

        cudnn.benchmark = True
        self.dataloaders = dataloaders
        
    def forward(self, images, targets):
        images = images.to(self.device)
        masks = targets.to(self.device)
        outputs = self.net(images)
        if isinstance(outputs, dict):
            if 'aux' in outputs.keys():
                outputs = (outputs['out']+outputs['aux'])/2
            else:
                outputs = outputs['out']

        loss = self.criterion(outputs, masks)
        return loss, outputs

    def iterate(self, phase):
        meter = Meter(phase)
        batch_size = self.batch_size[phase]
        self.net.train(phase=='train')
        dataloader = self.dataloaders[phase]

        self.optimizer.zero_grad()
        for itr, batch in enumerate(dataloader): # replace `dataloader` with `tk0` for tqdm
            images, targets = batch
            if phase == 'train':
                loss, outputs = self.forward(images, targets)
                loss.backward()
                if (itr + 1 ) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            else:
                with torch.no_grad():
                    loss, outputs = self.forward(images, targets)

            outputs = outputs.detach().cpu()
            progress = '%d/%d' % (itr, len(dataloader))
            meter.update(targets, outputs, loss, progress)
        torch.cuda.empty_cache()

        return meter.summary()

    def start(self, savedir):
        for epoch in range(self.num_epochs):
            if epoch % self.restart_epoch == 0:
                print('reset lr')
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr

            start = time.strftime('%m/%d-%H:%M:%S')
            print(f'Starting epoch: {epoch} | phase: train | {start}')
            print('lr:', self.optimizer.param_groups[0]['lr'])
            self.iterate('train')
            state = {
                'epoch': epoch,
                'best_loss': self.best_loss,
                'state_dict': self.net.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'score': self.best_score
            }

            start = time.strftime('%m/%d-%H:%M:%S')
            print(f'Starting epoch: {epoch} | phase: val | {start}')
            val_loss, val_score = self.iterate('val')
            self.scheduler.step(val_loss)
            # self.scheduler.step()

            if val_score > self.best_score:
                print('******** New dice optimal found, saving state ********')
                state['best_score'] = self.best_score = val_score
                print('save as %s/model_dice_best.pth' % savedir)
                torch.save(state, '%s/model_dice_best.pth' % savedir)

            if val_loss < self.best_loss:
                print('******** New loss optimal found, saving state ********')
                state['best_loss'] = self.best_loss = val_loss
                print('save as %s/model_loss_best.pth' % savedir)
                torch.save(state, '%s/model_loss_best.pth' % savedir)

            if epoch % self.save_frequency == 0 and epoch != 0:
                print('save as %s/model_%d.pth' % (savedir, epoch))
                torch.save(state, '%s/model_%d.pth' % (savedir, epoch))

            print('save as %s/model_latest.pth' % (savedir))
            torch.save(state, '%s/model_latest.pth' % (savedir))            

            print()

