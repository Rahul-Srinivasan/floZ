import os
import time
import copy
import torch
import numpy as np
import logging
from itertools import combinations
import matplotlib.pyplot as plt
import torch.nn.functional as F
import math
plt.rcParams.update({'font.size': 20})
plt.rc('font',**{'family':'serif'})
plt.rc('text', usetex=True)

logger = logging.getLogger(__name__)

from sklearn.model_selection import train_test_split

from .flow import Flow

def _pre_whitening(samples):
    
    cov     = np.cov(samples.T)
    w, v    = np.linalg.eig(cov)
    
    # project samples
    output  = np.zeros(samples.shape)
    for i in range(len(w)): # run over dimensions
        for j in range(len(samples)): # run over samples
            output[j][i]  = (v[:,i]*samples[j]).sum()/np.sqrt(w[i])
            
    return output, 0.5*np.log(np.abs(np.prod(w)))

class Trainer:

    def __init__(self,
                 # posterior input
                 posterior_samples,
                 log_prob,
                 # basic inputs with default
                 outdir                 = './',
                 seed                   = None,
                 device                 = None,
                 # normalizing flow settings
                 base_dist              = 'normal',
                 transform              = 'maf',
                 transform_dict         = {},
                 # optmizer settings
                 learning_rate          = 1e-4,
                 weight_decay           = 1e-6,
                 # data settings
                 n_train_batches        = 50,
                 validation_fraction    = None,
                 shuffle_batches        = True,
                 pre_whitening          = True,
                 lnZ_GT                 = None,
                 ):
        from .utils import _ensure_dir, _set_logger

        # check input data: device
        if (device == 'gpu' or device == 'cuda') and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        # check input data: samples
        assert len(posterior_samples) == len(log_prob)
        self.nsamples, self.ndim = posterior_samples.shape

        # check input data: output directory
        self.outdir = os.path.abspath(outdir)
        _ensure_dir(self.outdir)
        self.logger = _set_logger(outdir=self.outdir, silence=False)

        # Saving the GT of lnZ
        self.lnZ_GT = lnZ_GT
        self.logger.info('GT of lnZ: {}'.format(self.lnZ_GT))

        
        
        # check and set random seed
        if seed is None:
            seed = int(time.time())
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        
        # define flow
        # Initialize the model in CPU first
        self.flow           = Flow(ndim=self.ndim,
                                   transform=transform,
                                   base_dist=base_dist,
                                   transform_dict=transform_dict,
                                   device=torch.device('cpu'))
        
        # Check if multiple GPUs are available
        print('Available GPUs: ', torch.cuda.device_count())
        if torch.cuda.device_count() > 1:
            print("Using", torch.cuda.device_count(), "GPUs!")
            # Wrap the model in DataParallel
            self.flow = nn.DataParallel(self.flow)
            # Get the GPUs being used by DataParallel
            gpus_used = model.device_ids
            print("GPUs being used:", gpus_used)

        # Move model to GPU now
        self.flow.to(self.device)
    
        # define optmizer
        self.optimizer  = torch.optim.Adam(self.flow.parameters(),
                                           lr=learning_rate,
                                           weight_decay=weight_decay)
                                     
        # prints
        self.logger.info('Output directory: {}'.format(self.outdir))
        self.logger.info('Dimensions of parameter space: {}'.format(self.ndim))
        self.logger.info('Number of input samples: {}'.format(self.nsamples))
        self.logger.info('Number of network parameters: {}'.format(sum(p.numel() for p in self.flow.parameters())))
        self.logger.info('Working device: {}'.format(self.device))
            
        # pre-whitening
        if pre_whitening:
        
            posterior_samples, log_j_white  = _pre_whitening(posterior_samples)
            log_prob                        = log_prob + log_j_white
            _new_cov                        = np.cov(posterior_samples.T)
            self.logger.info('Whitening samples with covariance error of {:.3g}'.format(np.abs(np.max(_new_cov-np.eye(self.ndim)))))

        # store analyzed data
        self.posterior_samples  = torch.tensor(posterior_samples, dtype=torch.float32).to(self.device)
        self.log_prob           = torch.tensor(log_prob, dtype=torch.float32).to(self.device)

        # prepare data table (to be improved)
        _posterior_samples_t = posterior_samples.T
        x   = [_posterior_samples_t[i] for i in range(self.ndim)]
        x.append(log_prob)
        x   = np.transpose(x)
        
        # prepare data table: split validation and training sets
        if validation_fraction is None:
            X_train = x
            X_valid = x
        else:
            X_train, X_valid    = train_test_split(x, test_size=validation_fraction)
            self.logger.info('Splitting input samples in training set ({}) and validation set ({})'.format(X_train.shape[0], X_valid.shape[0]))

        # prepare data table: create training loader
        if n_train_batches is None:
            n_train_batches = X_train.shape[0]
            
        train_tensor    = torch.from_numpy(X_train.astype(np.float32)).to(self.device)
        train_dataset   = torch.utils.data.TensorDataset(train_tensor)
        train_loader    = torch.utils.data.DataLoader(train_dataset, batch_size=n_train_batches, shuffle=shuffle_batches)
        
        self.train_loader_data = []
        self.train_loader_prob = []
        for batch_idx, data in enumerate(train_loader):
            if isinstance(data, list):
                data = data[0]
            self.train_loader_prob.append(torch.t(data)[-1])
            self.train_loader_data.append(torch.t(torch.t(data)[:-1]))

        # prepare data table: create validation loader
        valid_tensor    = torch.from_numpy(X_valid.astype(np.float32)).to(self.device)
        valid_dataset   = torch.utils.data.TensorDataset(valid_tensor)
        # valid_loader    = torch.utils.data.DataLoader(valid_dataset, batch_size=X_valid.shape[0], shuffle=False, drop_last=True)
        valid_loader    = torch.utils.data.DataLoader(valid_dataset, batch_size=min(X_valid.shape[0], n_train_batches), shuffle=False, drop_last=True)

        self.valid_loader_data = []
        self.valid_loader_prob = []
        for batch_idx, data in enumerate(valid_loader):
            if isinstance(data, list):
                data = data[0]
            self.valid_loader_prob.append(torch.t(data)[-1])
            self.valid_loader_data.append(torch.t(torch.t(data)[:-1]))

        self.validation_lossweights_history = []
        self.train_lossweights_history = []
        
        # training arguments
        self.valid_loss_history = []
        self.train_loss_history = []
        self.train_loss_1_history = []
        self.train_loss_2_history = []
        self.train_loss_3a_history = []
        self.train_loss_3b_history = []
        self.validation_loss_1_history = []
        self.validation_loss_2_history = []
        self.validation_loss_3a_history = []
        self.validation_loss_3b_history = []
        self.l2_hist = []
        self.l2 = 0.
        
        self.evidence_error     = np.inf
        self.best               = { 'valid_loss'  : float('inf'),
                                    'train_loss'  : float('inf'),
                                    'epoch'       : 0,
                                    'flow'        : copy.deepcopy(self.flow)
                                    }

    def _get_logz_from_samples(self):
        self.flow.eval()
        with torch.no_grad():
            flow_prob = self.flow.log_prob(self.posterior_samples)
        return self.log_prob - flow_prob
        
    
    def _train(self):

        self.flow.train()
        # train_loss = np.zeros(2)
        train_loss = 0
        train_loss_1 = 0
        train_loss_2 = 0
        train_loss_3a = 0
        train_loss_3b = 0

        
        # Generate all combinations of pairs
        # print(self.train_loader_prob[0].shape, self.train_loader_prob[1].shape)
        pairs = np.array(list(combinations(np.arange(0,len(self.train_loader_prob[0]), 1), 2)))
        
        for batch_idx, (data,prob) in enumerate(zip(self.train_loader_data,  self.train_loader_prob)):
            self.optimizer.zero_grad()
            loss = 0
            # print(batch_idx)
            
            flow_prob   = self.flow.log_prob(data)
            log_z = prob - flow_prob
            
            log_meanz = (log_z-log_z.max()).exp().mean().log() + log_z.max()
            log_stdz  = (log_z-log_z.max()).exp().std().log()  + log_z.max()
            
            # loss_1 = -((flow_prob-flow_prob.max()).exp().mean().log() + flow_prob.max())
            # loss_1 = log_z.mean()
            # loss_1 = -(flow_prob*(prob-prob.max()).exp()).mean()/((prob-prob.max()).exp().sum())
            # loss_1 = -flow_prob.mean()
            # loss_1 = (((log_z-log_z.max()).exp()*prob).mean()/prob.sum()).log() + log_z.max()÷
            # weights = (prob - prob.max()).exp()/(prob - prob.max()).exp().sum()
            # loss_1 = ((log_z-log_z.max()).exp()*weights).sum().log() + log_z.max()
            loss_1 = log_meanz
            loss_2 = log_stdz
            train_loss_1 += loss_1.item()
            train_loss_2 += loss_2.item()

            # Compute loss3
            lnR = log_z[pairs[:,0]] - log_z[pairs[:,1]]
            loss_3b = (lnR-lnR.max()).exp().std().log() + lnR.max()
            loss_3a = ((lnR-lnR.max()).exp().mean().log() + lnR.max()).abs()
            # loss_3a = torch.pow( (lnR.exp().mean()-1)/lnR.exp().std() ,2)
            
            train_loss_3a += loss_3a.item()
            train_loss_3b += loss_3b.item()
            
            # loss = self.beta*loss_1 + (1-self.beta)*loss_2
            # loss_arr = [loss_1, loss_2, loss_3a, loss_3b]
            # loss_3 = torch.pow((lnR.exp()-1), 2).mean()
            loss_arr = [loss_1, loss_2, loss_3a, loss_3b]
                
            # loss = loss_1
            # loss = loss = self.l2*(loss_2+loss_3b) + (1-self.l2)*(loss_1+loss_3a)
            # loss = self.l2*loss_2 + loss_1

            # loss += loss_1 + .2*(self.beta*loss_arr[(self.lossseq)%len(loss_arr)] + (1-self.beta)*loss_arr[(self.lossseq+1)%len(loss_arr)])
            loss += self.beta*loss_arr[(self.lossseq)%len(loss_arr)] + (1-self.beta)*loss_arr[(self.lossseq+1)%len(loss_arr)]
           
            train_loss += loss.item()
            loss.backward()
            
            # loss_1.backward(retain_graph=True)
            # loss_3a.backward()
            self.optimizer.step()
            
        return train_loss / len(self.train_loader_data), train_loss_1 / len(self.train_loader_data), train_loss_2 / len(self.train_loader_data), train_loss_3a / len(self.train_loader_data), train_loss_3b / len(self.train_loader_data)
        
    def _validate(self):

        self.flow.eval()
        val_loss = 0
        val_loss_1 = 0
        val_loss_2 = 0
        val_loss_3a = 0
        val_loss_3b = 0

        # Generate all combinations of pairs
        # print(self.train_loader_prob[0].shape, self.train_loader_prob[1].shape)
        pairs = np.array(list(combinations(np.arange(0,len(self.valid_loader_data[0]), 1), 2)))
        
        for batch_idx, (data,prob) in enumerate(zip(self.valid_loader_data,  self.valid_loader_prob)):
            with torch.no_grad():
                flow_prob   = self.flow.log_prob(data)
                log_z = prob - flow_prob
                log_meanz = (log_z-log_z.max()).exp().mean().log() + log_z.max()
                log_stdz  = (log_z-log_z.max()).exp().std().log()  + log_z.max()
                
                # loss_1 = -((flow_prob-flow_prob.max()).exp().mean().log() + flow_prob.max())
                # loss_1 = log_z.mean()
                # loss_1 = -(log_z*(prob-prob.max()).exp()).mean()/((prob-prob.max()).exp().sum())
                # loss_1 = -flow_prob.mean()
                # loss_1 = (((log_z-log_z.max()).exp()*prob).mean()/prob.sum()).log() + log_z.max()÷
                # weights = (prob - prob.max()).exp()/(prob - prob.max()).exp().sum()
                # loss_1 = ((log_z-log_z.max()).exp()*weights).sum().log() + log_z.max()
                loss_1 = log_meanz
                loss_2 = log_stdz
                val_loss_1 += loss_1.item()
                val_loss_2 += loss_2.item()

                # Compute loss3
                lnR = log_z[pairs[:,0]] - log_z[pairs[:,1]]
                loss_3b = (lnR-lnR.max()).exp().std().log() + lnR.max()
                loss_3a = ((lnR-lnR.max()).exp().mean().log() + lnR.max()).abs()
                # loss_3a = torch.pow( (lnR.exp().mean()-1)/lnR.exp().std() ,2)
                val_loss_3a += loss_3a.item()
                val_loss_3b += loss_3b.item()
                
                # loss = self.beta*loss_1 + (1-self.beta)*loss_2
                # loss_arr = [loss_1, loss_2, loss_3a, loss_3b]
                # loss_3 = torch.pow((lnR.exp()-1), 2).mean()
                loss_arr = [loss_1, loss_2, loss_3a, loss_3b]
                
                # loss = loss_1
                # loss = loss = self.l2*(loss_2+loss_3b) + (1-self.l2)*(loss_1+loss_3a)
                # loss = self.l2*loss_2 + loss_1
    
                # loss = loss_1 + .2*(self.beta*loss_arr[(self.lossseq)%len(loss_arr)] + (1-self.beta)*loss_arr[(self.lossseq+1)%len(loss_arr)])
                loss = self.beta*loss_arr[(self.lossseq)%len(loss_arr)] + (1-self.beta)*loss_arr[(self.lossseq+1)%len(loss_arr)]
                      
                val_loss += loss.item()
                # val_loss += np.array([loss_1.item(), loss_3a.item()])
        
        return val_loss / len(self.valid_loader_data), val_loss_1 / len(self.valid_loader_data), val_loss_2 / len(self.valid_loader_data), val_loss_3a / len(self.valid_loader_data), val_loss_3b / len(self.valid_loader_data)
        
    def train(self,
              max_num_epoch      = 5000,
              patience           = 100,
              tolerance          = 0.1,
              log_interval       = 100,
              l2                 = 0):

        # define flow
        self.logger.info('Output directory: {}'.format(self.outdir))
        
        self.logger.info('Starting training of neural network ...')
        count_patience      = 0
        start_training_time = time.time()
        print('Total # batches: ',len(self.train_loader_data))
        self.beta = 0.
        self.delta_beta = -6/log_interval
        self.lossseq = -1
        start = 0
        
        # self.l2 = l2
        # if(l2 != 0):
        #     self.lossseq += 1
        # # self.l2_hist = []
        # lossvary = False
        cycle = True
        
        for epoch in range(1,max_num_epoch+1):
            # if(self.lossseq==4):
            #     self.logger.info('Completed L1 (conv) + 2 cycle + L1')
            #     break

            # # Check GPU memory usage
            # allocated = torch.cuda.memory_allocated()
            # cached = torch.cuda.memory_cached()
            # print(f"Allocated GPU memory: {allocated / 1024**3:.2f} GB")
            # print(f"Cached GPU memory: {cached / 1024**3:.2f} GB")

            train_loss, train_loss_1, train_loss_2, train_loss_3a, train_loss_3b = self._train()
            validation_loss, validation_loss_1, validation_loss_2, validation_loss_3a, validation_loss_3b  = self._validate()
           
                
            if(cycle == True and (epoch-start) % log_interval == 0):
                # self.delta_beta *= -1.
                self.beta = 1
                self.lossseq +=1
                
            if(cycle == False and epoch>1 and train_loss - self.train_loss_history[-1] > -.01 and train_loss - self.train_loss_history[-1] < 0):
                cycle = True
                start = epoch
                self.beta = 1
                self.lossseq +=1

            
            # self.beta = np.sin(epoch*8*np.pi/(2*log_interval))**2
            self.beta = min(1., max(0., self.beta+self.delta_beta))
            
            self.valid_loss_history.append(validation_loss)
            self.train_loss_history.append(train_loss)
            self.validation_loss_1_history.append(validation_loss_1)
            self.validation_loss_2_history.append(validation_loss_2)
            self.validation_loss_3a_history.append(validation_loss_3a)
            self.validation_loss_3b_history.append(validation_loss_3b)
            self.train_loss_1_history.append(train_loss_1)
            self.train_loss_2_history.append(train_loss_2)
            self.train_loss_3a_history.append(train_loss_3a)
            self.train_loss_3b_history.append(train_loss_3b)

            
            self.evidence_error = self._get_logz_from_samples().detach().cpu().numpy().std()
            self.evidence_mean = self._get_logz_from_samples().detach().cpu().numpy().mean()
            
            if (validation_loss < self.best['valid_loss']):# or (self.evidence_error/np.absolute(self.evidence_mean) < np.log(tolerance)):
                if validation_loss < self.best['valid_loss']:
                    count_patience      = 0
                self.best['epoch']       = epoch
                self.best['valid_loss']  = validation_loss
                self.best['train_loss']  = train_loss
                self.best['flow']        = copy.deepcopy(self.flow)
                
            count_patience += 1
            # print(epoch, train_loss,validation_loss,self.best['valid_loss'])
            
            
            if epoch == 1 or epoch % log_interval == 0:  
                self.logger.info('> Epoch {} - train loss {:.5f} - valid loss {:.5f} - best valid loss {:.5f}'.format(epoch,
                                                                                                                      train_loss,
                                                                                                                      validation_loss,
                                                                                                                      self.best['valid_loss']))
            
            if count_patience > patience:
                self.logger.info('Ran out of patience ({}) at epoch {}'.format(patience, epoch))
                break
                
            # if self.evidence_error/self.evidence_mean < np.log(tolerance):
            #     self.logger.info('Reached requested log_e (tolerance) ({}<{}) at epoch {}'.format(self.evidence_error/self.evidence_mean, np.log(tolerance), epoch))
            #     break

        end_training_time = time.time()
        self.logger.info('Completing training of neural network')

        self.flow.load_state_dict(self.best['flow'].state_dict())
        self.logger.info('Best epoch ({}): train loss {:.5f} - validation loss {:.5f}'.format(self.best['epoch'], self.best['train_loss'], self.best['valid_loss']))

        training_time     = end_training_time-start_training_time
        training_time_min = int(training_time/60)
        training_time_sec = training_time - training_time_min*60
        if training_time_min > 60:
            training_time_hour = int(training_time_min/60)
            training_time_min = training_time_min - training_time_hour*60
            self.logger.info('Total training time: {:.0f}hr {:.0f}min {:.3f}sec'.format(training_time_hour, training_time_min, training_time_sec))
        else:
            self.logger.info('Total training time: {:.0f}min {:.3f}sec'.format(training_time_min, training_time_sec))

        logz = self._get_logz_from_samples()
        self.logz = logz.detach().cpu().numpy()
        self.logger.info('Estimated logZ with ratio: {:.5f} ± {:.5f}'.format(logz.median(), logz.std()))
        
        self.logger.info('Saving output data')
        self._save_data()
        
        self.logger.info('L2 = {}'.format(self.l2))
        self.logger.info('Done.')

    def _load_data(self):
        self.flow.load_state_dict(torch.load(os.path.join(zflowpath, 'flow_state.pt')))
        self.flow.eval()
        
    def _save_data(self):
    
        self.logger.info('Output directory: {}'.format(self.outdir))
        
        # save flow state
        torch.save(self.flow.state_dict(), os.path.join(self.outdir, 'flow_state.pt'))
        
        # save logz samples
        np.savetxt(os.path.join(self.outdir, 'logz.txt'), self.logz)

        # save losses
        np.savetxt(os.path.join(self.outdir, 'losses.txt'), [self.train_loss_history, self.valid_loss_history, self.train_loss_1_history,
            self.validation_loss_1_history, self.train_loss_2_history, self.validation_loss_2_history, self.train_loss_3a_history,
                            self.validation_loss_3a_history , self.train_loss_3b_history, self.validation_loss_3b_history])

        
        fig = plt.figure(figsize=(8,7))
        # plt.plot(np.array(self.train_loss_history)[:,0], label='Training', ls = '-', c = 'k', lw=4)
        # plt.plot(np.array(self.train_loss_history)[:,1], label='Training', ls = '-', c = 'k', lw=4)
        # plt.plot(np.array(self.valid_loss_history)[:,0], label='Validation', ls = ':', c = 'k', lw=4)
        # plt.plot(np.array(self.valid_loss_history)[:,1], label='Validation', ls = ':', c = 'k', lw=4)
        plt.plot(np.array(self.train_loss_history), label='Training', ls = '-', c = 'k', lw=4)
        plt.plot(np.array(self.valid_loss_history), label='Validation', ls = ':', c = 'k', lw=4)
        p = plt.plot(self.train_loss_1_history, label=r'$\mathcal{L}_\mathrm{1}$', ls = '-')
        plt.plot(self.validation_loss_1_history, ls = ':', color = p[0].get_color())
        p = plt.plot(self.train_loss_2_history, label=r'$\mathcal{L}_\mathrm{2}$', ls = '-')
        plt.plot(self.validation_loss_2_history, ls = ':', color = p[0].get_color())
        p = plt.plot(self.train_loss_3a_history, label=r'$\mathcal{L}_\mathrm{3a}$', ls = '-')
        plt.plot(self.validation_loss_3a_history, ls = ':', color = p[0].get_color())
        p = plt.plot(self.train_loss_3b_history, label=r'$\mathcal{L}_\mathrm{3b}$', ls = '-')
        plt.plot(self.validation_loss_3b_history, ls = ':', color = p[0].get_color())
        # plt.axvline(self.best['epoch'], color='k', label='Best epoch', ls=':')
        # Get the current axis (ax1)
        ax1 = plt.gca()
        # ax2 = ax1.twinx()
        # ax2.plot(self.l2_hist, c = 'grey')
        # ax2.set_ylabel(r'$\alpha$')
        # Creating a second y-axis and plotting the second set of data
        
        ax1.set_ylabel(r'$\mathcal{L}$')
        ax1.set_xlabel('Epoch')
        # ax1.legend(loc='best')
        # ax2.legend(loc='best')
        # Combine legends for both axes into one
        lines, labels = ax1.get_legend_handles_labels()
        ax1.legend(lines , labels, loc='best')
        # plt.ylim(-15, 15)
        plt.savefig(os.path.join(self.outdir, 'Loss_history.png'), dpi=150, bbox_inches='tight')

        fig = plt.figure(figsize=(10,8))
        # l = np.array(self.train_loss_history[1:]) - np.array(self.train_loss_history[:-1])
        # l1 = np.array(self.train_loss_1_history[1:]) - np.array(self.train_loss_1_history[:-1])
        # l2 = np.array(self.train_loss_1_history[1:]) - np.array(self.train_loss_1_history[:-1])
        # l[l**2==(l1-l2)**2] = np.nan
        # plt.plot(l, label='Training', ls = '-', c = 'k', lw=.5)
        # # plt.plot(np.array(self.valid_loss_history[1:]) - np.array(self.valid_loss_history[:-1]), label='Validation', ls = '--', c = 'k', lw=2)
        p = plt.plot(np.array(self.train_loss_1_history[1:]) - np.array(self.train_loss_1_history[:-1]), label=r'$\mathcal{L}_\mathrm{1}$', ls = '--')
        # plt.plot(np.array(self.validation_loss_1_history[1:])-np.array(self.validation_loss_1_history[:-1]), ls = '--', color = p[0].get_color())
        p = plt.plot(np.array(self.train_loss_2_history[1:])-np.array(self.train_loss_2_history[:-1]), label=r'$\mathcal{L}_\mathrm{2}$', ls = '--')
        # plt.plot(np.array(self.validation_loss_2_history[1:])-np.array(self.validation_loss_2_history[:-1]), ls = '--', color = p[0].get_color())
        p = plt.plot(np.array(self.train_loss_3a_history[1:]) - np.array(self.train_loss_3a_history[:-1]), label=r'$\mathcal{L}_\mathrm{3a}$', ls = '--')
        # plt.plot(np.array(self.validation_loss_3a_history[1:])-np.array(self.validation_loss_3a_history[:-1]), ls = '--', color = p[0].get_color())
        p = plt.plot(np.array(self.train_loss_3b_history[1:])-np.array(self.train_loss_3b_history[:-1]), label=r'$\mathcal{L}_\mathrm{3b}$', ls = '--')
        # plt.plot(np.array(self.validation_loss_3b_history[1:])-np.array(self.validation_loss_3b_history[:-1]), ls = '--', color = p[0].get_color())
        # Get the current axis (ax1)
        ax1 = plt.gca()
        ax1.set_ylabel(r'$\delta$ Loss')
        ax1.set_xlabel('Epoch')
        # ax1.legend(loc='best')
        # ax2.legend(loc='best')
        # Combine legends for both axes into one
        lines, labels = ax1.get_legend_handles_labels()
        ax1.legend(lines, labels, loc='best')
        plt.savefig(os.path.join(self.outdir, 'grad_Loss_history.png'), dpi=150, bbox_inches='tight')

        
        def plot_histlnZ():
            logz = self._get_logz_from_samples()
            fig = plt.figure(figsize=(10,8))
            logz_npy = logz.detach().cpu().numpy()
            bins = np.arange(np.min(logz_npy), np.max(logz_npy), (np.max(logz_npy)-np.min(logz_npy))/101)
            pdf, _, _ = plt.hist(logz_npy, density=True, bins=bins, histtype = 'step', label = 'zflows')
            # print('Density integral ', np.sum(pdf)*(bins[1]-bins[0]))
            pdf_reweight, _, _ = plt.hist(logz_npy, density=True, bins=bins, histtype = 'step', label = 'zflows_reweight', weights = np.exp(self.log_prob.detach().cpu().numpy()))
            bins = (bins[1:] + bins[:-1])*.5
            #print(bins[(bins >= multinest_lnZ[0]-multinest_lnZ[1]) & (bins <= multinest_lnZ[0]+multinest_lnZ[1])])
                
            # Shade the regions of the zflows prediction
            
            plt.fill_between(bins, 0, pdf, where=((bins >= np.median(logz_npy)-np.std(logz_npy)) & (bins <= np.median(logz_npy)+np.std(logz_npy))), color='skyblue', alpha=0.8, label=r'Zflow: '+'{:.2f} ± {:.2f}'.format(np.median(logz_npy), np.std(logz_npy)))
            
            def weighted_mean_and_std(values, weights):
                weighted_mean = np.average(values, weights=weights)
                variance = np.average((values - weighted_mean)**2, weights=weights)
                weighted_std = np.sqrt(variance)
                return weighted_mean, weighted_std
            try:
                logz_mean_reweight, logz_std_reweight = weighted_mean_and_std(values=logz_npy, weights=np.exp(self.log_prob.detach().cpu().numpy()))
                plt.fill_between(bins, 0, pdf_reweight, where=((bins >= logz_mean_reweight - logz_std_reweight) & (bins <= logz_mean_reweight + logz_std_reweight)), color='orange', alpha=0.4, label=r'Zflow re-weighted: '+'{:.2f} ± {:.2f}'.format(logz_mean_reweight, logz_std_reweight))
            except ZeroDivisionError:
                self.logger.info('Z too small to compute reweighted distribution')
            plt.axvline(self.lnZ_GT, c='k', ls = '--', label= 'GT: logZ = {:.2f}'.format(self.lnZ_GT))
            
            plt.xlabel(r'log $Z$')
            plt.ylabel(r'p(log $Z$)')
            plt.yscale('log')
            plt.ylim(top = 1.1*max(np.max(pdf), np.max(pdf_reweight)), bottom= 1e-3*max(np.max(pdf), np.max(pdf_reweight)))
            #plt.ylim(top = 1e0, bottom= 1e-3)
            plt.xlim(left = -.1 + np.min(bins), right = .1 + np.max(bins))
            
            # Add a legend with handle and label
            handles, labels = plt.gca().get_legend_handles_labels()
            legend = plt.legend(handles=handles[2:], labels=labels[2:])
            # plt.tight_layout()
            plt.savefig(os.path.join(self.outdir, 'logz_hist.png'), dpi=150, bbox_inches='tight')
        plot_histlnZ()