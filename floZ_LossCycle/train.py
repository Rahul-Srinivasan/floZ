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

## Older slower non-vectorized version of pre-whitening
# def _pre_whitening(samples):
    
#     cov     = np.cov(samples.T)
#     w, v    = np.linalg.eig(cov)
    
#     # project samples
#     output  = np.zeros(samples.shape)
#     for i in range(len(w)): # run over dimensions
#         for j in range(len(samples)): # run over samples
#             output[j][i]  = (v[:,i]*samples[j]).sum()/np.sqrt(w[i])
#     if np.isinf(0.5*np.log(np.abs(np.prod(w)))):
#         return output, 0.5*np.sum(np.log(np.abs(w)))
    
#     return output, 0.5*np.log(np.abs(np.prod(w)))


def _pre_whitening(samples):
    # Compute covariance matrix of samples
    cov_coloured = np.cov(samples.T)
    
    # Eigenvalue decomposition
    w, v = np.linalg.eig(cov_coloured)
    
    # Project samples using vectorized operations
    output = (samples @ v) / np.sqrt(w)
    
    log_det = 0.5 * np.sum(np.log(np.abs(w)))
    
    # Handle edge case where log determinant might be infinite
    if np.isinf(log_det):
        log_det = 0.5 * np.log(np.abs(np.prod(w))), w, v

    return output, log_det, w, v


# Reconstruct the original (coloured) samples using the eigenvector (v_colour) and eigenvalues (w_colour)
def _re_colouring(whitened_samples, whitened_log_prob):
    v_colour = np.loadtxt(os.path.join(self.outdir, '..', 'eigenValVec.txt'))
    w_colour = v_colour[:,0]
    v_colour = v_colour[:,1:]
    means_colour = np.loadtxt(os.path.join(self.outdir, '..', 'sample_means.txt'))
    
    coloured_samples = (whitened_samples * np.sqrt(w_colour) @ v_colour.T) + means_colour
    
    # Compute the log of the Jacobian determinant for the transformation
    log_jacobian_det = np.sum(np.log(np.sqrt(w_colour)))
    
    # Transform the whitened log probability to coloured log probability
    coloured_log_prob = whitened_log_prob - log_jacobian_det
    
    return coloured_samples, coloured_log_prob

    

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
        
        
        # Initialize flow in CPU first
        self.flow           = Flow(ndim=self.ndim,
                                   transform=transform,
                                   base_dist=base_dist,
                                   transform_dict=transform_dict,
                                   device=torch.device('cpu'))
        
        # Check if multiple GPUs are available
        self.logger.info('Available GPUs: {}'.format(torch.cuda.device_count()))
        if torch.cuda.device_count() > 1:
            self.logger.info("Using {}".format(torch.cuda.device_count(), "GPUs!"))
            # Wrap the model in DataParallel
            self.flow = nn.DataParallel(self.flow)
            # Get the GPUs being used by DataParallel
            gpus_used = model.device_ids
            self.logger.info("GPUs being used: {}".format(gpus_used))
        if(self.device == torch.device('cpu')):
            self.logger.info("NOT using GPUs!")

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
        
        sample_means = np.mean(posterior_samples, axis = 0)
        np.savetxt(os.path.join(self.outdir, '..', 'sample_means.txt'), sample_means)
        self.logger.info('Saved sample means in {}'.format(os.path.join(self.outdir, '..', 'sample_means.txt')))
        posterior_samples -= sample_means

        # pre-whitening
        if pre_whitening:
        
            posterior_samples, log_j_white, w, v  = _pre_whitening(posterior_samples)
            log_prob                        = log_prob + log_j_white
            _new_cov                        = np.cov(posterior_samples.T)
            self.logger.info('Whitening samples with covariance error of {:.3g}'.format(np.abs(np.max(_new_cov-np.eye(self.ndim)))))
            np.savetxt(os.path.join(self.outdir, '..', 'eigenValVec.txt'), np.hstack((w[:, np.newaxis], v)))
            self.logger.info('Saved eigenvector and values in {}'.format(os.path.join(self.outdir, '..', 'eigenValVec.txt')))
        
        # store analyzed data
        self.posterior_samples  = torch.tensor(posterior_samples, dtype=torch.float32)#.to(self.device)
        self.log_prob           = torch.tensor(log_prob, dtype=torch.float32)#.to(self.device)

        # Move posterior samples and log_prob to GPU memory in batches of max size 2e5 due to GPU limit
        if  len(self.posterior_samples) <= 2e5:
            self.posterior_samples  = self.posterior_samples.to(self.device)
            self.log_prob  = self.log_prob.to(self.device)

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
        train_tensor    = torch.from_numpy(X_train.astype(np.float32))#.to(self.device)
        if  len(self.posterior_samples) <= 2e5:
            train_tensor = train_tensor.to(self.device)
        train_dataset   = torch.utils.data.TensorDataset(train_tensor)
        if(shuffle_batches):
            seed = int(time.time())
            np.random.seed(seed)
            torch.manual_seed(seed)

        # # Hyperparameters
        # num_workers = 4  # Adjust number of worker threads based on your CPU
        # prefetch_factor = 2  # Number of batches to preload
        
        # # Create DataLoader with optimizations
        # train_loader = torch.utils.data.DataLoader(
        #     train_dataset, 
        #     batch_size=n_train_batches, 
        #     shuffle=shuffle_batches, 
        #     num_workers=num_workers, 
        #     pin_memory=True if self.device == torch.device('cuda') else False,
        #     prefetch_factor=prefetch_factor
        # )

        train_loader    = torch.utils.data.DataLoader(train_dataset, batch_size=n_train_batches, shuffle=shuffle_batches)
        
        self.train_loader_data = []
        self.train_loader_prob = []
        for batch_idx, data in enumerate(train_loader):
            if isinstance(data, list):
                data = data[0]
            self.train_loader_prob.append(torch.t(data)[-1])
            self.train_loader_data.append(torch.t(torch.t(data)[:-1]))
        
        print('Training first batch: ', self.train_loader_prob[0].shape)
        print('Training last batch: ', self.train_loader_prob[-1].shape)
        
        # prepare data table: create validation loader
        valid_tensor    = torch.from_numpy(X_valid.astype(np.float32))#.to(self.device)
        if  len(self.posterior_samples) <= 2e5:
            valid_tensor = valid_tensor.to(self.device)
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
        
        print('Validation first batch: ', self.valid_loader_prob[0].shape)
        print('Validation last batch: ', self.valid_loader_prob[-1].shape)
        
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
        
        self.evidence_error     = np.inf
        self.best               = { 'valid_loss'  : float('inf'),
                                    'train_loss'  : float('inf'),
                                    'epoch'       : 0,
                                    'flow'        : copy.deepcopy(self.flow)
                                    }

    def _get_logz_from_samples(self):
        self.flow.eval()
        with torch.no_grad():
            if len(self.posterior_samples)<=2e5:
                return self.log_prob - self.flow.log_prob(self.posterior_samples)
        
            num_batches = 10
            batch_len = len(self.posterior_samples)//num_batches
            batch_idx = np.arange(0, len(self.posterior_samples), batch_len)
            # Ensure the last index spans the length
            if batch_idx[-1] != len(self.posterior_samples):
                batch_idx = np.append(batch_idx, len(self.posterior_samples))
    
            # print('posterior samples device: ', self.posterior_samples.device)
            flow_prob = np.zeros(len(self.posterior_samples))
            for i in range(len(batch_idx)-1):
                flow_prob[batch_idx[i] : batch_idx[i+1]] = self.flow.log_prob(self.posterior_samples[batch_idx[i] : batch_idx[i+1]].clone().detach().requires_grad_(True).to(self.device)).cpu().detach().numpy()
        return self.log_prob - flow_prob
        
    
    def _train(self):

        self.flow.train()
        train_loss = 0
        train_loss_1 = 0
        train_loss_2 = 0
        train_loss_3a = 0
        train_loss_3b = 0

        # l3_calc flag: Identifies if the ratio of evidence pairs need to be evaluated.
        l3_calc = False
        if (self.beta >0 and self.beta<1):
            if((self.lossseq)%4 == 2 or (self.lossseq+1)%4 ==2 
               or (self.lossseq)%4 == 3 or (self.lossseq+1)%4 ==3):
                l3_calc = True
        elif(self.beta==0):
            if((self.lossseq+1)%4 ==2 or (self.lossseq+1)%4 ==3):
                l3_calc = True
        else:
             if((self.lossseq)%4 == 2 or (self.lossseq)%4 == 3):
                l3_calc = True
        
        for batch_idx, (data,prob) in enumerate(zip(self.train_loader_data,  self.train_loader_prob)):
            if  data.device != self.device:
                data = data.to(self.device)
                prob = prob.to(self.device)
            self.optimizer.zero_grad()
            loss = 0
            
            flow_prob   = self.flow.log_prob(data)

            log_z = prob - flow_prob

            # log_meanz = (log_z-log_z.max()).exp().mean().log() + log_z.max()
            log_stdz  = (log_z-log_z.max()).exp().std().log()  + log_z.max()
            
            loss_1 = -flow_prob.mean()
            loss_2 = log_stdz

            train_loss_1 += loss_1.item()
            train_loss_2 += loss_2.item()
            
            # Compute loss3 if l3_calc is true
            if l3_calc:
                lnR = log_z[self.train_pairs[:,0]] - log_z[self.train_pairs[:,1]]
                lnR = lnR.abs()
                RbyRmax = (lnR-lnR.max()).exp()
                loss_3a = RbyRmax.mean().log() + lnR.max()
                loss_3b = RbyRmax.std().log() + lnR.max()
                train_loss_3a += loss_3a.item()
                train_loss_3b += loss_3b.item()
            else:
                loss_3a = 0.#np.nan
                loss_3b = 0.#np.nan
                train_loss_3a = 0.#np.nan
                train_loss_3b = 0.#np.nan

            # Define sequence of the loss schedule
            loss_arr = [loss_1, loss_2, loss_3a, loss_3b]
            loss += self.beta*loss_arr[(self.lossseq)%len(loss_arr)] + (1-self.beta)*loss_arr[(self.lossseq+1)%len(loss_arr)]
           
            train_loss += loss.item()
            loss.backward()
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
        l3_calc = False
        if (self.beta >0 and self.beta<1):
            if((self.lossseq)%4 == 2 or (self.lossseq+1)%4 ==2 
               or (self.lossseq)%4 == 3 or (self.lossseq+1)%4 ==3):
                l3_calc = True
        elif(self.beta==0):
            if((self.lossseq+1)%4 ==2 or (self.lossseq+1)%4 ==3):
                l3_calc = True
        else:
             if((self.lossseq)%4 == 2 or (self.lossseq)%4 == 3):
                l3_calc = True
            
        for batch_idx, (data,prob) in enumerate(zip(self.valid_loader_data,  self.valid_loader_prob)):
            if  data.device != self.device:
                data = data.to(self.device)
                prob = prob.to(self.device)
            with torch.no_grad():
                flow_prob   = self.flow.log_prob(data)
                log_z = prob - flow_prob
                
                # log_meanz = (log_z-log_z.max()).exp().mean().log() + log_z.max()
                log_stdz  = (log_z-log_z.max()).exp().std().log()  + log_z.max()               
                
                loss_1 = -flow_prob.mean()
                loss_2 = log_stdz

                val_loss_1 += loss_1.item()
                val_loss_2 += loss_2.item()
                
                # Compute loss3 if l3_calc is True
                if l3_calc:
                    lnR = log_z[self.validation_pairs[:,0]] - log_z[self.validation_pairs[:,1]]
                    lnR = lnR.abs()
                    RbyRmax = (lnR-lnR.max()).exp()
                    loss_3b = RbyRmax.std().log() + lnR.max()
                    loss_3a = (RbyRmax.mean().log() + lnR.max())
                    
                    val_loss_3a += loss_3a.item()
                    val_loss_3b += loss_3b.item()
                else:
                    loss_3a = 0.#np.nan
                    loss_3b = 0.#np.nan
                    val_loss_3a = 0.#np.nan
                    val_loss_3b = 0.#np.nan
                
                # Define sequence of the loss schedule
                loss_arr = [loss_1, loss_2, loss_3a, loss_3b]
                loss = self.beta*loss_arr[(self.lossseq)%len(loss_arr)] + (1-self.beta)*loss_arr[(self.lossseq+1)%len(loss_arr)]
                      
                val_loss += loss.item()
                
        return val_loss / len(self.valid_loader_data), val_loss_1 / len(self.valid_loader_data), val_loss_2 / len(self.valid_loader_data), val_loss_3a / len(self.valid_loader_data), val_loss_3b / len(self.valid_loader_data)
        
    def train(self,
              max_num_epoch      = 5000,
              patience           = 100,
              tolerance          = 0.,
              log_interval       = 100,
              l2                 = 0,
              flag_plot          = True):

        # define flow
        self.logger.info('Output directory: {}'.format(self.outdir))
        
        self.logger.info('Starting training of neural network ...')
        count_patience      = 0
        start_training_time = time.time()
        self.logger.info('Total # batches: {}'.format(len(self.train_loader_data)))
        
        self.beta = 0.
        self.delta_beta = -6./log_interval
        self.lossseq = -1
        
        self.flag_plot = flag_plot
        
        
        self.logger.info('Training starts')

        # cycle flag: start the scheduler from epoch 1. Set to True by default
        cycle = True
        # Epoch after which the scheduler starts. Set to 0 by default
        cycle_start = 0
        
        for epoch in range(1,max_num_epoch+1):

            if epoch % (4*log_interval) == 1:
                self.logger.info('Saving current best state dictionary')
                torch.save(self.flow.state_dict(), os.path.join(self.outdir, 'flow_state.pt'))
                
                self.logger.info('Creating pairs')
                A = np.arange(0,len(self.train_loader_prob[-1]), 1)
                A1 = np.random.choice(A, min(int(1e4), len(A)), replace = False)
                pairs = np.array(list(combinations(A1, 2)))
                self.train_pairs = pairs
                
                B = np.arange(0,len(self.valid_loader_data[-1]), 1)
                B1 = np.random.choice(B, min(int(1e4*len(B)/len(A)), len(B)), replace = False)
                pairs = np.array(list(combinations(B1, 2)))
                self.validation_pairs = pairs
                self.logger.info('Pairs created')
                self.logger.info('Train pair shape {}'.format(self.train_pairs.shape))
                self.logger.info('Validation pair shape {}'.format(self.validation_pairs.shape))

            
            train_loss, train_loss_1, train_loss_2, train_loss_3a, train_loss_3b = self._train()
            validation_loss, validation_loss_1, validation_loss_2, validation_loss_3a, validation_loss_3b  = self._validate()
           
                
            if(cycle == True and (epoch-cycle_start) % log_interval == 0):
                self.beta = 1
                self.lossseq +=1
                
            if(cycle == False and epoch>1 and train_loss - self.train_loss_history[-1] > -.01 and train_loss - self.train_loss_history[-1] < 0):
                cycle = True
                cycle_start = epoch
                self.beta = 1
                self.lossseq +=1

            # specify weighting between consequtive losses. Default: continous linearly relation during transition, and flat afterwards.
            self.beta = min(1., max(0., self.beta+self.delta_beta))
            # self.beta = np.sin(epoch*8*np.pi/(2*log_interval))**2
            
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

            # Reset patience counter if validation loss decreases.
            if (validation_loss < self.best['valid_loss']):
                if validation_loss < self.best['valid_loss']:
                    count_patience      = 0
                self.best['epoch']       = epoch
                self.best['valid_loss']  = validation_loss
                self.best['train_loss']  = train_loss
                self.best['flow']        = copy.deepcopy(self.flow)
                
            count_patience += 1
                
            if epoch == 1 or epoch % log_interval == 0:  
                if(epoch==1):
                    # Check GPU memory usage
                    allocated = torch.cuda.memory_allocated()
                    # cached = torch.cuda.memory_cached()
                    reserved = torch.cuda.memory_reserved()
                    self.logger.info(f"GPU memory: Allocated {allocated / 1024**3:.2f} GB, Reserved {reserved / 1024**3:.2f} GB")

                self.logger.info('> Epoch {} - train loss {:.5f} - valid loss {:.5f} - best valid loss {:.5f}'.format(epoch,
                                                                                                                      train_loss,
                                                                                                                      validation_loss,
                                                                                                                      self.best['valid_loss']))
            # Check if patience has exceeded.
            if count_patience > patience:
                self.logger.info('Ran out of patience ({}) at epoch {}'.format(patience, epoch))
                break

            # Check if tollerance reached
            if self.evidence_error/self.evidence_mean < np.log(tolerance):
                self.logger.info('Reached requested log_e (tolerance) ({}<{}) at epoch {}'.format(self.evidence_error/self.evidence_mean, np.log(tolerance), epoch))
                break

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

        # Compute the log_z from flow. Note: Does NOT include post-processing step of chosing the 1sigma B ellipsoid.
        logz = self._get_logz_from_samples()
        self.logz = logz.detach().cpu().numpy()
        self.logger.info('Estimated logZ with ratio: {:.5f} ± {:.5f}'.format(logz.median(), logz.std()))

        # Save data, model
        self.logger.info('Saving output data')
        self._save_data()

        # Plot loss evolution and Z histogram if flag_plot = True
        if self.flag_plot:
            self._plot_LossEvolution()
            self._plot_histlnZ()

        # End program
        self.logger.info('Done.')


    # Function to load floZ state
    def _load_data(self):
        self.flow.load_state_dict(torch.load(os.path.join(zflowpath, 'flow_state.pt')))
        self.flow.eval()

    
    # Function to save floZ state and plot loss evolution.
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

    
    # Plotting function for loss evolution
    def _plot_LossEvolution(self, saveflag = True):
        
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
        ax1.set_ylabel(r'$\mathcal{L}$')
        ax1.set_xlabel('Epoch')
        lines, labels = ax1.get_legend_handles_labels()
        ax1.legend(lines , labels, loc='best')
        plt.savefig(os.path.join(self.outdir, 'Loss_history.png'), dpi=150, bbox_inches='tight')

        fig = plt.figure(figsize=(10,8))
        # plt.plot(np.array(self.valid_loss_history[1:]) - np.array(self.valid_loss_history[:-1]), label='Validation', ls = '--', c = 'k', lw=2)
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
        lines, labels = ax1.get_legend_handles_labels()
        ax1.legend(lines, labels, loc='best')
        plt.savefig(os.path.join(self.outdir, 'grad_Loss_history.png'), dpi=150, bbox_inches='tight')

    
    # Plotting function for histogram of Z, without post-processing B ellipsoid step.
    def _plot_histlnZ(self):
        logz = self._get_logz_from_samples()
        fig = plt.figure(figsize=(10,8))
        logz_npy = logz.detach().cpu().numpy()
        bins = np.arange(np.min(logz_npy), np.max(logz_npy), (np.max(logz_npy)-np.min(logz_npy))/101)
        pdf, _, _ = plt.hist(logz_npy, density=True, bins=bins, histtype = 'step', label = 'zflows')
        # self.logger.info('Density integral ', np.sum(pdf)*(bins[1]-bins[0]))
        pdf_reweight, _, _ = plt.hist(logz_npy, density=True, bins=bins, histtype = 'step', label = 'zflows_reweight', weights = np.exp(self.log_prob.detach().cpu().numpy()))
        bins = (bins[1:] + bins[:-1])*.5
        #self.logger.info(bins[(bins >= multinest_lnZ[0]-multinest_lnZ[1]) & (bins <= multinest_lnZ[0]+multinest_lnZ[1])])
            
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