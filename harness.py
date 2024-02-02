# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import pickle
from util.env import pretty_device_name
from util.time import *
from util.checkpoint_io import *
from data.loaders import ValidationLoader
from model.model import MLP,Prediction_Model
import torch.nn.functional as F
from pathlib import Path

import os
import logging
import time

logger = logging.getLogger(__name__)

class Harness(object):
    """ This class is the base class containing the basic functionalities of the code.
    It is meant to be implemented by either an evaluation model or a training model.
    There are functions that are always necessary such as the determination of the device.
    E.g., there is always a check, if the GPU is available. """

    def __init__(self, opt):
        """ The main init function. All sub init functions should take
        no further arguments than opt. """

        logger.info('Starting initialization')            
        self._init_device(opt)
        self._init_paths(opt)
        self._init_state(opt)
        self._init_training_loaders(opt) #Overwrite statistics during training
        self._init_validation_loaders(opt) #Pass on statistics from the training set
        self._init_model(opt)
        self._init_training(opt)     
        self._init_losses(opt)
        logger.info('Finished initialization')

    def _init_device(self, opt):
        """ Initialise the device. If pytorch finds a GPU, use a GPU, if not, work on CPU.
        Take care that all tensors are put to self.device and that a device is never hard-coded. """

        cpu = not torch.cuda.is_available()
        #cpu = cpu or opt.sys_cpu

        self.device = torch.device("cpu" if cpu else "cuda")
        logger.info('Using device: ' + pretty_device_name(self.device))

    def _init_paths(self, opt):
        """ Initialise the paths. Take care that all code is written relative to these paths.
        There is a data path, where all data should be located and a checkpoint path,
        where the trained models should be stored """
        self.log_path = os.path.join(Path.cwd(),'experiments',opt.experiment,opt.model_name)
        os.makedirs(self.log_path, exist_ok=True)
        logger.info('Writing model and logs to: ' + self.log_path)

    def _init_validation_loaders(self,opt):
        """Initialize validationa dataloader"""
        logger.info('Loading parameters from '+str(self.log_path))
        statistics = self.load_dictionary(os.path.join(self.log_path,'parameters.pickle'))
        val_loader = ValidationLoader(
                mode='val',
                dataset_name=opt.validation_dataset_name,
                dataset_path=opt.validation_dataset_path,
                config_file_path=opt.config_file_path,
                statistics=statistics,
                log_path=self.log_path,
                num_workers=opt.sys_num_workers,
                batch_size=opt.validation_batch_size
        )

        val_loader.generate_validation_loader()
        self.val_loader = val_loader.get_loader()
        self.num_input_features = val_loader.get_feature_length()

        logger.info('Using validation dataset: ' + opt.validation_dataset_name +
              ' with ' + str(self.num_input_features))

    def _init_model(self,opt):
        """Initialize model"""
        model=MLP(num_features=self.num_input_features)

        logger.info(model)
        logger.info("model: {}".format(model.__class__.__name__))
        if opt.model_load is not None:
            model, optimizer, scheduler, training_state = self.checkpoint_manager.load_model(
                model,
                model_path=opt.model_load,
                optimizer=None,
                scheduler=None,
                training=None,
                resume_epoch=None,
                training_state=None
            )
            self.model = model.double()    
        else:  
            self.model = model.double()
        logger.info("Total number of parameters: {} (num. trained {})".format(self.count_parameters(self.model),self.count_parameters(self.model, trainable=True)))
        self.model.to(self.device)        
        
    def count_parameters(self,m, trainable=False):
        """Computes the number of trainable and non trainable parameters"""
        if trainable:
            return sum(p.numel() for p in m.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in m.parameters())


    def _init_state(self, opt):
        self.checkpoint_manager = CheckPointManager(self.log_path, opt.experiment, opt.model_name,
                                                    self.device)
                                                    

    def _init_training(self, opt):
        """ Load options necessary for training """
        pass

    def _init_training_loaders(self, opt):
        """ Load the data necessary for training here """
        pass

    def _init_testing(self, opt):
        """ Load options necessary for training """
        pass

    def _init_test_loaders(self, opt):
        """ Load the data necessary for testing here """
        pass

    def _init_losses(self, opt):
        """ Initialise the losses here """
        self.criterion = nn.MSELoss()
        
    def _init_device(self, opt):
        """ Initialise the device. If pytorch finds a GPU, use a GPU, if not, work on CPU.
        Take care that all tensors are put to self.device and that a device is never hard-coded. """

        cpu = not torch.cuda.is_available()
        #cpu = cpu or opt.sys_cpu

        self.device = torch.device("cpu" if cpu else "cuda")
        logger.info('Using device: ' + pretty_device_name(self.device))

    def _set_eval(self):
        """ Set the model to eval mode """
        # at the moment only the model needs to be set to train/eval mode,
        # but later on one can think of more advanced stuff
        self.model.eval()

    def _set_train(self):
        """ Set the model to training mode """
        self.model.train()

    def _batch_to_device(self, input_dict):
        for key in input_dict.keys():
            input_dict[key] = input_dict[key].to(self.device)
        return input_dict

    def load_dictionary(self, path_dict):
        with open(path_dict, 'rb') as fp:
            return(pickle.load(fp)) 

    def run_test(self):
        """ This function runs the validation, which can be used in training after each epoch
         but also during evaluation. This ensures that validation and evaluation
         give the same results """

        """Bring model into the correct state"""
        self._set_eval()        
        now = time.time()
        loss=0.0

        """Run validation and collect metrics"""
        logger.info('Starting validation')
        length_loader = len(self.val_loader)
        print(length_loader)
        with torch.no_grad():
            for idx, input_dict in enumerate(self.val_loader):
                input_dict = self._batch_to_device(input_dict)
                in_tensor = input_dict['input_data'].double()
                if np.isnan(in_tensor.detach().cpu().numpy()).any():
                    print('input data has nan')
                ground_truth = input_dict['gt'].double()

                """for prediction""" 
                pred = self.model(in_tensor)
                #pred=pred.detach().cpu().numpy()
                #ground_truth=ground_truth.detach().cpu().numpy()
                if np.isnan(pred).any():
                    print('pred data has nan')
                
                loss+=torch.sqrt(self.criterion(ground_truth, pred))
        val_loss=loss/length_loader
        return val_loss

    def get_score(self, mode, preds,actual):        
        pass








