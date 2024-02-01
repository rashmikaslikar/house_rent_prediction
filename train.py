# Python standard libraries
# Public libraries
import torch
import numpy as np 
import os
import sys
import logging
import time
import torch.nn.functional as F
# local imports
from harness import Harness
from arguments import TrainingArguments
from data.loaders import TrainingLoader
import pickle
import mlflow
import mlflow.sklearn

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("it_monitoring.train")

class Trainer(Harness):
    """ This class trains a model. It implements the init functions defined in Harness.
    Additionally, all training functions are defined."""
    def _init_training(self, opt):
        self.num_epochs = opt.training_num_epochs
        self.frequency = opt.checkpoint_frequency
        self.model_name = opt.model_name
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt.lr)
       
        self.eta = 1.0
        self.eps = 1e-6
        self.lr = opt.lr
        self.es_patience = opt.es_patience
        self.early_stopping = opt.early_stopping 
        #self.dataset=opt.dataset
        #self.validate=opt.validate

    def _init_logging(self, opt):
        """ Initialise the logging here, best use tensorboard """
        pass

    def _init_training_loaders(self, opt):
        # generate the correct loader
        train_loader = TrainingLoader(
                mode='train',
                dataset_name=opt.training_dataset_name,
                dataset_path=opt.training_dataset_path,
                config_file_path=opt.config_file_path,
                statistics=None,
                log_path=self.log_path,
                num_workers=opt.sys_num_workers,
                batch_size=opt.training_batch_size
            )
                                    
        train_loader.generate_training_loader()
        self.train_loader = train_loader.get_loader()
        self.statistics = train_loader.get_statistics()
        self.num_input_features = train_loader.get_feature_length()

        logger.info('Using training dataset: ' + opt.training_dataset_name +
              ' with inout feature dimension of ' + str(self.num_input_features))

    def _run_epoch(self, epoch):
        """ Run a single epoch of training."""
        # Bring model into the correct state
        self._set_train()
        logger.info(f'Begin training epoch {epoch}')
        logger.info(f"Optimizer LR: {self.optimizer.param_groups[0]['lr']}")
        total_loss=0.0
        length_loader=len(self.train_loader)

        for idx, input_dict in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            output_dict = dict()
            input_dict = self._batch_to_device(input_dict)
            in_tensor = input_dict['input_data'].double()
            ground_truth = input_dict['gt'].double()      
            prediction = self.model(in_tensor)
            loss = torch.sqrt(self.criterion(ground_truth, prediction))
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() #* in_tensor.size(0)
            if epoch == 0 and idx == 0:
                self.losses["train_forecast"].append(loss.item())
            #print("--- %s seconds ---" % (time.time() - start_time))
        epoch_loss = total_loss / length_loader
        logger.debug(f'input shape: {in_tensor.shape}')
        logger.info('Finished epoch ({} / {}) (Loss:{:.4f})'.format(epoch+1, self.num_epochs, epoch_loss))
        return epoch_loss
    
    def train(self):
        """ This function trains the model. It loops through a routine running a single epoch
        and evaluates the model after each epoch."""
        
        self.train_loss = []
        val_loss_list = []
        early_stop_win = 5    
        stop_improve_count = 0
        self.losses = {
            "train_forecast": [],
            "val_forecast": []
        }

        # Get the untrained validation scores
        # Execute validation routine only for mssql model
        #if self.validate:
        logger.info('Running validation on untrained network to get the untrained val loss')            
        val_loss = self.run_test()   
        logger.info('Finished validation on untrained network (Loss:{:.4f})'.format(val_loss))         
        min_score = val_loss
        val_loss_list.append(val_loss)

        initial_start = time.time()
        for epoch in range(self.num_epochs):
            epoch_start = time.time()
            forecast_epoch_loss = self._run_epoch(epoch)
            epoch_end = time.time()
            logger.info(f"Total epoch time: {(epoch_end-epoch_start)/60:.3f} minutes")
            self.losses["train_forecast"].append(forecast_epoch_loss)

            #if self.validate:
                # Validation routine after each epoch
            val_start = time.time()
            val_loss  = self.run_test()
            logger.info('Finished validation on epoch ({} / {}) (Loss:{:.4f})'.format(epoch+1, self.num_epochs, val_loss))    
            val_end = time.time()
            logger.info(f"Evaluation time: {(val_end-val_start)/60:.3f} minutes")
            
            val_loss_list.append(val_loss)
            if self.early_stopping: 
                if val_loss > min_score:
                    state_train = {'epoch': epoch+1} 
                    #self.checkpoint_manager.save_model(epoch+1, self.model, 'model.pth', self.optimizer, None, state_train)
                    min_score = val_loss
                    stop_improve_count = 0
                else:
                    stop_improve_count += 1 

                if stop_improve_count >= self.es_patience:
                    break 
                            
            state_train = {'epoch': epoch+1} 
            self.checkpoint_manager.save_model(epoch+1, self.model, 'model.pth', self.optimizer, None, state_train)
        
        end = time.time()
        logger.info(f"total training + validation time : {(end-initial_start)/60:.3f} minutes")
        
        with open(os.path.join(self.log_path, 'train_loss.txt'), "w") as f:
            for item in self.losses["train_forecast"]:
                f.write("%s\n" % item)
        
        #if self.validate:
        with open(os.path.join(self.log_path, 'auc.txt'), "w") as f:
            for item in val_loss_list:
                f.write("%s\n" % item)
            
    def dump_dictionary(self,dictionary):  
        logger.debug(f"log_path is {self.log_path}")
        with open(os.path.join(self.log_path, 'mean_std.pickle'), 'wb') as fp:
            pickle.dump(dictionary, fp, protocol=pickle.HIGHEST_PROTOCOL)
            

if __name__ == "__main__":
    """ The harness function is not meant to be called. 
    This function is for testing purposes solely."""
    options = TrainingArguments().parse()

    # Options for determinism as far as PyTorch supports it are set here
    if options.sys_best_effort_determinism:
        import numpy as np
        import random

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        random.seed(1)

    trainer = Trainer(options)
    trainer.train()