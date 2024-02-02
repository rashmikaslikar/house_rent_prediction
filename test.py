# Python standard libraries

# Public libraries
import torch
torch.cuda.empty_cache()
import os
import logging
import sys

# local imports
from harness import Harness
from arguments import TestArguments
from data.loaders import TestLoader

# We need to setup root logger before importing any fairseq libraries.
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)

logger = logging.getLogger("it_monitoring.test")

class Test(Harness):
    """ This class evaluates a model, all evaluation functions are defined here."""
    def __init__(self, opt):
        logger.info('Starting initialization')
        logger.info(os.getcwd())
        self._init_device(opt)
        self._init_paths(opt)
        self._init_state(opt)
        self._init_test_loaders(opt) #Overwrite statistics during training
        self._init_model(opt) #When the model is loaded, the statistics will also be loaded
        self._init_losses(opt)
        logger.info('Finished initialization')

    def _init_test_loaders(self,opt):
        logger.info('Loading parameters from '+str(self.log_path))
        logger.info('Loading parameters from '+str(self.log_path))
        statistics = self.load_dictionary(os.path.join(self.log_path,'parameters.pickle'))

        test_loader = TestLoader(
                mode='val',
                dataset_name=opt.test_dataset_name,
                dataset_path=opt.test_dataset_path,
                config_file_path=opt.config_file_path,
                statistics=statistics,
                log_path=self.log_path,
                num_workers=opt.sys_num_workers,
                batch_size=opt.test_batch_size
            )

        test_loader.generate_test_loader()
        self.val_loader = test_loader.get_loader()
        self.num_input_features = test_loader.get_feature_length()

        logger.info('Using test dataset: ' + opt.test_dataset_name +
              ' with ' + str(self.num_input_features))
    
    def predict(self):
        """ This function evaluates the model."""
        loss  = self.run_test()
        logger.info('Finished testing (Loss:{:.4f})'.format(loss))    
            

if __name__ == "__main__":
    """ The harness function is not meant to be called.
    This function is for testing purposes solely."""
    options = TestArguments().parse()

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

    evaluator = Test(options)
    evaluator.predict()  
