from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import argparse


class ArgumentsBase(object):
    """ This class contains all arguments necessary for training and evaluation of
    IT Monitoring models"""
    DESCRIPTION = 'IT Monitoring Arguments'
    

    def __init__(self):
        self.ap = ArgumentParser(
            description=self.DESCRIPTION,
            formatter_class=ArgumentDefaultsHelpFormatter
        )
        
    def str2bool(self,v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")


    def _init_system(self):
        self.ap.add_argument('-sys-cpu', default=False,action='store_true',
                             help='Disable GPU acceleration')
        self.ap.add_argument('-sys-num-workers', type=int, default=1,
                             help='Number of worker processes to spawn per DataLoader')
        self.ap.add_argument('-sys-best-effort-determinism', default=True, action='store_true',
                             help='Try and make some parts of the training/validation deterministic')
        self.ap.add_argument('-random-seed', type = int, default=5,
                             help='random seed',)
        self.ap.add_argument('-experiment',type = str, default='Prediction',
                             help='experiment directory')
        self.ap.add_argument('-model-name', type = str, default='mlp_task2_2',
                             help='model name', )
        self.ap.add_argument('-config-file-path', type=str, default='config.yaml',
                             help='path to the dataset to be used for evaluation')

    def _init_model(self): 
        self.ap.add_argument('-model-load', type=str, default=None)
    #'C:\\Users\\rashm\\Zoi\\experiments\\Prediction\\mlp\\checkpoints\\epoch_1'
    def _init_validation(self):
        self.ap.add_argument('-validation-dataset-name', type=str, default='validation_dataset',
                             help='Name of the validation loader')        
        self.ap.add_argument('-validation-dataset-path', type=str, default='immo_val.csv',
                             help='path to the dataset to be used for evaluation')
        self.ap.add_argument('-validation-batch-size', type=int, default=64,
                             help='Batch size used during validation')
        

    def _init_training(self):
        self.ap.add_argument('-training-num-epochs', type=int, default=10,
                             help='Number of epochs to run the training for')
        self.ap.add_argument('-lr',type=float, default=0.001,
                             help='learning rate')
        self.ap.add_argument('-training-dataset-name', type=str, default='training_dataset',
                             help='Name of the training loader')
        self.ap.add_argument('-training-dataset-path', type=str, default='immo_train.csv',nargs='?',
                             help='path to the dataset to be used for training')
        self.ap.add_argument('-training-batch-size', type=int, default=128,
                             help='Batch size used during training')
        self.ap.add_argument('-checkpoint-frequency', type=int, default=1,
                             help='No of epochs after which checkpoint should be saved')
        self.ap.add_argument('-initialize-lrs', default=False, action='store_true',
                             help='Switch on learning rate scheduling')
        self.ap.add_argument('-early-stopping', default=False, action='store_true',
                             help='Switch on early stopping')
        self.ap.add_argument('-lrs-patience', type=int, default=1,
                             help='how many epochs to wait before updating the lr')
        self.ap.add_argument('-es-patience', type=int, default=5,
                             help='how many epochs to wait before stopping the training')
        self.ap.add_argument('-min-learning-rate', type=float, default=1e-6,
                             help='least learning rate value to reduce to while updating')
        self.ap.add_argument('-factor', type=float, default=0.5,
                             help='factor by which the learning rate should be updated')


    def _init_testing(self):
        self.ap.add_argument('-test-dataset-name', type=str, default='test_dataset',
                             help='Name of the validation loader')        
        self.ap.add_argument('-test-dataset-path', type=str, default='immo_test.csv',
                             help='path to the dataset to be used for testing')
        self.ap.add_argument('-test-batch-size', type=int, default=1,
                             help='Batch size used during validation')
        
    def _init_eval(self):
        self.ap.add_argument('-model-path', type=str, default='model/',
                             help='path to model')
        self.ap.add_argument('-infer-ts', default=False, action='store_true',
                             help='use live model on torchserve for plotting inference')

    def _parse(self):
        return self.ap.parse_args()

class ValidationArguments(ArgumentsBase):
    DESCRIPTION = 'IT monitoring validation arguments'

    def __init__(self):
        super().__init__()

        self._init_system()
        self._init_model()
        self._init_validation()

    def parse(self):
        opt = self._parse()
        return opt


class TrainingArguments(ArgumentsBase):
    DESCRIPTION = 'IT monitoring training arguments'

    def __init__(self):
        super().__init__()

        self._init_system()
        self._init_model()
        self._init_validation()
        self._init_training()

    def parse(self):
        opt = self._parse()
        return opt

class TestArguments(ArgumentsBase):
    DESCRIPTION = 'IT monitoring validation arguments'

    def __init__(self):
        super().__init__()

        self._init_system()
        self._init_model()
        self._init_testing()

    def parse(self):
        opt = self._parse()
        return opt


class EvalArguments(ArgumentsBase):
    DESCRIPTION = 'IT monitoring validation arguments'

    def __init__(self):
        super().__init__()

        self._init_system()
        self._init_model()
        self._init_testing()
        self._init_eval()

    def parse(self):
        opt = self._parse()
        return opt