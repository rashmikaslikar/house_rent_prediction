# Public libraries
from torch.utils.data import DataLoader

# Local imports
from data.basedataset import Dataset_Prediction

VALID_TRAINING_DATASETS = {
                           'training_dataset': Dataset_Prediction
                          }

VALID_VALIDATION_DATASETS = {
                             'validation_dataset': Dataset_Prediction
                            }

VALID_TEST_DATASETS = {
                       'test_dataset': Dataset_Prediction
                      }

VALID_DATASETS = {**VALID_TRAINING_DATASETS,
                  **VALID_VALIDATION_DATASETS,
                  **VALID_TEST_DATASETS}

class BaseLoader(object):
    """ This class is used to generate a valid validation loader.
    The get_loader function returns the final validation loader. """

    def __init__(self,
                mode,
                dataset_name,
                dataset_path,
                statistics,
                log_path,
                config_file_path,
                num_workers,
                batch_size,
                 ):
        # generate the dataset

        if dataset_name not in VALID_DATASETS.keys():
            raise Exception('Please specify a valid dataset name')
        self.dataset_name = dataset_name
        self.dataset = VALID_DATASETS[self.dataset_name](
                mode=mode,
                dataset_path=dataset_path,
                config_file_path=config_file_path,
                statistics=statistics,
                log_path=log_path
                                
        )

        # Loading arguments
        self.num_workers = num_workers
        self.batch_size = batch_size

        # The loader is not yet generated. Use generate_loader function to
        self.loader_generated = False
        self.loader = None

    def get_statistics(self):
        """ Get the normalization parameters"""
        return self.dataset.return_statistics()

    def get_loader(self):
        """ Return the final loader """

        if self.loader_generated is not True:
            raise Exception('The loader has not been created yet.')
        return self.loader
    
    def get_feature_length(self):
        """ Get the number of input features for a certain dataset"""
        return self.dataset.return_feature_length()


class TrainingLoader(BaseLoader):
    def __init__(self, *args, **kwargs):
        super(TrainingLoader, self).__init__(*args, **kwargs)

    def generate_training_loader(self):
        """ This function fills the loader. In case there are computations
        necessary on the argument, the loader is not directly generated in
        the __init__ function, thereby several datasets or loaders can be combined later on."""

        if self.dataset_name not in VALID_TRAINING_DATASETS.keys():
            raise Exception('This dataset cannot be used for validation')

        # The loader is generated
        self.loader = DataLoader(dataset=self.dataset,
                                 batch_size=self.batch_size,
                                 shuffle=True,
                                 num_workers=self.num_workers,
                                 pin_memory=True,
                                 drop_last=True
                                )
        # Change the flag to True, indicating that the loader has been generated.
        self.loader_generated = True

class ValidationLoader(BaseLoader):
    def __init__(self, *args, **kwargs):
        super(ValidationLoader, self).__init__(*args, **kwargs)

    def generate_validation_loader(self):
        """ This function fills the loader. In case there are computations
        necessary on the argument, the loader is not directly generated in
        the __init__ function, thereby several datasets or loaders can be combined later on."""

        if self.dataset_name not in VALID_VALIDATION_DATASETS.keys():
            raise Exception('This dataset cannot be used for validation')

        # The loader is generated
        self.loader = DataLoader(dataset=self.dataset,
                                 batch_size=self.batch_size,
                                 shuffle=False,
                                 num_workers=self.num_workers,
                                 pin_memory=True,
                                 drop_last=False
                                 )
        # Change the flag to True, indicating that the loader has been generated.
        self.loader_generated = True


class TestLoader(BaseLoader):
    def __init__(self, *args, **kwargs):
        super(TestLoader, self).__init__(*args, **kwargs)

    def generate_test_loader(self):
        """ This function fills the loader. In case there are computations
        necessary on the argument, the loader is not directly generated in
        the __init__ function, thereby several datasets or loaders can be combined later on."""

        if self.dataset_name not in VALID_TEST_DATASETS.keys():
            raise Exception('This dataset cannot be used for validation')

        # The loader is generated
        self.loader = DataLoader(dataset=self.dataset,
                                 batch_size=self.batch_size,
                                 shuffle=False,
                                 num_workers=self.num_workers,
                                 pin_memory=True,
                                 drop_last=False
                                 )
        # Change the flag to True, indicating that the loader has been generated.
        self.loader_generated = True
