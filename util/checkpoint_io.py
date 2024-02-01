import os
import torch


# TODO: debug this class, after the training is actually executable

class CheckPointManager(object):
    def __init__(self, check_path, experiment_class, model_name, device):
        self.check_path = check_path
        self.experiment_class = experiment_class
        self.model_name = model_name
        self.device = device

        self.save_path_base = os.path.join(self.check_path,
                                      self.experiment_class,
                                      self.model_name,
                                      'checkpoints'
                                      )
        if not os.path.exists(self.save_path_base):
            os.makedirs(self.save_path_base)

    def load_model(self, model, optimizer=None, scheduler=None, training=None, model_path=None,
                   resume_epoch=None, training_state=False):
        """Load a saved model at a certain path into the current model.
        Optionally the whole training state can be loaded"""

        # explicitly pass a path to the checkpoint directory
        if model_path is not None:
            load_path = os.path.join(self.check_path,self.experiment_class,self.model_name,model_path)
            print(f'model path:{load_path}')
            if os.path.isdir(load_path) is not True:
                raise Exception('The path you specified does not exist, please specify a valid checkpoint directory')

        # just resume the training at a certain epoch
        elif resume_epoch is not None:
            load_path = os.path.join(self.check_path,
                                     self.experiment_class,
                                     self.model_name,
                                     'checkpoints',
                                     f'epoch_{resume_epoch}')
            if os.path.isdir(load_path) is not True:
                raise Exception('The path you specified does not exist, please supply a valid epoch')
        else:
            raise Exception('No checkpoint could be loaded')

        # either load the complete checkpoint if training_state is True or only the model file
        if training_state:
            return self._load_model(load_path, model, optimizer, scheduler, training)
        else:
            model, _, _, _ = self._load_model(load_path, model, optimizer=None, scheduler=None, training=None)
            return model, optimizer, scheduler, training

    def _load_model(self, path, model, optimizer=None, scheduler=None, training=None):
        path_model = os.path.join(path, 'model.pth')
        if os.path.isfile(path_model) is not True:
            raise Exception('The directory does not contain a valid model file')
        state_dict_model = torch.load(path_model, map_location=self.device)
        model.load_state_dict(state_dict_model)

        if optimizer is not None:
            path_optimizer = os.path.join(path, 'optimizer.pth')
            if os.path.isfile(path_optimizer) is not True:
                raise Exception('The directory does not contain a valid optimizer file')
            state_dict_optimizer = torch.load(path_optimizer, map_location=self.device)
            optimizer.load_state_dict(state_dict_optimizer)

        if scheduler is not None:
            path_scheduler = os.path.join(path, 'scheduler.pth')
            if os.path.isfile(path_scheduler) is not True:
                raise Exception('The directory does not contain a valid scheduler file')
            state_dict_scheduler = torch.load(path_scheduler, map_location=self.device)
            scheduler.load_state_dict(state_dict_scheduler)

        if training is not None:
            path_training = os.path.join(path, 'training.pth')
            if os.path.isfile(path_training) is not True:
                raise Exception('The directory does not contain a valid training file')
            state_dict_training = torch.load(path_training, map_location=self.device)
            training.load_state_dict(state_dict_training)

        return model, optimizer, scheduler, training

    def save_model(self, epoch, model,name, optimizer=None, scheduler=None, training=None):
        # set the save path
        save_path_base = os.path.join(self.save_path_base,
                                      f'epoch_{epoch}')
        if not os.path.exists(save_path_base):
            os.mkdir(save_path_base)
        # save the model state
        save_path_model = os.path.join(save_path_base,name)
        model_state_dict = model.state_dict()
        torch.save(model_state_dict,save_path_model) 

        # save the optimizer state
        if optimizer is not None:
            save_path_optimizer = os.path.join(save_path_base, 'optimizer.pth')
            state_dict_optimizer = optimizer.state_dict()
            torch.save(state_dict_optimizer,save_path_optimizer)

        # save the scheduler state
        if scheduler is not None:
            save_path_scheduler = os.path.join(save_path_base, 'scheduler.pth')
            state_dict_scheduler = scheduler.state_dict()
            torch.save(state_dict_scheduler,save_path_scheduler)

        # save the training state such as number of epochs, etc...
        if training is not None:
            save_path_training = os.path.join(save_path_base, 'training.pth')
            #state_dict_training = training.state_dict()
            torch.save(training,save_path_training)

