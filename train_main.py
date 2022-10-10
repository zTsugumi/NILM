import os # nopep8
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # nopep8

import argparse
from remove_space import remove_space
from seq2point_train import Trainer
# Allows a model to be trained from the terminal.

cwd = os.getcwd()
training_directory = 'D:\\Workspace\\Work\\EVN\\Data\\REFIT\\washingmachine\\washingmachine_training_.csv'
validation_directory = 'D:\\Workspace\\Work\\EVN\\Data\\REFIT\\washingmachine\\washingmachine_validation_H18.csv'

parser = argparse.ArgumentParser(description="Train sequence-to-point learning for energy disaggregation. ")

parser.add_argument("--appliance_name", type=remove_space, default="washingmachine", help="The name of the appliance to train the network with. Default is kettle. Available are: kettle, washingmachine, washing machine, dishwasher, and washingmachine. ")
parser.add_argument("--batch_size", type=int, default="100", help="The batch size to use when training the network. Default is 1000. ")
parser.add_argument("--crop", type=int, default="10000", help="The number of rows of the dataset to take training data from. Default is 10000. ")
#parser.add_argument("--pruning_algorithm", type=remove_space, default="default", help="The pruning algorithm that the network will train with. Default is none. Available are: spp, entropic, threshold. ")
parser.add_argument("--network_type", type=remove_space, default="seq2point", help="The seq2point architecture to use. ")
parser.add_argument("--epochs", type=int, default="50", help="Number of epochs. Default is 10. ")
parser.add_argument("--input_window_length", type=int, default="599", help="Number of input data points to network. Default is 599.")
parser.add_argument("--validation_frequency", type=int, default="1", help="How often to validate model. Default is 1. ")
parser.add_argument("--training_directory", type=str, default=training_directory, help="The dir for training data. ")
parser.add_argument("--validation_directory", type=str, default=validation_directory, help="The dir for validation data. ")

arguments = parser.parse_args()

# Need to provide the trained model
save_model_dir = os.path.join(cwd, "saved_models/" + arguments.appliance_name + "_" + arguments.network_type + "_model.h5")

trainer = Trainer(arguments.appliance_name,
                  arguments.batch_size,
                  arguments.crop,
                  arguments.network_type,
                  arguments.training_directory,
                  arguments.validation_directory,
                  save_model_dir,
                  epochs = arguments.epochs,
                  input_window_length = arguments.input_window_length,
                  validation_frequency = arguments.validation_frequency)
trainer.train_model()

