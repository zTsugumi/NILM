import os # nopep8
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # nopep8

import argparse
from remove_space import remove_space
from seq2point_test import Tester

# Allows a model to be tested from the terminal.

# You need to input your test data directory
test_directory = "D:\\Workspace\\Work\\EVN\\Data\\REFIT\\microwave\\microwave_test_H4.csv"

parser = argparse.ArgumentParser(description="Train a pruned neural network for energy disaggregation. ")

parser.add_argument("--appliance_name", type=remove_space, default="microwave", help="The name of the appliance to perform disaggregation with. Default is microwave. Available are: microwave, fridge, dishwasher, microwave. ")
parser.add_argument("--crop", type=int, default="100000", help="The number of rows of the dataset to take training data from. Default is 10000. ")
parser.add_argument("--algorithm", type=remove_space, default="seq2point", help="The pruning algorithm of the model to test. Default is none. ")
parser.add_argument("--network_type", type=remove_space, default="", help="The seq2point architecture to use. Only use if you do not want to use the standard architecture. Available are: default, dropout, reduced, and reduced_dropout. ")
parser.add_argument("--input_window_length", type=int, default="599", help="Number of input data points to network. Default is 599. ")
parser.add_argument("--test_directory", type=str, default=test_directory, help="The dir for training data. ")

arguments = parser.parse_args()

# You need to provide the trained model
saved_model_dir = os.path.join(os.getcwd(), "saved_models", arguments.appliance_name + "_" + arguments.algorithm + "_model.h5")

# The logs including results will be recorded to this log file
log_file_dir = os.path.join(os.getcwd(), "saved_models", arguments.appliance_name + "_" + arguments.algorithm + "_" + arguments.network_type + ".log")

tester = Tester(arguments.appliance_name, arguments.algorithm, arguments.crop,
                arguments.network_type, arguments.test_directory,
                saved_model_dir, log_file_dir, arguments.input_window_length)
tester.test_model()

