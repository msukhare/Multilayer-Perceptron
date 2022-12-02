import os
import sys
import argparse

from easyML import MLP,\
                    scaling_features,\
                    split_data
from utils import read_data

def main(args):
    try:
        X, Y, labels = read_data(args.data_path, train=True)
    except Exception as error:
        sys.exit('Error: ' + str(error))
    X, params_to_save = scaling_features(X, None, args.type_of_features_scaling)
    classificator = MLP(args.config_file_path,\
                        args.show_model,\
                        args.type_of_initialization,\
                        args.epochs,\
                        args.batch_size,\
                        args.learning_rate,\
                        args.validation_fraction,\
                        args.l2,\
                        args.lambda_value,\
                        args.early_stopping,\
                        args.n_epochs_no_change,\
                        args.tol,\
                        args.accuracy,\
                        args.precision,\
                        args.recall,\
                        args.f1_score)
    classificator.fit(X, Y)
    try:
        classificator.fit(X, Y)
    except Exception as error:
        sys.exit('Error:' + str(error)) 
    classificator.save_weights(args.file_where_store_weights, params_to_save, labels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path',\
                        nargs='?',\
                        type=str,\
                        help="""correspond to path of csv file""")
    parser.add_argument('config_file_path',\
                        nargs='?',\
                        type=str,\
                        help="""correspond to path of cfg file which contains architecture of MLP""")
    parser.add_argument('--show_model',\
                        dest='show_model',\
                        action='store_true',
                        help="""if pass as params will show complete architecture of MLP""")
    parser.add_argument('--type_of_initialization',\
                        nargs='?',\
                        type=str,\
                        default="xavier",\
                        const="xavier",\
                        choices=['xavier'],\
                        help="""correspond to technic use for weights initialization.
                                By default xavier""")
    parser.add_argument('--file_where_store_weights',\
                        nargs='?',\
                        type=str,\
                        help="""correspond to path where store weights after training and
                                informations about pipeline""")
    parser.add_argument('--type_of_features_scaling',\
                        nargs='?',\
                        type=str,\
                        default="standardization",\
                        const="standardization",\
                        choices=['standardization', 'rescaling', 'normalization'],\
                        help="""correspond to technic use for features scaling.
                                By default standardization""")
    parser.add_argument('--validation_fraction',\
                        nargs='?',\
                        type=float,\
                        default=0.10,\
                        const=0.10,\
                        help="""correspond to percentage data use during training as val set in gradient descent.
                                By default 0.10 percentage of data""")
    parser.add_argument('--epochs',\
                        nargs='?',\
                        type=int,\
                        default=100,\
                        const=100,\
                        help="""correspond to numbers of epochs to do during training.
                                By default 100""")
    parser.add_argument('--batch_size',\
                        nargs='?',\
                        type=int,\
                        default=None,\
                        const=None,\
                        help="""correspond to numbers of sample to use for one iteration.
                                By default None all samples are used during one iteration""")
    parser.add_argument('--learning_rate',\
                        nargs='?',\
                        type=float,\
                        default=0.1,\
                        const=0.1,\
                        help="""correspond to learning rate used during training.
                                By default 0.1""")
    parser.add_argument('--early_stopping',\
                        dest='early_stopping',\
                        action='store_true',
                        help="""if pass as params will do early stopping on val set, base on tol and
                                n_epochs_no_change in gradient descent""")
    parser.add_argument('--n_epochs_no_change',\
                        nargs='?',\
                        type=int,\
                        default=5,\
                        const=5,\
                        help="""correspond to numbers of epochs wait until cost function don't change.
                                Only used in gradient descent and if --early_stoping is set at True.
                                By default 5 epochs""")
    parser.add_argument('--tol',\
                        nargs='?',\
                        type=float,\
                        default=1e-3,\
                        const=1e-3,\
                        help="""correspond to stopping criteron in early stopping.
                            Only used in gradient descent and if --early_stopping is set at True.
                            By default 1e-3""")
    parser.add_argument('--l2',\
                        dest='l2',\
                        action='store_true',
                        help="""if pass as params will do a logistic Ridge regression
                                by default logitic regression""")
    parser.add_argument("--lambda_value",\
                        nargs='?',\
                        type=float,\
                        default=0.01,\
                        const=0.01,\
                        help="""correspond to value to use if l2 regularization is use.
                                By default 0.01""")
    parser.add_argument('--accuracy',\
                        dest='accuracy',\
                        action='store_true',
                        help="""if pass as params will compute accuracy on validation set
                                validate must be pass as params to show accuracy""")
    parser.add_argument('--precision',\
                        dest='precision',\
                        action='store_true',
                        help="""if pass as params will compute precision on validation set
                                validate must be pass as params to show precision""")
    parser.add_argument('--recall',\
                        dest='recall',\
                        action='store_true',
                        help="""if pass as params will compute recall on validation set
                                validate must be pass as params to show recall""")
    parser.add_argument('--f1_score',\
                        dest='f1_score',\
                        action='store_true',
                        help="""if pass as params will compute f1_score on vaslidation set
                                validate must be pass as params to show f1_score""")
    parsed_args = parser.parse_args()
    if parsed_args.data_path is None:
        sys.exit("Error: missing name of CSV data to use")
    if os.path.exists(parsed_args.data_path) is False:
        sys.exit("Error: %s doesn't exists" %parsed_args.data_path)
    if os.path.isfile(parsed_args.data_path) is False:
        sys.exit("Error: %s must be a file" %parsed_args.data_path)
    if parsed_args.config_file_path is None:
        sys.exit("Error: missing name of CSV data to use")
    if os.path.exists(parsed_args.config_file_path) is False:
        sys.exit("Error: %s doesn't exists" %parsed_args.config_file_path)
    if os.path.isfile(parsed_args.config_file_path) is False:
        sys.exit("Error: %s must be a file" %parsed_args.config_file_path)
    main(parsed_args)