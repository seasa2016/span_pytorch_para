import os
from datetime import datetime

def add_test_args(parser):
    parser.add_argument(
            "--test-data",
            required=True,
            help="path for the testing data"
            )

    parser.add_argument(
            "--test-elmo",
            required=True,
            help="path for the elmo file of testing data"
            )

    parser.add_argument(
            "--result-path",
            required=True,
            help="path to output data"
            )

    parser.add_argument(
            "--save-model",
            required=True,
            help="path to output data"
            )
    parser.add_argument(
            "--batchsize",
            default=8,
            help="batch size for input data"
            )


    return parser




def add_default_args(parser):
    parser.add_argument(
           "--seed",
           default=39,
           type=int
           )
    parser.add_argument(
            "--device",
            action='store_true'
            )
    parser.add_argument(
           "-g",
           "--gpu-id",
           default=-1,
           type=int
           )
    parser.add_argument(
           "--train", default=False,
            action='store_true'
           )
    parser.add_argument(
           "--dev", default=False,
            action='store_true'
           )
    parser.add_argument(
           "--test", default=False,
            action='store_true'
           )

    parser.add_argument(
           "--save-dir", required=True,
           type=str
           )

    return parser


def add_model_arch_args(parser):

    ######################
    # Hierarchical LSTMs #
    ######################
    parser.add_argument(
            "--lstm-ac",
            action="store_true",
            )
    parser.add_argument(
            "--lstm-shell",
            action="store_true",
            )
    parser.add_argument(
            "--lstm-ac-shell",
            action="store_true",
            )
    parser.add_argument(
            "--lstm-type",
            action="store_true",
            )

    parser.add_argument(
            "--bow_feat",
            action="store_true",
            )

    ###########
    # encoder #
    ###########
    parser.add_argument("--reps-type",
                        type=str,
                        default="contextualized")

    ############
    # decoder #
    ############
    parser.add_argument("--decoder",
                        type=str,
                        default="proposed"
                        )

    ###################
    # baseline models #
    ###################
    parser.add_argument(
            "--baseline-heuristic",
            action="store_true",
            help="baseline_heuristic"
            )

    ##############
    # dimensions #
    ##############
    parser.add_argument(
            "-ed",
            "--eDim",
            default=300,
            type=int
            )
    parser.add_argument(
            "-hd",
            "--hDim",
            default=256,
            type=int
            )

    ###########
    # dropout #
    ###########
    parser.add_argument(
            "-d",
            "--dropout",
            default=0.5,
            type=float
       )
    parser.add_argument(
            "-dl",
            "--dropout-lstm",
            default=0.1,
            type=float
       )
    parser.add_argument(
            "-de",
            "--dropout-embedding",
            default=0.1,
            type=float
       )

    return parser


def add_optim_args(parser):

    #############
    # optimizer #
    #############
    parser.add_argument(
           "--optimizer",
           default="Adam"
           )
    parser.add_argument(
           "--lr",
           default=0.001,
           type=float
           )

    ###########################
    # loss interpolation rate #
    ###########################
    parser.add_argument(
            "--ac-type-alpha",
            default=0,
            type=float,
            help="the rate of loss interpolation (ac type prediction)"
    )
    parser.add_argument(
            "--link-type-alpha",
            default=0,
            type=float,
            help="the rate of loss interpolation (link type prediction)"
    )
    return parser


def add_trainer_args(parser):
    #############
    # iteration #
    #############
    parser.add_argument(
           "--epoch",
           default=32,
           type=int
           )
    parser.add_argument(
            "--batchsize",
            default=16,
            type=int
            )
    parser.add_argument(
            "--max_n_spans_para",
            default=128,
            type=int
            )

    return parser


def add_embed_args(parser):
    ########
    # ELMo #
    ########
    parser.add_argument(
            "--use-elmo",
            type=int,
            default=0
            )
    parser.add_argument(
            "--elmo-path",
            default=""
            )

    parser.add_argument(
            "--elmo-layers",
            choices=["1", "2", "3", "avg", "weighted"],
            default="avg"
            )

    parser.add_argument(
            "--elmo-task-gamma",
            action="store_true"
            )

    return parser


def add_dataset_args(parser):
    parser.add_argument(
            "--data-path",
            default="",
            type=str
            )
    parser.add_argument(
            "--pred-result",
            default="./pred",
            type=str
            )

    return parser


def add_log_args(parser):
    parser.add_argument(
            "-o",
            "--out",
           )
    parser.add_argument(
           "--dir-prefix",
           default="{}".format(datetime.today().strftime('%Y%m%d-%H%M%S')),
           )
    parser.add_argument(
           "--log-output",
           action='store_true'
           )
    return parser


def post_process_args_info(args):

    ###########################################################
    # considering the args, modifying the path information... #
    ###########################################################

    args.vocab_path = os.path.join(os.path.dirname(__file__),
                                   "../../work/{}4ELMo.tsv.vocab_t3_tab".format(args.dataset))

    if args.elmo_path:
        args.elmo_path = args.elmo_path
    else:
        args.elmo_path = os.path.join(os.path.dirname(__file__),
                                      "../../work/{}4ELMo.hdf5".format(args.dataset))

    args.glove_path = "{}/glove.6B.{}d".format(args.glove_dir,
                                               args.eDim)

    return args
