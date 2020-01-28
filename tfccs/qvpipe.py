import numpy as np
import tensorflow as tf
import sys
import os
import os.path as op
import json
import argparse
import logging
from tfccs.utils import execute, write_to_script, load_fextract_npz, mkdir, add_filter_args
from tfccs.constants import MIN_DIST2END, ALLOWED_STRANDS, ALLOWED_CIGARS

FORMATTER = op.basename(__file__) + ':%(levelname)s:'+'%(message)s'
logging.basicConfig(level=logging.DEBUG, format=FORMATTER)
log = logging.getLogger(__name__)

# Input Json file should contain four mandatory sections:
#  "Name", "Input", "Output", "TensorflowParameters",
# while "ValidationData" is optional.
# {
#    "Name": "NameOfThisModel",
#    "Input": {"FextractCsv": "",  "Ccs2GenomeBam": ""},
#    "Output": {"ModelDirectory": "", "BenchmarkDirectory": ""},
#    "TensorflowParameters": {},
#    "ValidationData": {"FextractCsv": ""},
# }
#
# And "TensorflowParameters" section must contain the following sections:
# {
#    "BatchSize: integer,
#    "Epochs": integer,
#    "EarlyStoppingPatience": integer,
#    "Layers": []
# }
# , where "Layers" is a list of objects, which can be empty, indicating no hidden layer (e.g.,
# "Layers": [] -> empty hidden layer), or specify hidden layers sequenentially.
# Each layer must contain the following sections:
# {
#    "LayerType": "dense",
#    "NodeSize": integer,
#    "Activation": string e.g., "relu" or "sigmoid"
#    "KernelInitializer": string e.g., "random_uniform",
#    "KernelRegularizer": string e.g., "l2",
#    "RegularizerWeight": float, e.g., 0.001
# }
# e.g., "Layers": [{"LayerType":"dense", "NodeSize": 64, "Activation": "relu"},
# {"LayerType":"dense", "NodeSize": 32, "Activation": "sigmoid"}]

NPROC = 16


class Layer(object):
    ALLOWED_ACTIVATIONS = ["relu", "sigmoid", "softmax"]
    ALLOWED_LAYER_TYPES = {"dense": tf.keras.layers.Dense}
    ALLOWED_KERNEL_INITIALIZERS = ["random_uniform"]
    ALLOWED_KERNEL_REGULARIZERS = {"l2": tf.keras.regularizers.l2}

    def __init__(self, layer_type, node_size, activation, kernel_initializer="random_uniform",
                 kernel_regularizer="l2", regularizer_weight=0):
        self.layer_type = layer_type.lower()
        self.node_size = int(node_size)
        self.activation = activation.lower()
        self.kernel_initializer = kernel_initializer.lower()
        self.kernel_regularizer = None
        if kernel_regularizer is not None and len(kernel_regularizer) > 0:
            self.kernel_regularizer = kernel_regularizer.lower()
        self.regularizer_weight = float(regularizer_weight)

        if self.layer_type not in self.ALLOWED_LAYER_TYPES.keys():
            raise ValueError(f"Unsupported layer type {layer_type}! Only support {self.ALLOWED_LAYER_TYPES}!")
        if self.activation not in self.ALLOWED_ACTIVATIONS:
            raise ValueError(f"Unsupported activation function {activation}! Only support {self.ALLOWED_ACTIVATIONS}!")
        if self.kernel_initializer not in self.ALLOWED_KERNEL_INITIALIZERS:
            raise ValueError(
                f"Unsupported kernel initializer {kernel_initializer}! Only support {self.ALLOWED_KERNEL_INITIALIZERS}!")
        if self.kernel_regularizer and self.kernel_regularizer not in self.ALLOWED_KERNEL_REGULARIZERS.keys():
            raise ValueError(
                f"Unsupported kernel regularizer {kernel_regularizer}! Only support {self.ALLOWED_KERNEL_REGULARIZERS}!")

    @classmethod
    def from_dict(cls, d):
        for key in ["LayerType", "NodeSize", "Activation"]:  # Mandatory
            if not key in d:
                raise ValueError(f"Could not find '{key}' from Layer param {d}!")

        layer_type = d["LayerType"]
        node_size = d["NodeSize"]
        activation = d["Activation"]

        # Optional
        kernel_initializer = "random_uniform"  # default
        kernel_regularizer = None
        regularizer_weight = 0

        if "KernelInitializer" in d:
            kernel_initializer = d["KernelInitializer"]

        if (("KernelRegularizer" in d and not "RegularizerWeight" in d) or
                ("KernelRegularizer" not in d and "RegularizerWeight" in d)):
            raise ValueError(f"KernelRegularizer and RegularizerWeight must both set or not set!")
        elif "KernelRegularizer" in d and "RegularizerWeight" in d:
            kernel_regularizer = d["KernelRegularizer"]
            regularizer_weight = d["RegularizerWeight"]
        return Layer(layer_type=layer_type, node_size=node_size, activation=activation,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer,
                     regularizer_weight=regularizer_weight)

    def to_dict(self):
        d = {"LayerType": self.layer_type, "NodeSize": self.node_size, "Activation": self.activation,
             "KernelInitializer": self.kernel_initializer}
        if (self.kernel_regularizer is None or self.regularizer_weight == 0):
            d["KernelRegularizer"] = ""
            d["RegularizerWeight"] = 0
        else:
            d["KernelRegularizer"] = self.kernel_regularizer
            d["RegularizerWeight"] = self.regularizer_weight
        return d

    def __repr__(self):
        return "Layer({}, {}, {})".format(self.layer_type, self.node_size, self.activation)

    def keras_layer(self, input_ncols, output_ncols=None, layer_name=None):
        keras_layer_func = self.ALLOWED_LAYER_TYPES[self.layer_type]
        if input_ncols <= 0:
            raise ValueError(repr(self) + f" invalid input column size {input_ncols}!")

        new_output_size = self.node_size
        override_output_ncols = (output_ncols is not None) and (output_ncols > 0)
        if override_output_ncols:
            if self.node_size > 0:
                raise ValueError(
                    repr(self) + " output node size is a positive integer, and output_ncols set to {self.node_size}, conflicting!")
            else:
                new_output_size = output_ncols

        if new_output_size <= 0:
            raise ValueError(repr(self) + f" invalid output column size {new_output_size}!")

        log.info(f"Creating layer: input columns={input_ncols}, output columns={new_output_size}")

        if self.regularizer_weight == 0 or self.kernel_regularizer is None:
            if layer_name is None:
                return keras_layer_func(new_output_size, kernel_initializer=self.kernel_initializer,
                                        activation=self.activation, input_shape=(input_ncols,))
            else:
                return keras_layer_func(new_output_size, kernel_initializer=self.kernel_initializer,
                                        activation=self.activation, input_shape=(input_ncols,), name=layer_name)
        else:
            keras_regularizer_func = self.ALLOWED_KERNEL_REGULARIZERS[self.kernel_regularizer](self.regularizer_weight)
            if layer_name is None:
                return keras_layer_func(new_output_size, kernel_initializer=self.kernel_initializer,
                                        kernel_regularizer=keras_regularizer_func,
                                        activation=self.activation, input_shape=(input_ncols,))
            else:
                return keras_layer_func(new_output_size, kernel_initializer=self.kernel_initializer,
                                        kernel_regularizer=keras_regularizer_func,
                                        activation=self.activation, input_shape=(input_ncols,), name=layer_name)


class ParamConfig(object):
    EARLY_STOP_MIN_DELTA = 0.0001
    EARLY_STOP_MONITOR = 'accuracy'

    def __init__(self, batch_size, epochs, early_stop_patience, layers, num_train_rows, optimizer):
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.early_stop_patience = int(early_stop_patience)
        self.layers = list(layers)
        self.num_train_rows = num_train_rows
        self.optimizer = optimizer
        self.assert_out_layer()

    def assert_out_layer(self):
        """Assert output layer must be {"LayerType": "dense", "NodeSize": -1, "Activation": "softmax"}!"""
        if len(self.layers) == 0:
            raise ValueError("Input model has empty layers!")
        out_layer = self.layers[-1]
        if out_layer.layer_type != "dense" or out_layer.node_size != -1 or out_layer.activation != "softmax":
            raise ValueError(
                """Expected output layer must be {"LayerType": "dense", "NodeSize": -1, "Activation": "softmax"}, while observed output layer is """ + repr(out_layer))

    @classmethod
    def from_dict(cls, d):
        for key in ["BatchSize", "Epochs", "EarlyStoppingPatience", "Layers", "NumTrainRows"]:
            if key not in d:
                raise ValueError(f"Could not find {key} in ParameterConfig {d}!")
        batch_size = d["BatchSize"]
        epochs = d["Epochs"]
        patience = d["EarlyStoppingPatience"]
        num_train_rows = d["NumTrainRows"]
        optimizer = d["Optimizer"]
        layers = [Layer.from_dict(layer) for layer in d["Layers"]]
        return ParamConfig(batch_size=batch_size, epochs=epochs,
                           early_stop_patience=patience, layers=layers,
                           num_train_rows=num_train_rows, optimizer=optimizer)

    def to_dict(self):
        d = {"BatchSize": self.batch_size, "Epochs": self.epochs,
             "EarlyStoppingPatience": self.early_stop_patience,
             "NumTrainRows": self.num_train_rows,
             "Optimizer": self.optimizer,
             "Layers": [layer.to_dict() for layer in self.layers]}
        return d

    def keras_layers(self, input_ncols, output_ncols):
        ret = []
        current_input_ncols = -1
        current_output_ncols = None
        current_layer_name = None
        for index, layer in enumerate(self.layers):
            if index == 0:  # Input layer
                current_input_ncols = input_ncols
            if index == len(self.layers) - 1:  # Output layer
                current_output_ncols = output_ncols
                current_layer_name = "out_layer"
            ret.append(layer.keras_layer(input_ncols=current_input_ncols,
                                         output_ncols=current_output_ncols,
                                         layer_name=current_layer_name))
            current_input_ncols = layer.node_size
        return ret

    def early_stop_callback_func(self):
        return tf.keras.callbacks.EarlyStopping(monitor=self.EARLY_STOP_MONITOR,
                                                min_delta=self.EARLY_STOP_MIN_DELTA,
                                                patience=self.early_stop_patience)


def generate_config_json(args):
    out_json = os.path.realpath(args.out_json)
    if not out_json.endswith('.json'):
        raise ValueError(f"Output file {out_json} must ends with .json!")

    name = op.basename(out_json).split('.')[0]
    default_batch_size = 32
    default_epochs = 100
    default_early_stop_patience = 15
    default_num_train_rows = 1000000
    default_layers = [Layer(layer_type="dense", node_size=-1, activation="softmax")]
    default_optimizer = 'adam'
    param_config = ParamConfig(batch_size=default_batch_size, epochs=default_epochs,
                               early_stop_patience=default_early_stop_patience, layers=default_layers,
                               num_train_rows=default_num_train_rows, optimizer=default_optimizer)
    config_obj = CcsQvConfig(name=name, in_fextract_csv="FIXME", in_ccs2genome_bam="FIXME", out_model_dir="FIXME",
                             out_benchmark_dir="FIXME", param_config=param_config, validation_fextract_csv="FIXME",
                             min_dist2end=args.min_dist2end, allowed_strands=args.allowed_strands,
                             allowed_cigars=args.allowed_cigars)
    config_obj.save_json(out_json)


class CcsQvConfig(object):

    def __init__(self, name, in_fextract_csv, in_ccs2genome_bam,
                 param_config, out_model_dir, out_benchmark_dir,
                 validation_fextract_csv=None, min_dist2end=MIN_DIST2END,
                 allowed_strands=ALLOWED_STRANDS, allowed_cigars=ALLOWED_CIGARS):
        self.name = name
        self.in_fextract_csv = in_fextract_csv
        self.in_ccs2genome_bam = in_ccs2genome_bam
        self.out_model_dir = out_model_dir
        self.out_benchmark_dir = out_benchmark_dir
        self.param_config = param_config
        self.validation_fextract_csv = validation_fextract_csv
        self.min_dist2end = int(min_dist2end)
        self.allowed_strands = allowed_strands
        self.allowed_cigars = allowed_cigars

    def mkdir(self):
        mkdir(self.out_model_dir)
        mkdir(self.out_benchmark_dir)

    def assert_input_exist(self):
        input_files = [self.in_fextract_csv, self.in_ccs2genome_bam]
        if self.validation_fextract_csv is not None:
            input_files.append(self.validation_fextract_csv)
        for filename in input_files:
            if not op.exists(filename):
                raise IOError(f"Input file {filename} does not exists!")

    @classmethod
    def from_file(cls, filename):
        d = json.load(open(filename, 'r'))
        for key in ["Name", "Input", "Output", "TensorflowParameters"]:
            if key not in d:
                raise ValueError(f"Could not find expected key {key} in file {filename}")
        name = d["Name"]

        def parse_input(d, filename):
            obj = d["Input"]
            if "FextractCsv" not in obj:
                raise ValueError(f"Could not find input FextractCsv in file {filename}")
            if "Ccs2GenomeBam" not in obj:
                raise ValueError(f"Could not find input Ccs2GenomeBam in file {filename}")
            return obj["FextractCsv"], obj["Ccs2GenomeBam"]

        def parse_output(d, filename):
            obj = d["Output"]
            if "ModelDirectory" not in obj:
                raise ValueError(f"Could not find input ModelDirectory in file {filename}")
            if "BenchmarkDirectory" not in obj:
                raise ValueError(f"Could not find input BenchmarkDirectory in file {filename}")
            return obj["ModelDirectory"], obj["BenchmarkDirectory"]

        in_fextract_csv, in_ccs2genome_bam = parse_input(d, filename)
        out_model_dir, out_benchmark_dir = parse_output(d, filename)
        param_config = ParamConfig.from_dict(d["TensorflowParameters"])
        validation_fextract_csv = None
        if "ValidationData" in d and "FextractCsv" in d["ValidationData"]:
            if d["ValidationData"]["FextractCsv"].strip():
                validation_fextract_csv = d["ValidationData"]["FextractCsv"]

        min_dist2end = MIN_DIST2END
        allowed_strands = ALLOWED_STRANDS
        allowed_cigars = ALLOWED_CIGARS
        if 'SAMPLING' in d:
            sampling_config = d['SAMPLING']
            if 'MIN_DIST2END' in sampling_config:
                min_dist2end = sampling_config['MIN_DIST2END']
            if 'ALLOWED_STRANDS' in sampling_config:
                allowed_strands = sampling_config['ALLOWED_STRANDS']
            if 'ALLOWED_CIGARS' in sampling_config:
                allowed_cigars = sampling_config['ALLOWED_CIGARS']
        return CcsQvConfig(name=name, in_fextract_csv=in_fextract_csv,
                           in_ccs2genome_bam=in_ccs2genome_bam,
                           param_config=param_config,
                           out_model_dir=out_model_dir,
                           out_benchmark_dir=out_benchmark_dir,
                           validation_fextract_csv=validation_fextract_csv,
                           min_dist2end=min_dist2end,
                           allowed_strands=allowed_strands,
                           allowed_cigars=allowed_cigars)

    def to_dict(self):
        d = {
            "Name": f"{self.name}",
            "Input": {"FextractCsv": f"{self.in_fextract_csv}",  "Ccs2GenomeBam": f"{self.in_ccs2genome_bam}"},
            "Output": {"ModelDirectory": f"{self.out_model_dir}", "BenchmarkDirectory": f"{self.out_benchmark_dir}"},
            "TensorflowParameters": self.param_config.to_dict(),
        }
        if self.validation_fextract_csv is not None:
            d["ValidationData"] = {"FextractCsv": f"{self.validation_fextract_csv}"}
        sampling_config = {'MIN_DIST2END': self.min_dist2end,
                           'ALLOWED_STRANDS': self.allowed_strands,
                           'ALLOWED_CIGARS': self.allowed_cigars}
        d['SAMPLING'] = sampling_config
        return d

    def save_json(self, out_json):
        with open(out_json, 'w') as writer:
            json.dump(self.to_dict(), writer, indent=4, sort_keys=True)

    @property
    def prev_train_script(self):
        return op.join(self.out_model_dir, 'prev_train.sh')

    @property
    def post_train_script(self):
        return op.join(self.out_model_dir, 'post_train.sh')

    @property
    def train_script(self):
        return op.join(self.out_model_dir, 'train.sh')

    @property
    def benchmark_script(self):
        return op.join(self.out_model_dir, 'benchmark.sh')

    @property
    def feature_stat_json(self):
        return op.join(self.out_model_dir, 'features.stat.json')

    @property
    def feature_order_json(self):
        return op.join(self.out_model_dir, 'features.order.json')

    @property
    def npz_prefix(self):
        return op.join(self.out_model_dir, 'out')

    @property
    def train_npz(self):
        return self.npz_prefix + '.train.npz'

    @property
    def sampling_base_map_prob_json(self):
        return self.npz_prefix + '.base_map_probability.json'

    @property
    def population_base_map_prob_json(self):
        return op.join(self.out_model_dir, 'population.base_map_probability.json')

    @property
    def base_map_prob_json(self):
        return op.join(self.out_model_dir, 'base_map_probability.json')

    @property
    def baseqv_csv(self):
        return op.join(self.out_model_dir, 'out.baseqv.csv')

    def create_prev_train_script(self):
        def gen_stat_cmd(in_fextract_csv, out_stat_json):
            return f'fextract2stat {in_fextract_csv} {out_stat_json} --min-dist2end {self.min_dist2end} --allowed-strands {self.allowed_strands} --allowed-cigars {self.allowed_cigars}'

        def gen_npz_cmd(in_fextract_csv, in_stat_json, out_prefix, out_order_json, num_train_rows):
            c0 = f'fextract2numpy {in_fextract_csv} {out_prefix} --stat-json {in_stat_json} --num-train-rows {num_train_rows}'
            c1 = f'mv {out_prefix}.features.order.json {out_order_json}'
            return c0 + '\n' + c1

        def qvtools_cmd(ccs2genome_bam, out_baseqv_csv):
            return f'qvtools {ccs2genome_bam} {out_baseqv_csv} -j {NPROC} --forward-only '

        def population_prob_cmd(in_baseqv_csv, out_population_base_map_prob_json):
            rscript_path = '/mnt/software/c/ccsqv/master/bin/R/base_map_prob_json_from_baseqv_csv.R'
            return f'Rscript --vanilla {rscript_path} -i {in_baseqv_csv} -o {out_population_base_map_prob_json}'

        def merge_base_map_prob_cmd(in_sampling_json, in_population_json, out_merged_json):
            return f'merge-base-map-prob {in_sampling_json} {in_population_json} {out_merged_json}'

        c0 = gen_stat_cmd(in_fextract_csv=self.in_fextract_csv,
                          out_stat_json=self.feature_stat_json)
        c1 = gen_npz_cmd(in_fextract_csv=self.in_fextract_csv,
                         in_stat_json=self.feature_stat_json,
                         out_prefix=self.npz_prefix,
                         out_order_json=self.feature_order_json,
                         num_train_rows=self.param_config.num_train_rows)
        c2 = qvtools_cmd(ccs2genome_bam=self.in_ccs2genome_bam,
                         out_baseqv_csv=self.baseqv_csv)
        c3 = population_prob_cmd(in_baseqv_csv=self.baseqv_csv,
                                 out_population_base_map_prob_json=self.population_base_map_prob_json)
        c4 = merge_base_map_prob_cmd(in_sampling_json=self.sampling_base_map_prob_json,
                                     in_population_json=self.population_base_map_prob_json,
                                     out_merged_json=self.base_map_prob_json)
        write_to_script([c0, c1, c2, c3, c4], self.prev_train_script)
        log.info(f"Written prev_train script: {self.prev_train_script}")
        return self.prev_train_script

    def create_prost_train_script(self):
        sr2ccs = "/home/UNIXHOME/yli/repo/ccsqv/tests/data/applyqv/one-read.sr2ccs.bam"
        ccs2genome = "/home/UNIXHOME/yli/repo/ccsqv/tests/data/applyqv/one-read.ccs2genome.bam"
        out = op.join(self.out_model_dir, 'tiny.bam')
        c0 = f'saved_model_cli show --dir {self.out_model_dir} --tag_set serve --signature_def serving_default'
        c1 = f'rm -f {out}'
        c2 = f'/mnt/software/c/ccsqv/master/bin/applyqv {sr2ccs} {ccs2genome} {out} -p {self.out_model_dir}'
        c3 = f'samtools view {out} | cut -f 1-4'
        write_to_script([c0, c1, c2, c3], self.post_train_script)
        log.info(f"Written post_train script: {self.post_train_script}")
        return self.post_train_script

    def create_benchmark_script(self):
        c0 = """. /mnt/software/Modules/current/init/bash
module load ccsqv/master
input=/pbi/dept/consensus/ccsqv/data/Mule/lambda/one_percent.lambda.arrowqv.ccs2genome.tsv
model={abs_model_dir}
name={name}
""".format(name=self.name, abs_model_dir=op.realpath(self.out_model_dir))
        c1 = "bash cromwell-ccsqv-apply.sh ${input} ${model} ${name} " + op.realpath(self.out_benchmark_dir)
        write_to_script([c0, c1], self.benchmark_script)
        return self.benchmark_script

    def train(self):
        from tfccs.train import train as train_func
        fextract_input, _, _, ccs2genome_cigars, _, _ = load_fextract_npz(self.train_npz)

        def create_and_compile_model_func(x_ncol, y_ncol):
            model = tf.keras.models.Sequential(self.param_config.keras_layers(
                input_ncols=x_ncol, output_ncols=y_ncol))
            model.compile(optimizer=self.param_config.optimizer,
                          loss='categorical_crossentropy', metrics=['accuracy'])
            return model

        model, evl = train_func(x_train=fextract_input, y_train=ccs2genome_cigars,
                                out_dir=self.out_model_dir, name=self.name,
                                batch_size=self.param_config.batch_size,
                                epochs=self.param_config.epochs,
                                create_and_compile_model_func=create_and_compile_model_func,
                                early_stop_callback=self.param_config.early_stop_callback_func())
        return model, evl


def run(args):
    in_json = os.path.realpath(args.in_json)
    if not op.exists(in_json):
        raise IOError(f"Input json {in_json} does not exist!")
    config_obj = CcsQvConfig.from_file(in_json)
    config_obj.assert_input_exist()
    out_model_dir = config_obj.out_model_dir
    log.info(f"Output directory: {out_model_dir}")
    if op.exists(out_model_dir):
        log.warning(f"Output model directory already exists! Overriding '{out_model_dir}'!")
    config_obj.mkdir()

    prev_train_sh = config_obj.create_prev_train_script()
    execute(f'bash {prev_train_sh}')

    config_obj.train()

    post_train_sh = config_obj.create_prost_train_script()
    execute(f'bash {post_train_sh}')

    benchmark_sh = config_obj.create_benchmark_script()
    log.info(f'To run benchmark:\n bash {benchmark_sh}\n')


def get_ccsqv_pipeline_parser():
    """Set up and return argument parser."""
    p = argparse.ArgumentParser("CCS Qv pipeline with two subcommands: `gen` and `run`.")
    subparsers = p.add_subparsers(help="Subcommand help")
    config_json_desc = "A json file which configs a CCS Qv model by specifying input fexctract.csv, ccs2genome.bam, Tensorflow hyperparameters and etc."

    gen_desc = "gen: generate a json which specifies default configurations for CCS Qv model."
    gen_subparser = subparsers.add_parser('gen', help=gen_desc)
    gen_subparser.add_argument('out_json', help=config_json_desc)
    add_filter_args(gen_subparser)
    gen_subparser.set_defaults(func=generate_config_json)

    run_desc = "run: train a CCS QV model using tensorflow and benchmark performance at ReadQv metrics."
    run_subparser = subparsers.add_parser('run', help=run_desc)
    run_subparser.add_argument("in_json", help=config_json_desc)
    run_subparser.set_defaults(func=run)
    return p


def main(args=sys.argv[1:]):
    """main"""
    args = get_ccsqv_pipeline_parser().parse_args(args)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main(args=sys.argv[1:]))
