import argparse
import os

advance_parser = argparse.ArgumentParser(description='Process advance parameter.')
advance_parser.add_argument('--root_path', type=str, default=os.path.dirname(__file__), help='Root path of main.py.')
advance_parser.add_argument('--debug', type=bool, default=False, help='Whether start in debug environment of cuda.')
advance_parser.add_argument('--device', type=str, default="cuda:0", help='Train on which device.')
ADVANCED_OPTIONS = advance_parser.parse_known_args()[0]

data_parser = argparse.ArgumentParser(description='Process dataset parameter.')
data_parser.add_argument('--dataset_name', type=str, default='genia', help='Selection of dataset.')
data_parser.add_argument('--concat', action="store_true", help='Whether concatenate the train and development dataset.')
data_parser.add_argument('--mini_size', type=int, default=None, help='How large the mini dataset will be.')
data_parser.add_argument('--max_length', type=int, default=None, help='Maximum length of sentence in the corpus.')
DATA_OPTIONS = data_parser.parse_known_args()[0]

procedure_parser = argparse.ArgumentParser(description='Process procedure parameter.')
procedure_parser.add_argument('--checkpoint_name', type=str, default='mini', help='Extra name of checkpoint.')
procedure_parser.add_argument('--evaluate', type=str, action='append', default=[], help='Which dataset to evaluate on.')
procedure_parser.add_argument('--load', type=bool, default=False, help='Whether load the parameters of model.')
procedure_parser.add_argument('--save', type=bool, default=True, help='Whether save the best parameters.')
procedure_parser.add_argument('--batch_size', type=int, default=4, help='Batch size per iteration.')
procedure_parser.add_argument('--epochs', type=int, default=5, help='How many epochs to run.')
PROCEDURE_OPTIONS = procedure_parser.parse_known_args()[0]

optimize_parser = argparse.ArgumentParser(description='Process procedure parameter.')
optimize_parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate.')
optimize_parser.add_argument('--max_grad_norm', type=float, default=1e0, help='The maximum norm of the per-sample gradients.')
optimize_parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay rate of optimizer.')
optimize_parser.add_argument('--lr_warmup', type=float, default=0.1, help='Warmup rate for scheduler.')
OPTIMIZE_OPTIONS = optimize_parser.parse_known_args()[0]

model_parser = argparse.ArgumentParser(description='Process model hyper-parameter.')
model_parser.add_argument('--pretrain_select', type=str, default="dmis-lab/biobert-base-cased-v1.2",
                          help='Selection of pretrain model.')
model_parser.add_argument('--model_max_length', type=int, default=64, help='Maximum length of sentence inputted to the model.')
model_parser.add_argument('--pos_dim', type=int, default=None, help='Dimension of part-of-speech embedding.')
model_parser.add_argument('--char_dim', type=int, default=None, help='Dimension of char embedding.')
model_parser.add_argument('--word2vec_select', type=str, default=None, help='Select pretrain word2vec embeddings.')
model_parser.add_argument('--use_backward', type=bool, default=True, help='Whether use the backward block in proposer.')
model_parser.add_argument('--kernels_size', type=int, default=[2, 3], action='append', help='Convolution size of proposer.')
model_parser.add_argument('--use_category_embed', type=bool, default=True, help='Whether use the category embedding.')
model_parser.add_argument('--use_logits_iter', type=bool, default=True, help='Whether use the iterative logits.')
model_parser.add_argument('--use_locations_iter', type=bool, default=True, help='Whether use the iterative locations.')
model_parser.add_argument('--use_spatial', type=bool, default=True, help='Whether use the spatial guidance.')
model_parser.add_argument('--num_heads', type=int, default=8, help='Number of heads of attention.')
model_parser.add_argument('--num_layers', type=int, default=3, help='Layers of regressor.')
model_parser.add_argument('--dim_feedforward', type=int, default=2048, help='Feed-forward dimension for linear block.')
model_parser.add_argument('--dropout', type=float, default=0.1, help='The general model dropout.')
MODEL_OPTIONS = model_parser.parse_known_args()[0]
