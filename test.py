"""Evaluates the model"""

import os
import json
import torch
import logging
import argparse

import torch.nn.functional as F

from easydict import EasyDict
from termcolor import colored
from torchvision import transforms

import model.net as net
import model.data_loader as data_loader

from utils import utils
from utils.manager import Manager
from model.loss import compute_test_metrics, update_metrics

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--gif_name')
parser.add_argument('--restore_file',
                    default='best',
                    help="name of the file in --model_dir \
                     containing weights to load")


class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 64) + 1) * 64 - self.ht) % 64
        pad_wd = (((self.wd // 64) + 1) * 64 - self.wd) % 64
        if mode == 'sintel':
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


def evaluate(model, manager):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """
    # set model to evaluation mode
    model.eval()

    # val/test status initial
    for k, v in manager.test_status.items():
        manager.test_status[k].reset()

    # compute metrics over the dataset
    with torch.no_grad():
        for idx, loader in enumerate(manager.test_dataloader):

            eval_with_homo = True if idx == 0 else False

            flag = 'Clean' if idx == 0 else 'Final'

            for _, data_batch in enumerate(loader):

                transformer = transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255])
                # move to GPU if available
                data_batch = utils.tensor_gpu(data_batch)

                img1, img2 = data_batch['img1'], data_batch['img2']
                padder = InputPadder(img1.shape)
                img1, img2 = padder.pad(img1, img2)

                data_batch["imgs"] = torch.cat([transformer(img1), transformer(img2)], 1)

                # compute model output
                output_batch = model(data_batch)

                output_batch["flow_fw"][0] = padder.unpad(output_batch["flow_fw"][0])
                output_batch["homo_fw"][0] = padder.unpad(output_batch["homo_fw"][0])

                # compute all metrics on this batch and auto update to manager
                metrics = {}
                # compute metrics
                B = data_batch["img1"].size()[0]

                ret = compute_test_metrics(data_batch, output_batch, eval_with_homo)

                if data_batch["label"][0] == "RE":
                    update_metrics(ret, metrics, B, manager, "RE-{}".format(flag), eval_with_homo)
                elif data_batch["label"][0] == "Rain":
                    update_metrics(ret, metrics, B, manager, "RAIN-{}".format(flag), eval_with_homo)
                elif data_batch["label"][0] == "Dark":
                    update_metrics(ret, metrics, B, manager, "DARK-{}".format(flag), eval_with_homo)
                elif data_batch["label"][0] == "Fog":
                    update_metrics(ret, metrics, B, manager, "FOG-{}".format(flag), eval_with_homo)
                elif data_batch["label"][0] == "SNOW":
                    update_metrics(ret, metrics, B, manager, "SNOW-{}".format(flag), eval_with_homo)

        # print results for homography (on GHOF-Clean) and optical flow (on GHOF-Clean and GHOF-Final)
        utils.print_overall_test_metrics(manager)


if __name__ == '__main__':
    """
        Evaluate the model on the test set.
    """
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    with open(json_path) as f:
        params = EasyDict(json.load(f))

    # use GPU if available
    params.cuda = torch.cuda.is_available()  # use GPU is available

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)

    manager = Manager()
    manager.params = params
    manager.params.update(vars(args))

    manager.params.restore_file = args.restore_file
    manager.gif_name = args.gif_name

    # Get the logger
    logger = utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))
    manager.logger = logger

    # Create the input data pipeline
    logging.info("Creating the dataset...")

    # fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(['test'], manager)
    manager.test_dataloader = dataloaders['test']

    logging.info("- done.")

    # Define the model and optimizer
    if params.model_name == "UFlowSGFGyroHomo":
        model = net.UFlowSGFGyroHomo(params)
    else:
        raise NotImplementedError

    # Define the model
    if params.cuda:
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    manager.train_status['model'] = model

    manager.load_checkpoints()

    logging.info("Starting evaluation")

    evaluate(model, manager)
