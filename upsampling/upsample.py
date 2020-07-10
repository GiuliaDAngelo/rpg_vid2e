import argparse
import os
# Must be set before importing torch.
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

from utils import Upsampler


class Flags:
  def __init__(self, input_dir, output_dir):
    self.input_dir = input_dir
    self.output_dir = output_dir
    self.device = 'cpu'

# def get_flags():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--input_dir", required=True, help='Path to input directory. See README.md for expected structure of the directory.')
#     parser.add_argument("--output_dir", required=True, help='Path to non-existing output directory. This script will generate the directory.')
#     parser.add_argument("--device", type=str, default="cpu", help='Device to be used (cpu, cuda:X)')
#     args = parser.parse_args()
#     return args


def main():
    # flags = get_flags()

    input_dir = '/home/giulia/workspace/disparity-attention/saliency-dataset/tree/interpolated_depth/original'
    output_dir = '/home/giulia/workspace/disparity-attention/saliency-dataset/tree/interpolated_depth/original/upsampled'
    flags = Flags(input_dir=input_dir, output_dir=output_dir)

    upsampler = Upsampler(
            input_dir=flags.input_dir,
            output_dir=flags.output_dir,
            device=flags.device)
    upsampler.upsample()


if __name__ == '__main__':
    main()
