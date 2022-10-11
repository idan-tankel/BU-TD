import argparse

def Get_basilnes_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--ewc_lambda', default = 1e14, type = float, help = 'The ewc strength')
    parser.add_argument('--si_lambda', default=0.0, type = float, help='The ewc strength')

    return parser.parse_args()


