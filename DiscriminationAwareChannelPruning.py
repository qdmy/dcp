# coding=utf-8
import typing
import numpy as np
import argparse
import os
import yaml
from time import strftime, localtime
from dcp.auxnet.option import Option as AuxOption
from dcp.auxnet.main import Experiment as AuxExp
from dcp.channel_selection.option import Option as CSOption
from dcp.channel_selection.main import Experiment as CSExp
from dcp.finetune.option import Option as FTOption
from dcp.finetune.main import Experiment as FTExp

class DCP():
    '''
    模型压缩算法
    '''
    # 说明模型压缩算法压缩的网络
    __id2net__ = {0: "resnet50", 1:"resnet152"}

    def __init__(self, config_path, net_type):
        '''
        Arg
        ---
            input: dcp config file, 包括auxnet, channel_selection, finetune所需使用的config文件的路径
                type: str
            net_type: 0 or 1, decide which network to compress
        Return
        ---
            DCP() instance
        '''
        # config
        f = open(config_path)
        self.cfg = yaml.load(f)

        for c in ['conf', 'save_path', 'pretrained']:
            for p in ['auxnet', 'channel_selection', 'finetune']:
                self.cfg[p][c] = self.cfg[p][c].replace('*', DCP.__id2net__[net_type])
        final_name = ''
        if net_type == 0:
            final_name = '04'
        elif net_type == 1:
            final_name = '10'
        self.cfg['finetune']['pretrained'] = self.cfg['finetune']['pretrained'].replace('+', final_name)
        print(self.cfg)
        
        # auxnet
        self.auxOption = AuxOption(self.cfg['auxnet']['conf'])
        self.auxOption.save_path = self.cfg['auxnet']['save_path']
        self.auxOption.data_path = self.cfg['auxnet']['data_path']
        self.auxOption.pretrained = self.cfg['auxnet']['pretrained']
        self.auxOption.n_epochs = self.cfg['auxnet']['n_epochs']
        self.auxOption.manualSeed = int(strftime('%Y%m%d',localtime()))
        self.auxOption.experiment_id = strftime('%Y%m%d%H%M%S',localtime())
        self.auxOption.resume = self.cfg['auxnet']['resume'] if self.cfg['auxnet']['resume'] != '' else None 
        self.auxExp = AuxExp(self.auxOption)


    def __call__(self):
        '''
        Return
        ---
            压缩前后的模型文件路径及其在公交车数据集上的准确率都会在log信息中打印
        '''

        # # auxnet
        # self.auxExp.run()

        # channel_selection
        # csOption = CSOption(self.cfg['channel_selection']['conf'])
        # csOption.pretrained = os.path.join(self.auxOption.save_path, self.cfg['channel_selection']['pretrained'])
        # csOption.save_path = self.cfg['channel_selection']['save_path']
        # csOption.data_path = self.cfg['channel_selection']['data_path']
        # csOption.experiment_id = strftime('%Y%m%d%H%M%S',localtime())
        # csOption.pruning_rate = self.cfg['channel_selection']['pruning_rate']
        # rates = csOption.pruning_rate.split(',')
        # csOption.pruning_rate = [1-float(i) for i in rates]
        # csOption.resume = self.cfg['channel_selection']['resume'] if self.cfg['channel_selection']['resume'] != '' else None
        # csExp = CSExp(csOption)
        # # csExp.channel_selection_for_network()

        # finetune
        ftOption = FTOption(self.cfg['finetune']['conf_path'])
        ftOption.save_path = self.cfg['finetune']['save_path']
        ftOption.data_path = self.cfg['finetune']['data_path']
        # ftOption.pretrained = os.path.join(csOption.save_path, self.cfg['finetune']['pretrained'])
        ftOption.n_epochs = self.cfg['finetune']['n_epochs']
        ftOption.manualSeed = int(strftime('%Y%m%d',localtime()))
        ftOption.experiment_id = strftime('%Y%m%d%H%M%S',localtime())
        ftOption.resume = self.cfg['finetune']['resume'] if self.cfg['finetune']['resume'] != '' else None
        ftExp = FTExp(ftOption)
        ftExp.pruning()
        ftExp._load_resume()
        ftExp.fine_tuning()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DCP')
    parser.add_argument('-c', '--conf_path', type=str, metavar='conf_path',
                        help='the path of config file', default='config.yaml')
    parser.add_argument('-n', '--net', type=int, metavar='network_type',
                        help='net_type', default=0)
    args = parser.parse_args()
    engine = DCP(args.conf_path, args.net)
    engine()

