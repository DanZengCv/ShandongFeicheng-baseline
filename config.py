class DefaultConfigs(object):
    # cuda
    cuda = True

    # Dataset selection
    dataset = 'ShandongDowntown' # ShandongDowntown/ShandongFeicheng
    inference = False # For inference
    inference_onlyTrainData = False 
    maxTrain = True # Whether use limited data to train
    max_trainData = 200
    
    # train/test parameters
    model_name = 'C3F4_CNN' # ResNetv2/C3F4_CNN/
    CBLoss_gamma = 1
    optimizer = 'SGD' # Adagrad/SGD/Adam
    epochs = 200
    step_size = 40
    seed = 80 # 80-86
    weight_decay = 1e-2

    # Dataset
    if dataset == 'ShandongDowntown':
        band = 63
        num_classes = 15
        patch_size = 29

        train_percent = 0.75
        val_percent = 0
        test_percent = 0.25

        lr = 0.01
        batch_size = 100

    if dataset == 'ShandongSuburb':
        band = 63
        num_classes = 19
        patch_size = 29

        train_percent = 0.75
        val_percent = 0
        test_percent = 0.25

        lr = 0.01
        batch_size = 200

    patch_mode = 'Center' # Center/TopLeft/PP(Pixel-Pair)
  
    def print_config(self):
        print('#######################PARAMETERS#######################'
          '# cuda\n'
          '\tcuda: {}\n'
          
          '# train/test parameters'
          '\tmodel_name: {}\n'
          '\toptimizer: {}\n'
          '\tepochs: {}\n'
          '\tbatch_size: {}\n'
          '\tseed: {}\n'
          '\tlr: {}\n'
          '\tweight_decay: {}\n'
          
          '# data preparation parameters\n'
          '\tdataset: {}\n'
          '\tpatch_size: {}\n'
          '\tband: {}\n'
          '\tnum_classes: {}\n'
          '\ttrain_percent: {}\n'
          '\tval_percent: {}\n'.format(
          config.cuda, config.model_name, config.optimizer,
          config.epochs, config.batch_size, config.seed,
          config.lr, config.weight_decay, config.dataset, config.patch_size, 
          config.band, config.num_classes, config.train_percent, config.val_percent))
        print('#'*60)
        print('\n')

config = DefaultConfigs()
if __name__ == '__main__':
    config.print_config()