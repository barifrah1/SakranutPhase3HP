from torch import nn

args = {
    'fileName': 'ks-projects-201801_bar.csv',
    'trainSize': 0.8,
    'batch_size': 2000,
    'n_epochs': 500,

}

loss_function = nn.BCELoss()
