from torch import nn

args = {
    'fileName': 'ks-projects-201801_bar.csv',
    'trainSize': 0.8,
    'batch_size': 2000,
    'n_epochs': 500,
    'weight_decay': 1e-6,
    'lr': 1e-3,
    'number_of_words_features': 250,

}

loss_function = nn.BCELoss()
