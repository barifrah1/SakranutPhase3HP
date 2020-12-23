from torch import nn

args = {
    'fileName': 'ks-projects-201801_bar.csv',
    'trainSize': 0.8,
    'batch_size': 26779,
    'n_epochs': 500,
    'weight_decay': 2.780552870258695e-05,
    'lr': 0.0012530911275610264,
    'number_of_words_features': 365,

}

loss_function = nn.BCELoss()
