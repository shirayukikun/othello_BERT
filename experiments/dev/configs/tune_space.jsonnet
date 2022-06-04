{
  initial_space: {
    lr_and_end_lr: {
      lr: [1e-6, 5e-6, 1e-5, 5e-5, 1e-4],
      end_lr: [lr * 0.1 for lr in self.lr],
    },
    dropout: [0.01, 0.1, 0.3, 0.5, 0.9],
    activation_dropout: self.dropout,
    attention_dropout: self.dropout,
    batch_size: [8, 16, 32, 64, 128],
  },

  during_training_space: {
  },
}


#PBT
#{
#  initial_space: {
#    lr_and_end_lr: {
#      lr: [8e-6, 9e-6, 1e-5, 2e-5, 3e-5],
#      end_lr: [lr * 0.1 for lr in self.lr],
#    },
#  },
#
#  during_training_space: {
#    dropout: [0.05, 0.075, 0.1, 0.125, 0.15],
#    activation_dropout: [0.05, 0.075, 0.1, 0.125, 0.15],
#    attention_dropout: [0.05, 0.075, 0.1, 0.125, 0.15],
#    batch_size: [8, 16, 32, 64, 128],
#  },
#}
