from argparse import ArgumentParser

# custom
from utils.configs import read_configs


if __name__ == "__main__":
    # read config yaml paths
    parser = ArgumentParser()
    parser.add_argument("-c", "--configs", nargs="+", required=True)

    args = parser.parse_args()
    config = read_configs(args.configs)

    # load data
    f_train_augmentation = # TODO
    f_val_preprocessing = # TODO

    train_dataset = # TODO
    val_dataset = # TODO

    train_dataloader = # TODO
    val_dataloader = # TODO

    # callbacks
    wandb_callback = #TODO
    checkpoint_callback = #TODO
    ...

    # model
    model = # TODO(instance of pl.LightningModule)
    trainer = pl.Trainer()


    trainer.fit(model, train_dataloader, val_dataloader)

