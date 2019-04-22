from model.data_utils import CoNLLDataset
from model.config import Config

from model.ripple_model import RippleModel



def main():
    # create instance of config
    config = Config()

    # build model
    model = RippleModel(config)
    model.build("train")
    model.restore_session(config.dir_model)

    # create dataset
    test = CoNLLDataset(config.filename_test, config.processing_word,
                        config.processing_action, config.max_iter)

    # evaluate and interact
    model.evaluate(test)
    # interactive_shell(model)


if __name__ == "__main__":
    main()
