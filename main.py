import os

import config
import modules
import utils

if __name__ == "__main__":
    dataset_name = config.DATA_OPTIONS.dataset_name
    logger = utils.init_logger(os.path.dirname(__file__) + f"/result/{dataset_name}/record.log")
    logger.info("Main procedure started with launch config:\n"
                f"MODEL_OPTIONS:{config.MODEL_OPTIONS}\n"
                f"DATA_OPTIONS:{config.DATA_OPTIONS}\n"
                f"PROCEDURE_OPTIONS:{config.PROCEDURE_OPTIONS}\n")

    utils.init_environment()
    corpus = modules.FormatCorpus(**vars(config.DATA_OPTIONS))
    model = utils.init_model(corpus)
    optimizer = utils.init_optimizer(model)
    scheduler = utils.init_scheduler(corpus, optimizer)
    trainer = utils.Trainer(
        model,
        optimizer,
        scheduler,
        corpus,
        config.OPTIMIZE_OPTIONS.max_grad_norm,
        **vars(config.PROCEDURE_OPTIONS)
    )
    trainer()
