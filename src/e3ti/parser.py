@hydra.main(config_path="../configs", config_name="base", version_base="1.3")
def main(cfg: DictConfig):
    torch.autograd.set_detect_anomaly(True)
    if cfg.exp_train.warm_start is not None and cfg.expeexp_trainriment.warm_start_cfg_override:
        # Loads warm start config.
        warm_start_cfg_path = os.path.join(
            os.path.dirname(cfg.exp_train.warm_start), 'config.yaml')
        warm_start_cfg = OmegaConf.load(warm_start_cfg_path)

        # Warm start config may not have latest fields in the base config.
        # Add these fields to the warm start config.
        OmegaConf.set_struct(cfg.model, False)
        OmegaConf.set_struct(warm_start_cfg.model, False)
        cfg.model = OmegaConf.merge(cfg.model, warm_start_cfg.model)
        OmegaConf.set_struct(cfg.model, True)
        logger.info(f'Loaded warm start config from {warm_start_cfg_path}')

    exp = Experiment(cfg)
    exp.train()

if __name__ == "__main__":
    main()