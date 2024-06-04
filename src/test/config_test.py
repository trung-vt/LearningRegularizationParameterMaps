import hydra

@hydra.main(config_path=".", config_name="model_turtle_config")
def config_test(config) -> None:
    print(f"\n{config}")
    print(f"\n{config.pdhg}")
    assert config.pdhg.T_train == 128, f"Expect T_train to be 128, got {config.pdhg.T_train}"

config_test()