import lightning
from utils.pretrained import load_model_weights


class TeacherModelKD:
    def __init__(self, model_cfg, training_cfg, fuse_bn=True):
        print("[*] Initializing teacher model for knowledge distillation!")
        assert (
            "state_dict_path" in model_cfg
        ), "Teacher model must be initialized with pretrained state_dict."
        lightning_module = lightning.get(training_cfg["ID"])
        model = lightning_module(model_cfg, training_cfg)
        if "state_dict_path" in model_cfg:
            load_model_weights(
                model=model,
                state_dict_path=model_cfg["state_dict_path"],
                is_ckpt=model_cfg.get("is_ckpt", False),
            )
        # freeze weights
        for param in model.parameters():
            param.requires_grad = False
        # TODO bn fusion
        # save in Python-native list so model is not added to state_dict of student model.
        self.model = [model]

    def normalize_input(self, x, mean, std):
        # TODO
        return x

    def __call__(self, x):
        r = self.model[0](x)
        return r
