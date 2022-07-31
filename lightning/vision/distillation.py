import warnings

import catalog
from utils.configs import read_configs
from utils.experiment import print_to_end
from utils.hook import Hook


def DistillationTrainer(model_cfg, training_cfg, const_cfg=None, *args, **kwargs):
    assert "base_ID" in training_cfg
    BaseLightningmodule = catalog.lightning.get(training_cfg["base_ID"])

    class _DistillationTrainer(BaseLightningmodule):
        """
        LightningModule for training knowldege distillation on a variaty of tasks.

        Config should look like this:
        training.distillation: {
            "scale_main_loss": 1-alpha,
            "early_stop_epoch": 30,
            "criterion": {
                name: "LogitKLCriterion",
                args: { "alpha": 1.0, "T": 3, ... }
            },
            "teacher": {
                Choose one of the following:
                    1) provide `model`, `training`, and `const` keys similar to the model.
                    The `state_dict_path` argument of `model` must be used to load weights.
                    2) provide `cfg` as a list of path to config that can create
                    the model
            }
            "hooks": {
                "student":{
                    (optional)
                },
                "teacher": {
                    (optional)
                }
            }
        }
        """

        def init(self, model_cfg, training_cfg):
            super().init(model_cfg, training_cfg)
            assert "distillation" in training_cfg

            print_to_end("-")
            print(
                "[*] Configuring `DistillationTrainer` wrapper for knowledge distillation."
            )
            distill_cfg = training_cfg["distillation"]

            # create teacher model
            print("<1/4> Creating teacher model for knowldege distillation")
            print_to_end(".")
            teacher_cfg = distill_cfg["teacher"]
            if "model" in teacher_cfg and "training" in teacher_cfg:
                assert "cfg" not in teacher_cfg
                if "state_dict_path" not in teacher_cfg["model"]:
                    warnings.warn(
                        "`training.distillation.teacher.model.state_dict_path` \
                        is not specified, the teacher model might be starting from a random state.",
                        RuntimeWarning,
                    )
                self.t_model = catalog.lightning.build_from_cfg(
                    model_cfg=teacher_cfg["model"],
                    training_cfg=teacher_cfg["training"],
                    const_cfg=teacher_cfg.get("const", {}),
                )

            elif "cfg" in teacher_cfg:  # geneate model from path to config files.
                assert "model" not in teacher_cfg and "training" not in teacher_cfg
                _teacher_cfg = read_configs(teacher_cfg["cfg"])

                if "state_dict_path" not in _teacher_cfg["model"]:
                    warnings.warn(
                        "`training.distillation.teacher.model.state_dict_path` \
                        is not specified, the teacher model might be starting from a random state.",
                        RuntimeWarning,
                    )
                self.t_model = catalog.lightning.build_from_cfg(
                    model_cfg=_teacher_cfg["model"],
                    training_cfg=_teacher_cfg["training"],
                    const_cfg=_teacher_cfg.get("const", {}),
                )

            else:
                raise ValueError(
                    f"Recieved invalid arguments for teacher model: {teacher_cfg.keys()}"
                )

            print_to_end(".")
            self.t_model.eval()
            for param in self.t_model.parameters():
                param.requires_grad = False
            # attach hooks
            print("<2/4> Hooks to teacher & student for knowldege distillation")
            if "hooks" in distill_cfg:
                self.use_s_hook = False
                self.use_t_hook = False
                if "student" in distill_cfg["hooks"]:
                    self.use_s_hook = True
                    self.s_hook = Hook(
                        network=self, cfg=distill_cfg["hooks"]["student"]
                    )
                if "teacher" in distill_cfg["hooks"]:
                    self.use_t_hook = True
                    self.t_hook = Hook(
                        network=self.t_model, cfg=distill_cfg["hooks"]["teacher"]
                    )

            # build criterion
            print("<3/4> Criterion for computing knowledge distillation loss.")
            criterion_cfg = distill_cfg["criterion"]
            self.distillation_criterion = catalog.distillation.build(
                name=criterion_cfg["name"],
                args=criterion_cfg["args"],
            )
            # more arguments
            print("<4/4> Additional arguments for knowledge distillation.")
            self.kd_is_enabled = True
            if "early_stop_epoch" in distill_cfg:
                assert (
                    "early_stop_step" not in distill_cfg
                ), "Only one should be specified."
                print("early_stop_epoch:", distill_cfg["early_stop_epoch"])
                self.kd_early_stop_epoch = distill_cfg["early_stop_epoch"]
            if "early_stop_step" in distill_cfg:
                assert (
                    "early_stop_epoch" not in distill_cfg
                ), "Only one should be specified."
                print("early_stop_step:", distill_cfg["early_stop_step"])
                self.kd_early_stop_step = distill_cfg["early_stop_step"]
            """
            The alpha argument in basic KD is implemented slightly differently in a number of papers.
            Most representatively, I noticed two ways papers descrbed KD:
                - (CE loss) + alpha * (kd loss)
                - (1 - alpha) * (CE loss) + alpha * (kd loss)
            By default, we use the first method as it is more generalizable to feature-based KD, ...
            however, `scale_main_loss` value can be set to (1-alpha) for the second case.
            """
            print("scale_main_loss:", distill_cfg.get("scale_main_loss", 1))
            self.kd_scale_main_loss = distill_cfg.get("scale_main_loss", 1.0)

        def state_dict(self):
            # remove teacher model from state dict.
            sd = super().state_dict()
            sd.pop("t_model")
            return sd

        def get_kd_loss(self, x):
            if self.use_s_hook:
                s_out = self.s_hook.get_all(device=self.device)
            else:
                s_out = {}

            if self.use_t_hook:  # get teacher forward.
                _ = self.t_model(x)
                t_out = self.t_hook.get_all(device=self.device)
            else:
                t_out = {}
            return self.distillation_criterion(s_out, t_out)

        def _training_step(self, batch, *args):
            loss, res = super()._training_step(batch, *args)

            # check early stopping
            if (
                hasattr(self, "kd_early_stop_epoch")
                and self.kd_early_stop_epoch == self.current_epoch
            ):
                print("Early-stopping knowledge distillation.")
                self.kd_is_enabled = False
            if (
                hasattr(self, "kd_early_stop_step")
                and self.kd_early_stop_step == self.global_step
            ):
                print("Early-stopping knowledge distillation.")
                self.kd_is_enabled = False

            # compute kd loss
            if self.kd_is_enabled:
                kd_loss, kd_res = self.get_kd_loss(batch["images"])
                self.log("step/kd_loss", kd_loss)

                loss = loss * self.kd_scale_main_loss + kd_loss
                if kd_res is not None:
                    res.update(kd_res)
            return loss, res

    return _DistillationTrainer(model_cfg, training_cfg, const_cfg, *args, **kwargs)
