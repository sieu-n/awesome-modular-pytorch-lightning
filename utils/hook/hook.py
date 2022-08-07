from utils import rgetattr


class Hook:
    def __init__(self, network=None, cfg=None):
        """
        Create and record hooks that record forward / backward passes of a
        layer in a network.
        """
        self.hooks = []
        self.forward_cache = {}
        self._network = network

        if cfg is not None:
            self.build_from_cfg(cfg)

    def set_network(self, network):
        self._network = network

    def get_all(self, device=None):
        return {key: self.get(key, device) for key in self.hooks}

    def build_from_cfg(self, cfg):
        if type(cfg) == list:
            cfg = {idx: val for idx, val in enumerate(cfg)}
        else:
            assert isinstance(
                cfg, dict
            ), f"`build_from_cfg` expected a list or dict, got \
                {cfg} of type {type(cfg)}."

        for hook_name, hook_cfg in cfg.items():
            print(f"Registering `{hook_name}` hook to `{hook_cfg['layer_name']}`. ")
            self.register_forward_hook(
                name=hook_name,
                layer_name=hook_cfg["layer_name"],
                **hook_cfg.get("args", {}),
            )

    def register_forward_hook(
        self, name, layer_name, network=None, mode="output", idx=None
    ):
        """
        Attach a new forward hook to `layer_name` layer in `network`. The
        results can be retrieved through calling `get_hook`.
        """

        def save_to(name, mode="output", idx=None):
            def hook(m, i, output):
                # initialize array for device
                # TODO: tensor might be on CPU?
                if hasattr(m, "device"):
                    device_idx = m.device.index
                elif hasattr(i, "device"):
                    device_idx = i.device.index
                elif hasattr(output, "device"):
                    device_idx = output.device.index

                if device_idx not in self.forward_cache:
                    self.forward_cache[device_idx] = {}

                assert mode in ("output", "input")
                if mode == "output":
                    f = output.detach()
                elif mode == "input":
                    f = i.detach()

                if idx is not None:
                    f = f[idx].detach()

                self.forward_cache[output.device.index][name] = f

            return hook

        if network is None:
            network = self._network

        layer = rgetattr(network, layer_name)
        layer.register_forward_hook(save_to(name, mode=mode, idx=idx))
        self.hooks.append(name)

    def get(self, key, device=None):
        if device is None:
            assert (
                len(self.forward_cache) == 1
            ), f"Current `device` must be provided when \
                using multi-node training. GPU used: {self.forward_cache}"
            device_index = list(self.forward_cache.keys())[0]
        else:
            device_index = device.index
        return self.forward_cache[device_index][key]
