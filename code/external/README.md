# External code

| Folder | Upstream | Commit | How stored | Local changes |
|---|---|---|---|---|
| auto-circuit | https://github.com/UFO-101/auto-circuit | 8fc5afd | plain files (vendored) | data.py, prune_algos/parameter_integrated_gradients.py, utils/patchable_model.py (TransformerLens KV-cache import fix), visualize.py. Full diff: auto-circuit_local_changes.patch |
| MIB-circuit-track | https://github.com/hannamw/MIB-circuit-track | pinned by submodule | git submodule | none -- never edit; outputs go to results/mib/ |

EAP-IG (https://github.com/hannamw/EAP-IG, e24bafd) was removed; MIB-circuit-track includes it as its own submodule.
