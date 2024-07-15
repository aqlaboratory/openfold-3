# %%
import yaml
from openfold3.model_implementations.af2_monomer import base_config

with open('openfold3/model_implementations/af2_monomer/runner.yaml') as f:
    runner_config = yaml.safe_load(f)

print(runner_config)

with open('openfold3/model_implementations/af2_monomer/reference_config.yml') as f:
    reference_configs = yaml.safe_load(f) 

# load presets first
base = base_config.config.copy_and_resolve_references()
preset = runner_config['general']['preset'] 
print(preset)
print(reference_configs[preset])

base.update(reference_configs[preset])
print(base)

# load other updates from runner config
if 'model' in runner_config:
    base['model'].update(runner_config['model'])

print(f"{base.model.template.enabled=}")

# %%
