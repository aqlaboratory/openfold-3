"""
This module contains all inference configuration settings.

This would be another way to organize configuration settings 
that would be more dense, but would introduce a second way of changing
parameters
"""
import ml_collections as mlc

# AF2 Suppl. Table 5, Model 1.1.1
model_1_config_dict = mlc.ConfigDict({
        "data.train.max_extra_msa": 5120,
        "data.predict.max_extra_msa": 5120,
        "data.common.reduce_max_clusters_by_max_templates": True,
        "data.common.use_templates": True,
        "data.common.use_template_torsion_angles": True,
        "model.template.enabled": True,
}) 

# AF2 Suppl. Table 5, Model 1.1.2
model_2_config_dict = mlc.ConfigDict({
        "data.common.reduce_max_clusters_by_max_templates": True,
        "data.common.use_templates": True,
        "data.common.use_template_torsion_angles": True,
        "model.template.enabled": True,
}) 

# AF2 Suppl. Table 5, Model 1.2.1
model_3_config_dict = mlc.ConfigDict({
        "data.train.max_extra_msa": 5120,
        "data.predict.max_extra_msa": 5120,
        "model.template.enabled": False,
}) 

# AF2 Suppl. Table 5, Model 1.2.2
model_4_config_dict = model_3_config_dict

model_5_config_dict = mlc.ConfigDict({
        "model.template.enabled": False,
}) 

model_1_ptm_config_dict = mlc.ConfigDict({
        "data.train.max_extra_msa": 5120,
        "data.predict.max_extra_msa": 5120,
        "data.common.reduce_max_clusters_by_max_templates": True,
        "data.common.use_templates": True,
        "data.common.use_template_torsion_angles": True,
        "model.template.enabled": True,
        "model.heads.tm.enabled": True,
        "loss.tm.weight": 0.1,
}) 

model_2_ptm_config_dict = mlc.ConfigDict({
        "data.common.reduce_max_clusters_by_max_templates": True,
        "data.common.use_templates": True,
        "data.common.use_template_torsion_angles": True,
        "model.template.enabled": True,
        "model.heads.tm.enabled": True,
        "loss.tm.weight": 0.1,
}) 

model_3_ptm_config_dict = mlc.ConfigDict({
        "data.train.max_extra_msa": 5120,
        "data.predict.max_extra_msa": 5120,
        "model.template.enabled": False,
        "model.heads.tm.enabled": True,
        "loss.tm.weight": 0.1,
}) 

model_4_ptm_config_dict = model_3_ptm_config_dict

model_5_ptm_config_dict = mlc.ConfigDict({
        "model.template.enabled": False,
        "model.heads.tm.enabled": True,
        "loss.tm.weight": 0.1,
})