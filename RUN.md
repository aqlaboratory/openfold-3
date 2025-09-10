## Antibody-Antigen evaluation
This branch is for antibody-antigen benchmarking from batch dictionaries I dumped from old czi-rnp-inference branch.
This uses OpenFold3 MSA and templates. All the features are already processed and saved to dictionary. So it doesn't uses the normal data module.
I copied 2 most largest structures to `/global/cfs/cdirs/m4351/ab_ag_test`

Test using 1 GPU (80GB)
```
python run_openfold.py predict-batch-dict \
  --batch_dir /global/cfs/cdirs/m4351/ab_ag_test \
  --inference_ckpt_path /global/cfs/cdirs/m4351/of3_checkpoints/pohpyeb4/fp32/164-78000.ckpt \
  --min_seed 0 --max_seed 1 \
  --runner_yaml inference_1gpu.yml \
  --output_dir $YOUR_OUTPUT_DIR
```

`inference_1gpu.yml` sets `num_recycles=0` for fast testing. 