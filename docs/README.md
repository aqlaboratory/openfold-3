
conda env create --file environment.yml

conda activate openfold3-docs

make html

Then you can serve

cd build/html
python -m http.server 8080
