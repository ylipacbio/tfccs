## To install tensorflow v2.0 + python 3.7 with pacbio module environment.
```
bash script/install-tf/install-tensorflow-python.sh
```
## To install tensorflow v2.0 + C++ library with pacbio module environment.
```
bash script/install-tf/compile-tensorflow-c++.sh
```

## To compute mean/stdev/min/max of fextract.csv file:
```
fextract2stat input.fextract.csv fextract.stat.json
```

## To standardize input fextract.csv and save as numpy compressed model:
```
fextract2numpy input.fextract.csv output --standardize fextract.stat.json
```

## Train a simple multiple-normial model:
bash train_hist/lambda-multi-ccs2genome.sh
