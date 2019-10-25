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
