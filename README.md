# collaborative-filtering

A small project involving Collaborative filtering

## Autoencoder in Torch

### Links
![http://www.grappa.univ-lille3.fr/~mary/cours/stats/centrale/reco/](Data exploration, SVD/ALS Baseline)
![https://github.com/fstrub95/torch.github.io/blob/master/blog/_posts/2016-02-21-cfn.md](introduction Autoencoder for collaboration filtering)

### Install
```
CC=gcc-4.8 CXX=g++4.8 luarocks install cutorch
```


### Run
```
th data.lua  -ratings ../../input/customeraffinity.train -out ../../input/customer-train.t7 -fileType movieLens -ratio 0.9
th main.lua  -file ../../input/customer-train.t7 -conf ../conf/conf.movieLens.10M.U.lua  -save network.t7 -type V -meta 0 -gpu 1
th computeMetrics.lua -file ../../input/customer-train.t7 -network network.t7 -type V -gpu 1
```