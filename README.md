# collaborative-filtering

A small project involving Collaborative filtering

## Autoencoder in Torch

### Links
[Data exploration, SVD/ALS Baseline](http://www.grappa.univ-lille3.fr/~mary/cours/stats/centrale/reco/)
[Autoencoders for collaborative filtering](https://github.com/fstrub95/torch.github.io/blob/master/blog/_posts/2016-02-21-cfn.md)

### Install
```
CC=gcc-4.8 CXX=g++4.8 luarocks install cutorch
```


### Running the network
Start by parsing the ratings file into a torch data file:
```
th data.lua  -ratings ../../input/customeraffinity.train -out ../../input/customer-train.t7 -fileType alix -ratio 0.9
```
Run the network using a neural net configuration:
```
th main.lua  -file ../../input/customer-train.t7 -conf ../conf/conf.ratings.U.lua  -save ../output/network.R.t7 -type U -meta 0 -gpu 0
```
Compute metrics with the final network weights:
```
th computeMetrics.lua -file ../../input/customer-train.t7 -network ../output/network.R.t7 -type U -gpu 0
```