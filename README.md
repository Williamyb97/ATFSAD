# ATFSAD_Enhancing_Long_Sequence_Time-Series_Forecasting_on_Air_Temperature_Prediction

* Make sure the following files are present as per the directory structure before running the code：
```
├── data
|   └── jena_climate

|   └── data_loader.py

├── Model
|    ├── pheme_dual_32_2024-05-22-09-42-09.log
|    ├── attention.py
|    ├── decoder.py
|    ├── embed.py
|    ├── encoder.py
|    ├── exp_main.py
|    ├── myformer.py
├── positiional Embedding
|    ├── positional Embedding.py
├── units
|    ├── untils.py
├── S_T_Label.npy
├── run.py
├── README.md

## Datasets
dataset is inclouded  in folder "dataset".



## Dependencies
* torch==1.12.1
* scipy==1.5.4
* numpy==1.21.5
* pandas==1.1.5

## Run
*  1：Run run.py 


## Note
* The default configuration of the code assumes that the number of states in the dynamic graph is 3. Trying other values requires modifying the configuration and some code.
* The hyperparameters of different models may vary, requiring adjustments to the configuration file.
* The data preprocessing process takes some time, and some of the generated files are quite large, making it inconvenient to upload them. We will share the preprocessing results later on.









