# Adaptive Multi-scale Perception-based Decoupled Network for Efficient and Accurate Photovoltaic Forecasting

>**Authors:** Lei Liu, Lelin Hu, Zhongwei Guo, Bin Li, Hongwei Zhao

This repo contains the code and data from our paper published in Journal of Solar Energy.

Website：https://www.sciencedirect.com/science/article/abs/pii/S0038092X26002513.

## 1. Abstract

Photovoltaic (PV) power is significantly affected by meteorological factors such as irradiance and temperature,and exhibits strong volatility under variable weather conditions,which brings great challenges to PV power forecasting. Existing forecasting methods mainly suffer from two limitations: 1) Multi factor coupled modeling approach is adopted without considering the heterogeneity in the intrinsic characteristics of different meteorological variables. 2) Multi-scale feature perception is introduced with fixed strategies, making it difficult to adaptively capture power fluctuations and periodic dynamic features driven by weather changes. To address these problems, this paper proposes an Adaptive Multi-scale Perception-based Decoupled Network (AMPDNet) for PV power forecasting. Unlike previous RNN-based models and Transformer-based models, AMPDNet is a lightweight and efficient forecasting model entirely based on convolutional operations. It introduces a decoupling paradigm that achieves triple decoupling of input information along temporal, channel, and variable dimensions, thereby enabling more fine-grained and effective feature extraction. The proposed Adaptive Multi-scale Convolution (AMSConv) module captures multi-scale temporal dependencies, aiming to handle forecasting tasks under variable weather conditions. Experimental results on two solar power stations operated by the State Grid Corporation of China show that AMPDNet reduces NMSE, NRMSE, and NMAE by 12.6%, 7.8%, and 9.2%, respectively, on average across the 15-minute, 1-hour, and 4-hour forecasting tasks. In terms of forecasting accuracy and efficiency, AMPDNet significantly outperforms mainstream methods such as PatchTST and Transformer, validating the effectiveness of its architectural design.

## 2. Environment setup

```bash
  conda create -n solar_predict python=3.9.23 
  conda activate solar_predict 
  pip install -r requirements.txt
```

## 3. Datasets

The Renewable-energy-generation-input-feature-variables-analysis dataset is already placed in the datasets folder. 

The Renewable-energy-generation-input-feature-variables-analysis dataset is as follows:

https://github.com/Bob05757/Renewable-energy-generation-input-feature-variables-analysis.

## 4. Usage

* An example for train and evaluate a new model：

```bash
bash ./scripts/run_PV1_old.sh
```

- You can get the following output:

```bash
bash ./scripts/run_PV1_old.sh🚀 model: ConvPVNet | task: 4h_64 | input_length: 64 | output_length:16
/home/ghy/ghy/anaconda3/envs/solar/lib/python3.9/site-packages/timm/models/layers/__init__.py:48: FutureWarning: Importing from timm.models.layers is deprecated, please import via timm.layers warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.layers", FutureWarning)
Solar_processed_shape:(38012, 8)

特征列表: ['Power', 'TSI', 'DNI', 'GHI']feat_names:['Power', 'TSI', 'DNI', 'GHI']

train_len:22838

max_value:Power 48.321732TSI 1328.000000DNI 923.000000GHI 989.000000Temp 41.200001Humidity 69.900002Atmos 934.500000dtype: float32min_value:Power 0.000000TSI 0.000000DNI 0.000000GHI 0.000000Temp -18.200001Humidity 0.000000Atmos 894.900024dtype: float32

时间戳范围: 2019-01-01 06:00:00 到 2020-12-31 18:45:00data1_x_shape: (37933, 64, 4) data1_y_shape: (37933, 16)x_train_shape: (22759, 64, 4) y_train_shape: (22759, 16)x_valid_shape: (7587, 64, 4) y_valid_shape: (7587, 16)x_test_shape: (7587, 64, 4) y_test_shape: (7587, 16)ConvPVNet( (embed_layer): Embedding( (conv): Conv1d(1, 64, kernel_size=(4,), stride=(2,)) (dropout): Dropout(p=0, inplace=False) ) (backbone): ModuleList( (0-1): 2 x ConvPVNetBlock( (multi_dw_conv): MultiDWConvBNWeightedSum2( (convs): ModuleList( (0): Conv1d(256, 256, kernel_size=(5,), stride=(1,), padding=same, groups=256) (1): Conv1d(256, 256, kernel_size=(5,), stride=(1,), padding=same, dilation=(2,), groups=256) (2): Conv1d(256, 256, kernel_size=(5,), stride=(1,), padding=same, dilation=(3,), groups=256) (3): Conv1d(256, 256, kernel_size=(5,), stride=(1,), padding=same, dilation=(4,), groups=256) ) (bns): ModuleList( (0-3): 4 x BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) ) (weights1): ParameterList( (0): Parameter containing: [torch.float32 of size (cuda:6)] (1): Parameter containing: [torch.float32 of size (cuda:6)] (2): Parameter containing: [torch.float32 of size (cuda:6)] (3): Parameter containing: [torch.float32 of size (cuda:6)] ) ) (conv_glu1): ConvGLU( (dropout1): Dropout(p=0.0, inplace=False) (dropout2): Dropout(p=0.0, inplace=False) (pw_con1): Conv1d(256, 512, kernel_size=(1,), stride=(1,), groups=4) (pw_con2): Conv1d(256, 512, kernel_size=(1,), stride=(1,), groups=4) (pw_con3): Conv1d(512, 256, kernel_size=(1,), stride=(1,), groups=4) ) (conv_glu2): ConvGLU( (dropout1): Dropout(p=0.0, inplace=False) (dropout2): Dropout(p=0.0, inplace=False) (pw_con1): Conv1d(256, 512, kernel_size=(1,), stride=(1,), groups=64) (pw_con2): Conv1d(256, 512, kernel_size=(1,), stride=(1,), groups=64) (pw_con3): Conv1d(512, 256, kernel_size=(1,), stride=(1,), groups=64) ) ) ) (head): Linear(in_features=2048, out_features=16, bias=True))Number of trainable parameters: 263512Number of all parameters: 263512Memory usage before training: 0.039818763732910156GBStart training...Epoch [1/10], train_mse: 0.02840920, train_mae: 0.11264890, train_r2: 0.68364096, valid_mse: 0.01890338, valid_mae: 0.09524594, valid_r2: 0.75385791Epoch [2/10], train_mse: 0.01753057, train_mae: 0.08754724, train_r2: 0.80368096, valid_mse: 0.01651401, valid_mae: 0.08662295, valid_r2: 0.78497005Epoch [3/10], train_mse: 0.01555797, train_mae: 0.08002504, train_r2: 0.82583767, valid_mse: 0.01502449, valid_mae: 0.08621636, valid_r2: 0.80436510Epoch [4/10], train_mse: 0.01502424, train_mae: 0.07748817, train_r2: 0.83209836, valid_mse: 0.01495358, valid_mae: 0.08255765, valid_r2: 0.80528843Epoch [5/10], train_mse: 0.01477540, train_mae: 0.07641881, train_r2: 0.83486795, valid_mse: 0.01508719, valid_mae: 0.08350282, valid_r2: 0.80354875Epoch [6/10], train_mse: 0.01457418, train_mae: 0.07552493, train_r2: 0.83720750, valid_mse: 0.01470133, valid_mae: 0.08148909, valid_r2: 0.80857307Epoch [7/10], train_mse: 0.01443302, train_mae: 0.07458779, train_r2: 0.83848643, valid_mse: 0.01560068, valid_mae: 0.08303490, valid_r2: 0.79686248Epoch [8/10], train_mse: 0.01436755, train_mae: 0.07399451, train_r2: 0.83929914, valid_mse: 0.01472248, valid_mae: 0.07939390, valid_r2: 0.80829763Epoch [9/10], train_mse: 0.01424890, train_mae: 0.07357519, train_r2: 0.84076142, valid_mse: 0.01471976, valid_mae: 0.07865573, valid_r2: 0.80833310Epoch [10/10], train_mse: 0.01414327, train_mae: 0.07306417, train_r2: 0.84219384, valid_mse: 0.01463911, valid_mae: 0.07906239, valid_r2: 0.80938315all_preds_test_shape: torch.Size([7587, 16])all_true_test_shape: torch.Size([7587, 16])

NMSE_test: 0.01440780 NRMSE_test: 0.12003248 NMAE_test: 0.07164998 NR2_test: 0.84243327 NSMAPE_test: 0.65067333

MSE_test: 33.64205933 RMSE_test: 5.80017753 MAE_test: 3.46225119 R2_test: 0.84243321 SMAPE_test: 0.65067333
done!
```

## 5. Citation

If you find our work useful in your research, please consider citing:

```latex
@article{liu2026adaptive,
  title={Adaptive multi-scale perception-based decoupled network for efficient and accurate photovoltaic forecasting},
  author={Liu, Lei and Hu, Lelin and Guo, Zhongwei and Li, Bin and Zhao, Hongwei},
  journal={Solar Energy},
  volume={312},
  pages={114563},
  year={2026},
  publisher={Elsevier}
}
```

If you have any problems, contact me via liulei13@ustc.edu.cn.
