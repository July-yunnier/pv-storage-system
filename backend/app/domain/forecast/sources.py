"""Citations and verified implementation licenses for experiment metadata."""
from __future__ import annotations

import importlib.metadata

from app.domain.forecast.models import package_for

PAPERS = {
    "ARIMA": "Box and Jenkins, Time Series Analysis: Forecasting and Control (1970)",
    "SARIMA": "Box and Jenkins, seasonal ARIMA (1970)",
    "Prophet": "Taylor and Letham, Forecasting at Scale (2018), doi:10.1080/00031305.2017.1380080",
    "SVR": "Smola and Scholkopf, A tutorial on support vector regression (2004), doi:10.1023/B:STCO.0000035301.49549.88",
    "LightGBM": "Ke et al., LightGBM: A Highly Efficient Gradient Boosting Decision Tree (NeurIPS 2017)",
    "XGBoost": "Chen and Guestrin, XGBoost (KDD 2016), doi:10.1145/2939672.2939785",
    "LSTM": "Hochreiter and Schmidhuber, Long Short-Term Memory (1997), doi:10.1162/neco.1997.9.8.1735",
    "GRU": "Cho et al. (2014), arXiv:1406.1078",
    "BiLSTM": "Schuster and Paliwal, Bidirectional recurrent neural networks (1997), doi:10.1109/78.650093",
    "TFT": "Lim et al., Temporal Fusion Transformers (2021), doi:10.1016/j.ijforecast.2021.03.012",
    "Informer": "Zhou et al., Informer (AAAI 2021), arXiv:2012.07436",
    "Autoformer": "Wu et al., Autoformer (NeurIPS 2021), arXiv:2106.13008",
    "PatchTST": "Nie et al., A Time Series is Worth 64 Words (ICLR 2023), arXiv:2211.14730",
    "DLinear": "Zeng et al., Are Transformers Effective for Time Series Forecasting? (AAAI 2023), arXiv:2205.13504",
    "RLinear": "Li et al., Revisiting Long-term Time Series Forecasting: An Investigation on Linear Mapping (2023), arXiv:2305.10721",
    "NHiTS": "Challu et al., N-HiTS (AAAI 2023), arXiv:2201.12886",
}
PACKAGES = {
    "statsmodels": ("statsmodels", "BSD-3-Clause", "https://github.com/statsmodels/statsmodels"),
    "prophet": ("prophet", "MIT", "https://github.com/facebook/prophet"),
    "sklearn": ("scikit-learn", "BSD-3-Clause", "https://github.com/scikit-learn/scikit-learn"),
    "lightgbm": ("lightgbm", "MIT", "https://github.com/microsoft/LightGBM"),
    "xgboost": ("xgboost", "Apache-2.0", "https://github.com/dmlc/xgboost"),
    "torch": ("torch", "BSD-3-Clause", "https://github.com/pytorch/pytorch"),
    "neuralforecast": ("neuralforecast", "Apache-2.0", "https://github.com/Nixtla/neuralforecast"),
}


def citations(name: str) -> dict:
    package = package_for(name)
    result = {"algorithm": PAPERS.get(name, "Persistence/historical-average project baseline"),
              "dataset": {"citation": "Hong et al. (2016), Probabilistic energy forecasting: GEFCom2014 and beyond",
                          "doi": "10.1016/j.ijforecast.2016.02.001",
                          "license": "Not established by local V2 archive; do not assume an open-source code license applies to data."}}
    if package:
        dist, license_id, url = PACKAGES[package]
        try:
            version = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            version = "unavailable"
        result["implementation"] = {"package": dist, "version": version, "license": license_id, "url": url,
                                    "adaptation": "official forward implementation" if package == "neuralforecast" else "project adapter"}
    else:
        result["implementation"] = {"package": "project baseline", "license": "No project license established in local repository; confirm with owner before redistribution."}
    if name == "RLinear":
        result["implementation"]["adaptation"] = "project RevIN+linear reference variant with optional historical exogenous head, not official author weights"
    result["intervals"] = {"method": "validation residual P10/P90; split-conformal central 80%",
                           "citation": "Lei et al. (2018), Distribution-Free Predictive Inference for Regression, doi:10.1080/01621459.2017.1307116",
                           "limitation": "chronological residuals need not be exchangeable; no unconditional time-series coverage guarantee"}
    return result
