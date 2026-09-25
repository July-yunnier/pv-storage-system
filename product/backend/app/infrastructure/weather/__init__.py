"""天气 Provider 抽象：优化器不依赖具体气象源。

首版实现：
- OpenMeteoProvider：Open-Meteo 免费 API（逐小时辐射预报）
- MockWeatherProvider：确定性晴空曲线，供测试和离线开发
"""
