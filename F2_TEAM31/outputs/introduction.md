# 1. INTRODUCTION

## 1.1 Background and Motivation

Network traffic changes over time and varies across links. Forecasting traffic in advance can help network administrators anticipate demand and plan network resources. Traffic on one link may also be related to traffic on other links, depending on the network layout and the patterns observed in past measurements.

This project studies traffic forecasting using the Abilene network dataset. It compares a temporal model, which learns from past traffic values, with graph-based models that also use relationships between links. The project also examines whether links with similar past traffic patterns can improve forecasting.

## 1.2 Problem Statement

The objective is to predict the traffic load on each of the Abilene network’s 30 internal directed links five minutes ahead, using the previous 12 five-minute observations. The challenge is to determine whether representing relationships between links improves forecasts compared with a model that learns only from past traffic.

## 1.3 Objectives

The objectives of this project are:

- Prepare Abilene origin-destination traffic measurements as link-load time series.
- Implement and compare a GRU baseline, a physical-topology Graph-GRU, and graph models using traffic-similarity relationships.
- Select model settings using validation data and evaluate them on a later test period.
- Compare forecast errors using MAE and RMSE, including lower-, medium-, and higher-load periods where the data allows.
- Discuss the results and limitations without assuming that a graph-based model will always perform better.

## 1.4 Scope and Limitations

The project uses one week of Abilene traffic data, sampled every five minutes. It forecasts one step ahead and evaluates 30 internal directed links using a chronological 60/20/20 training, validation, and test split. The experiments compare three random seeds.

The evaluation is limited to this dataset, forecast horizon, model configuration, and split. Under the training-defined load thresholds, all test windows fall in the low-load regime. Therefore, the primary test evaluation does not establish performance on medium- or high-load test conditions. Results should not be treated as proof that the selected model will generalize to other networks or traffic periods.

## 1.5 Mapping to the Computer Networks Lab

| Course concept | Project application |
|---|---|
| Network topology and routing | Representing internal links and using the routing matrix to derive link loads from origin-destination traffic |
| Network traffic and performance | Studying link-load patterns and forecasting traffic demand |
| Experimental evaluation | Comparing models using a chronological data split, repeated runs, MAE, and RMSE |
| Python-based network analysis | Preparing the dataset, training models, and producing result tables and graphs |

