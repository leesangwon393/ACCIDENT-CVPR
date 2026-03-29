# ACCIDENT-CVPR

## Overview

This challenge focuses on understanding real traffic car accidents in fixed-view CCTV video. Given a clip, your goal is to predict when the accident happens, where it happens in the frame, and what type of collision it is. In other words, this competition combines temporal localization + spatial localization + accident-type classification. Submissions are evaluated with a single leaderboard score that reflects performance across all three predictions: time, location, and collision type.

Unlike most Kaggle competitions, we don’t provide real labeled training data. The benchmark is meant to evaluate how well your method works out of the box on real CCTV accidents. To help you build and test your pipeline, we provide a synthetic training set you can use for pretraining and debugging.
