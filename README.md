# ACCIDENT-CVPR

Traffic accident understanding pipeline for the [Kaggle ACCIDENT Competition](https://www.kaggle.com/competitions/accident).

This repository contains experiments and inference pipelines for traffic accident video understanding.  
The goal is to predict **when**, **where**, and **what type** of accident occurs in each video.

---

## Project Overview

The ACCIDENT competition focuses on understanding traffic accident videos.  
Given a video, the model should estimate:

1. **Accident Time**  
   - The timestamp of the first physical impact.

2. **Accident Location**  
   - The normalized center point of the accident impact area.
   - Output format: `center_x`, `center_y` in the range `[0, 1]`.

3. **Accident Type**  
   - One of the predefined accident categories:
     - `rear-end`
     - `head-on`
     - `sideswipe`
     - `t-bone`
     - `single`

This project explores multimodal and video-based approaches for robust accident understanding, especially under the domain gap between simulation training videos and real-world test videos.

---

## Main Ideas

This repository includes experiments based on the following ideas:

### 1. Vision-Language Model Based Inference

We use a Vision-Language Model to analyze sampled video frames and predict structured accident information.

The model is prompted to output JSON only:

```json
{
  "accident_time": 1.23,
  "center_x": 0.52,
  "center_y": 0.47,
  "type": "rear-end"
}
