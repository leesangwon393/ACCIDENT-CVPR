Qwen3.5-9B + optical flow를 섞어서 사고 하나당 아래 4개를 예측합니다:
- accident_time
- center_x
- center_y
- type
흐름
1. accident/test_metadata.csv 읽음
2. 영상마다:
   - optical flow로 사고 시점 추정
   - Qwen3.5로 사고 시점 refine
   - 그 시점 주변 프레임으로 사고 위치(center_x/y) 추정
   - 짧은 클립으로 사고 유형(type) 분류
3. 결과 저장:
   - opticalflow/result/predictions.csv
   - opticalflow/accident/raw_outputs.jsonl
핵심 포인트
- pipeline.optical_flow.compute_motion_curve(...)
  → motion curve 뽑아서 시간 후보를 찾는 데 사용
- AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3.5-9B")
  → VLM으로 시간/위치/유형을 재검증
- 즉, optical flow + Qwen3.5 하이브리드 사고 추론 파이프라인입니다.