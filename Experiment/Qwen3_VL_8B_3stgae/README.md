- 1.baseline.py: 공용 inference/helper  
  - 프레임 샘플링, prompt 구성, JSON 파싱 같은 공통 함수

- 2.making_rationle.py: stage-1 rationale 생성
  - 라벨된 비디오로부터 설명(rationale) 데이터셋 만들기

- 3.FT_stage1.py: 
  - 외부 교통사고 데이터를 활용해서 학습

- 4.find_object.py: 객체 추출 inference
  - stage1 adapter로 사고 관련 객체만 뽑는 스크립트

- 5. find_object_train_sampled.py: 샘플링된 영상용 객체 추출/학습 데이터 생성
  - 4번을 학습/샘플링 데이터용으로 돌리는 버전
  
- 6.FT_stage2.py: stage-2 최종 파인튜닝
  - stage1 adapter에서 이어받아 최종 competition JSON 출력용으로 학습
  - language LoRA attention 전부
  - vision LoRA도 일부