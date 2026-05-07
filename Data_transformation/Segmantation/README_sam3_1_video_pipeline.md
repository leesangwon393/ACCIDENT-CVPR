# SAM 3.1 traffic video runner

이 디렉터리의 데이터 생성 스크립트들은 이동된 `../Experiment/Segmantation/run_sam3_1_objects_to_single_mask_video.py`를 호출해, `facebook/sam3.1` 체크포인트로 **모든 프롬프트를 순차 추론한 뒤 하나의 합성 마스크 영상**을 생성합니다.

## 기본 실행

```bash
/root/Desktop/workspace/yuyeon/Experiments/sam3_1_env/bin/python \
  "/Users/isang-won/Desktop/CVPR/Experiment/Segmantation/run_sam3_1_objects_to_single_mask_video.py"
```

기본값은 아래 경로를 사용합니다.

- video: `/root/Desktop/workspace/yuyeon/Experiments/16. RB-FT/sim_dataset_aug_fixed3_aug2/videos/single/Town10HD_single_sunset_22_aug2.mp4`
- classes: `/root/Desktop/workspace/yuyeon/Experiments/16. RB-FT/sim_dataset_aug_fixed3_aug2/annotation_classes.yaml`
- checkpoint: `/root/Desktop/workspace/yuyeon/Experiments/17. SSSSS/checkpoints/sam3.1/sam3.1_multiplex.pt`

## 출력

기본 출력 디렉터리:

`/root/Desktop/workspace/yuyeon/Experiments/17. SSSSS/runs/sam3_1_all_classes_<video_stem>`

생성 파일:

- `all_classes_combined_mask.mp4`: 전체 프롬프트 마스크가 합성된 최종 영상
- `combined_mask.json`: 실행 설정, 프롬프트별 통계, 출력 경로 요약

## 자주 쓰는 옵션

- `--exclude-labels Unlabeled Other`: 특정 라벨 제외
- `--prompt-frame-index 0`: 텍스트 프롬프트를 걸 기준 프레임
- `--propagation-direction both`: 기준 프레임 앞뒤로 모두 전파
- `--offload-video-to-cpu`: 비디오 프레임을 CPU 쪽으로 오프로딩
- `--compile`: `torch.compile` 활성화
- `--use-fa3`: Flash Attention 3 사용. `flash_attn_interface`가 설치된 환경에서만 켜세요.
- `--image-size 1008`: 내부 추론 해상도. 기본값은 라이브러리 기본 설정입니다.
- `--postprocess-batch-size 4 --batched-grounding-batch-size 4`: 배치 메모리 사용량 조절

예시:

```bash
/root/Desktop/workspace/yuyeon/Experiments/sam3_1_env/bin/python \
  "/Users/isang-won/Desktop/CVPR/Experiment/Segmantation/run_sam3_1_objects_to_single_mask_video.py" \
  --exclude-labels Unlabeled Other \
  --offload-video-to-cpu
```
