[모델 특징]
1. SentencePiece 활용
 - 비격식 한글 문장들과 영어, 특수문자 등을 효과적으로 처리하기 위함
 
2. 작은 크기의 BERT 모델 기반
 - 긴 문장 데이터를 효과적으로 처리하기 위해 BERT 모델 채택
 - 데이터의 수가 적기 때문에 작은 크기의 모델 구축
 
3. RNN을 활용하여 결과 예측
 - Process: 입력 문단을 일정 크기의 문장으로 나눈 뒤, 각 문장을 BERT에 입력. 그리고 문장 별 특징들을 RNN에 입력하여 결과 예측
 - 가변 길이의 문단을 효과적으로 처리하기 위함
 
4. 4개 모델 앙상블
 - 입력 문단을 나누는 기준 토큰 개수를 달리한(16, 32, 64, 128) 4개 모델들 사용
 - 모델 성능의 안정성을 위해 사용

[dependency]
pytorch==1.4.0
sentencepiece
knlpy
easydict

[코드 실행 순서]
# 학습/평가 데이터 파일 생성 (임베딩 방식 선택 가능)
python preprocessing.py

# 사전 학습 (only Masked Language Model)
python train_lm.py

# 분류 모델 학습
python train_cls.py

# 모델 앙상블로 결과 파일 생성
python inference_bertcls.py
