# [데이터마이닝학회] Attention-based-Sequential-Recommendation-System-Using-Multimodal
멀티모달을 이용한 어텐션 기반 시퀀셜 추천시스템

# Novelty

(1) 시퀀셜 추천시스템에 멀티모달 정보인 이미지와 텍스트 정보 반영을 제안


(2) 학습 과정 중 멀티모달을 효과적으로 반영하기 위해 Fused item으로 부터 Q, K 연산 수행, Pure item으로 부터 V 연산 수행 

(참조 : non-invasive self-attention for side information fusion in sequential recommendation)

# Summary

  사용자의 구매 이력을 바탕으로 다음 구매할 아이템을 예측하는 시퀀셜 추천시스템은 온라인 플랫폼에서 필수적인 요소이다. 최근 구매 이력 정보에 어텐션을 적용한 시퀀셜 추천시스템 모델은 좋은 성능을 보여주고 있다. 한편, 아이템의 부가 정보를 입력 데이터로 활용하기 위해 멀티모달을 사용한 방법들이 연구되고 있다. 단순히 사용자의 구매 패턴뿐만 아니라, 아이템의 멀티모달 정보는 새로운 아이템을 추천하기 위한 중요한 요소이다. 특히 이미지와 텍스트의 경우 사용자가 아이템을 구매하기 이전에 확인하는 필수 정보이다.  
  본 연구에서는 시퀀셜 추천시스템에 사용자가 구매했던 아이템의 멀티모달 정보인 이미지와 텍스트를 활용하는 방법을 제안한다. 사전 학습된 VGG와 BERT를 활용해서 이미지와 텍스트의 특징들을 추출한 후 Fusion function을 통해 순차 정보와 멀티모달 정보를 합쳐준다. 이후 self-attention 연산 수행 시 멀티모달 정보가 반영된 Fused item representation은 Q, K의 입력으로 사용하고, 순차 정보만 담겨있던 Pure item representation은 V의 입력으로 아이템의 정보는 유지한 채 멀티모달 정보를 효과적 학습할 수 있도록 한다. 아마존에서 제공하는 데이터 셋을 통해 Hit@10과 NDCG@10의 성능을 비교해본 결과 ‘Luxury Beauty’에서 10.52%, 3.83%, ‘Sports and Outdoors’에서 0.65%, 4.9% 증가하였다. 이를 통해 시퀀셜 추천시스템에 이미지와 텍스트 같은 멀티모달 정보는 유의미한 것을 확인하였다. 
