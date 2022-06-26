# a3c-mujoco-pytorch

1. 실행
```python
python main.py
```

2. config에서 hyper param 수정



---

- [x] env: mujoco
- [x] model
- [x] simulate
- [x] test
- [x] train: 
  - [x] loss에 entropy항 추가해서 sigma 작게 나올 수 있게 함.
  - [x] train시 lock이 있을 필요가 없음. - mutex 쓰면 느려지고, 에이전트들도 가끔씩 업데이트 하기 때문에 겹칠일 없고 겹치더라도 크게 영향 주진 않음.
- [x] logging(wandb)
  - [ ] model 보여주기
  - [ ] cfg 보여주기

---
- 스코어 비교하는 그래프 넣기
- 하이퍼파라미터 꼭 적기 (optimization 포함)
- input 처리 내용 반드시 넣기
- input state 몇개가지고 하는지 state representation 명시
- network 명시
- output action - 액션 2번 줬는지 3번 줬는지 명시, policy gradient 쓴 사람은 확률 어떻게 했는지 명시
- 몇번 step 하고 global update 했는지 설명
- agent 16개 

---
- 멀티 프로세싱을 위해 참고

  - https://pytorch.org/docs/stable/notes/multiprocessing.html

- 참고 깃허브 

  - https://github.com/ikostrikov/pytorch-a3c.git

  - https://github.com/andrewliao11/pytorch-a3c-mujoco.git

  - https://github.com/MorvanZhou/pytorch-A3C.git
