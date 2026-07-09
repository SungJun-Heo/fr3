# Task-space(Cartesian) IK 제어: 핵심 문제와 해결법 — 스터디 노트

> 목적: `writeOnce(CartesianPose)` 뒤에 들어갈 **IK 기반 Cartesian 제어**를 만들 때
> 마주치는 두 가지 고전 문제와 그 해결법을 공부하기 위한 노트.
> real(libfranka)엔 IK가 없어서(펌웨어가 담당) **sim에선 우리가 직접 구현**해야 한다.

---

## 0. 맥락: 우리가 만들려는 것

**resolved-rate motion control** (= Jacobian/미분 IK 기반 Cartesian 제어):

```
매 tick:  Cartesian setpoint(목표 EE pose) ──IK──▶ joint 명령 ──▶ 로봇/sim step
```

- setpoint 소스는 무엇이든 됨: 직선 궤적, 원, **VLA delta**, teleop.
- 컨트롤러는 "이번 tick의 pose 하나"를 관절로 바꿔줄 뿐. **경로(trajectory)는 setpoint 스트림 자체.**
- 이 방식이 필요한 이유: **직선 이동·원·task-space 경로 추종**처럼 EE가 지나는 *경로*를 제어해야 할 때.
  (경로 무관하게 "그냥 거기 가면 됨"이면 joint-space 보간이 더 싸고 안전 — 아래 5.4 참고.)

libfranka는 IK·특이점 처리를 **제공하지 않음**. `franka::Model`로 FK·자코비안·동역학과
`limitRate` 헬퍼만 줌. Cartesian→joint 변환은 **로봇 펌웨어(블랙박스)**가 하고, 한계 위반은
**대부분 fault(에러)로** 처리한다. → sim에선 이 부분을 우리가 만든다.

---

## 1. 먼저 알아야 할 기초 개념

- **자코비안 J(q)** (6×7 for FR3): 관절속도 → EE 속도(twist).
  `ẋ = J(q) · q̇`   (ẋ = EE 6D twist[선속도3+각속도3], q̇ = 관절속도 7D)
- **미분(differential) IK**: 위 식을 역으로. `q̇ = J⁺ · ẋ`
- **의사역행렬(pseudoinverse) J⁺**: J가 정사각이 아니라(6×7) 역행렬이 없음.
  - 여유(redundant)일 때 최소노름 해: `J⁺ = Jᵀ(JJᵀ)⁻¹` (right pseudoinverse)
- **여유자유도(redundancy)**: FR3는 7관절, 6D task → **1자유도 남음**.
  → EE를 안 움직이고도 관절을 바꿀 여지가 있음 (null space).
- **특이점(singularity)**: 어떤 자세에서 J가 rank를 잃음(6→5 이하).
  그 방향으로 EE를 움직이려면 관절속도가 "무한대"가 필요 → J⁺ 폭발.
- **manipulability(조작성) w**: 특이점 근접도 척도. `w = √det(J·Jᵀ)`. w→0이면 특이점.

**한 스텝 업데이트**: `q_{k+1} = q_k + q̇ · Δt` (q̇를 적분).
**full IK(반복법, Newton)**: `q ← q + J⁺·(x_target − FK(q))` 를 수렴까지 반복.

---

## 2. 문제 ①: 매 tick IK 비용

**원인**: full IK를 매 tick(1kHz) 수렴까지 반복해서 풀면 비쌈.

**해결**
1. **미분 IK 한 스텝만**: setpoint가 직전과 가까우니 완전히 다시 풀 필요 없음.
   `q̇ = J⁺ dx` (작은 Cartesian 오차 dx) → **자코비안 1개 + 6×6 선형해 1개**. 마이크로초.
2. **Warm-start**: 굳이 반복 IK를 써도 직전 q로 초기화하면 1~2회에 수렴.
3. **해석적(closed-form) IK**: 6-DOF 구면손목 로봇은 즉시. (FR3는 7-DOF 여유라 보통 미분 IK)

**결론**: 단일 팔에선 **거의 문제 아님**. 다관절·다로봇·초고주파일 때만 신경.

**공부 키워드**: differential inverse kinematics, Jacobian pseudoinverse, resolved-rate control,
Newton-Raphson IK, warm start.

---

## 3. 문제 ②-a: 특이점(singularity)

**원인**: J가 rank 손실 → `JJᵀ`가 near-singular → `J⁺`의 값이 폭발 →
작은 Cartesian 움직임에도 관절속도가 폭주. (직관: 그 방향으로 못 움직이는 자세라 무리하게 밀어붙임)

**해결**
1. **Damped Least Squares (DLS / Levenberg–Marquardt)** ← 제일 기본:
   `q̇ = Jᵀ(JJᵀ + λ²I)⁻¹ · ẋ`
   - 감쇠 λ²가 특이점 근처에서 관절속도를 **유한하게** 억제. 정확도를 조금 희생, 안정성 획득.
2. **적응 감쇠(adaptive damping)**: 특이점 가까우면 λ↑, 멀면 λ→0(정확도 유지).
   예: `λ² = λ₀²(1 − w/w₀)²  (w < w₀일 때), 아니면 0`   (w = manipulability)
3. **출력 속도 클램핑**: 계산된 q̇를 관절 속도/가속 한계로 잘라냄. (= libfranka `limitRate` 개념)
4. **시간 스케일링(감속)**: 경로가 관절속도를 과하게 요구하면 **궤적 자체를 느리게** 재파라미터화.
   경로는 유지, 속도만 낮춤. ("path–velocity decoupling")
5. **계획 단계 회피**: 완전신전(팔꿈치 특이점)·손목축 정렬(손목 특이점) 지나는 경로를 애초에 배제.

**공부 키워드**: kinematic singularity, damped least squares, singularity-robust inverse,
manipulability ellipsoid, adaptive damping, SVD of Jacobian, time scaling / path parameterization.

---

## 4. 문제 ②-b: 관절 한계(joint limits)

**원인**: 지정한 Cartesian 경로가 어떤 관절이 범위를 넘도록 요구할 수 있음.

**해결**
1. **여유자유도 null-space 회피 (7-DOF의 핵심 강점)**:
   `q̇ = J⁺·ẋ + (I − J⁺J)·q̇₀`
   - 뒤 항 `(I − J⁺J)q̇₀`는 **EE pose를 바꾸지 않으면서** 관절만 움직임(null-space 투영).
   - `q̇₀ = −k·∇H(q)`로 두면, 관절-한계 회피 비용 H를 내리는 쪽으로 관절을 재배치.
     예: `H(q) = Σ ((q_i − q_mid,i)/(q_max,i − q_min,i))²` → 중앙범위로 밀어냄.
   - 즉 **EE 경로는 그대로, 남는 1자유도로 한계를 피함.** manipulability 최대화·자세 유지도 동일 원리.
2. **가중 IK(Weighted DLS)**: 한계에 가까운 관절엔 큰 가중을 줘 덜 쓰게.
3. **QP 기반 IK**: IK를 "관절·속도 한계를 **제약(constraint)**으로 건 최적화(QP)"로 풀기.
   한계·특이점을 설계상 깔끔히 처리. (예: **TRAC-IK** = DLS + SQP로 지역최소 탈출·한계 준수)
4. **feasibility 사전검사**: 계획 단계에서 경로 전체의 도달성·한계·충돌을 먼저 확인.
5. **정말 불가능하면**: 안전하게 **감속/정지 또는 fault**. (real Franka가 하는 방식)

**공부 키워드**: redundancy resolution, null-space projection, task-priority / hierarchical IK,
joint-limit avoidance, weighted least squares, QP-based IK, TRAC-IK.

---

## 5. 실무에서 실제로 쓰는 조합 & 라이브러리

### 5.1 워크호스 조합 (스트리밍 Cartesian = VLA/teleop)
> **DLS(적응 감쇠) + null-space 관절한계 회피 + 출력 속도 클램핑 + 특이점 근처 자동 감속**

### 5.2 대표 라이브러리 (공부하며 참고)
- **MoveIt Servo** — 스트리밍 Cartesian 명령을 딱 이렇게 처리(특이점 접근 시 속도 스케일다운,
  한계 전 정지). "어떻게 실무에서 하나"의 정석 레퍼런스.
- **TRAC-IK / KDL** — IK solver(전자는 QP+DLS 하이브리드).
- **Pinocchio** — 강력한 동역학·미분 IK(QP) 라이브러리.
- **Drake** — differential IK를 QP+제약으로.
- **mink** — **MuJoCo용** 미분 IK QP 라이브러리(한계/특이점을 제약으로). ← 우리 환경에 바로 맞음.

### 5.3 real Franka(libfranka) 위치
- libfranka: FK·자코비안·동역학(`franka::Model`) + `limitRate` + 스트리밍만. **IK 없음.**
- Cartesian→joint는 **펌웨어**가 담당, 한계 위반은 **fault로** 처리(우아한 회피 아님).

### 5.4 언제 task-space가 아니라 joint-space를 쓰나 (트레이드오프)
| 방식 | 보장 | 비용/안전 | 용도 |
|---|---|---|---|
| **joint-space** (끝점 IK 1회 + joint 보간) | 끝점만, 경로는 곡선 | 싸고 특이점 안전 | 자유공간 이동, reset, "그냥 도달" |
| **task-space** (매 tick IK) | EE 경로 전체 | IK 매 tick + 특이점/한계 관리 필요 | 직선·원·**VLA delta** 추종 |

---

## 6. 우리 sim에 적용 (설계 방향)

1. **베이스라인**: DLS + 출력 클램핑. MuJoCo `mj_jacSite`로 자코비안 획득.
2. **개선**: null-space 관절한계 회피(FR3 7-DOF 활용).
3. **프로덕션급 원하면**: `mink`(QP)로 교체 — 인터페이스는 그대로.

### ⚠️ fidelity 주의점 (검증 프로젝트라서 중요)
우리 sim의 DLS IK는 감쇠·null-space로 **real 펌웨어보다 관대**할 수 있음.
→ sim이 슬쩍 통과시킨 특이점/한계 근접 경로가 **real에선 fault**날 수 있음.
→ 충실한 검증을 위해 sim도 **특이점(w<임계)/관절한계 근접 시 flag 또는 trip** 하게 만들어야 함.
이는 앞서 만든 **collision reflex와 같은 철학**: "real에서 실패할 동작을 sim에서 미리 드러낸다."

---

## 7. 공부 로드맵 (추천 순서)

1. **자코비안 & 미분 운동학**: ẋ = J q̇, geometric vs analytic Jacobian.
   - 교재: *Modern Robotics* (Lynch & Park) Ch.5–6, 또는 Siciliano *Robotics: Modelling* Ch.3.
2. **의사역행렬 & 여유자유도**: 최소노름 해, null-space.
3. **미분 IK / resolved-rate control**: `q̇ = J⁺ẋ`, 적분으로 추종.
4. **특이점 & DLS**: manipulability, `Jᵀ(JJᵀ+λ²I)⁻¹`, adaptive damping.
   - 원 논문 키워드: Nakamura & Hanafusa (1986), Wampler (1986) — singularity-robust IK.
5. **null-space 회피 & task-priority**: 관절한계·자세 유지, `(I−J⁺J)q̇₀`.
6. **QP 기반 IK**: 한계를 제약으로. TRAC-IK / Drake / mink 코드 읽기.
7. **실무 통합**: MoveIt Servo 문서/코드로 "다 합치면 어떻게 되나" 확인.
8. **적용**: 우리 sim의 `robot/` 에 DLS solver 구현 → FK로 왕복 오차 검증 → goto-pose → 스트리밍.

### 빠른 검색어 모음
`differential inverse kinematics`, `resolved-rate control`, `damped least squares IK`,
`Jacobian pseudoinverse null space`, `manipulability singularity`, `redundancy resolution`,
`joint limit avoidance null space`, `TRAC-IK`, `MoveIt Servo`, `mink mujoco IK`,
`Modern Robotics Lynch Park`.

---

## 8. 한 줄 요약

- **문제①(비용)**: 미분 IK 한 스텝이면 단일 팔엔 사실상 무료.
- **문제②(특이점/한계)**: **DLS(감쇠) + 속도 클램핑 + null-space 한계회피 + 감속**,
  그래도 안 되면 **감속/정지**. QP-IK가 이걸 제약으로 한 번에 처리.
- **우리 sim**: DLS+클램프 베이스라인 → null-space 추가 → **real처럼 특이점/한계 trip 감지**(fidelity).
