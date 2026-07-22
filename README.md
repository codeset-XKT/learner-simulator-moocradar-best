# Learner Simulator

## Current locked baseline

The current code has been fixed as
`baseline-2026-07-04-foundationalassist-calibrated-v1`.

Before changing prompt logic, profile modules, simulator decision logic, or
ablation wiring, read `VERSION_LOCK.md`. It records the current Full simulator,
known limitations, reproduction command, and latest FOUNDATIONALASSIST 10x10
Full vs `w/o ability profile` result.

The historical MoocRadar good-result run is fixed separately as
`baseline-2026-07-04-moocradar-ability-summary-no-irt-v1`; see
`VERSION_LOCK_MOOCRADAR.md`. That lock preserves the MoocRadar 10x10 ability
summary result and notes that later prompt-calibration edits changed the active
source state.

## Cognitive strategy controller

The LLM no longer receives an unrestricted instruction to solve first and then
imitate a learner. Before each target exercise, a non-LLM controller selects one
of six cognitive modes:

```text
mastered / partial / misconception / careless / guessing / unknown
```

The selection uses current concept mastery, related historical outcomes,
learner ability, attention, fatigue, carelessness, and guessing tendency. It
outputs only process constraints:

- available knowledge;
- required approach;
- learner-level reasoning budget;
- whether verification is allowed;
- how any error must arise.

It never reads or emits the target response, reference answer, reference
analysis, `p_correct`, or a sampled correct/incorrect label. Full Four-tier and
answer-only simulation both follow the same preselected strategy.

## Experiment suites

Formal comparison and ablation entry points are separated from the core simulator:

```text
experiments/
  common.py                         # shared fixed-cohort runner and report format
  comparison/
    run_comparison.py               # full vs Agent4Edu vs random
    run_agent4edu.py                # isolated Agent4Edu reproduction
    run_random.py                   # random/statistical baseline
  ablation/
    run_ablation.py                 # key-module ablations
```

Comparison:

```powershell
python experiments/comparison/run_comparison.py --source-rows 1000 --max-users 3 --progress --save-steps
```

The Agent4Edu reproduction follows its official Task1-Task4 action prompt, Task4 response prediction, reflection and forgetting flow. Its reference answer and analysis exposure are preserved and explicitly marked in the report. The original DNeuralCDM proficiency input is replaced by this project's dynamic mastery state.

Ablation:

```powershell
python experiments/ablation/run_ablation.py --source-rows 1000 --max-users 3 --progress --save-steps
```

The four key ablations are `no-profile`, `no-memory`, `no-proficiency`, and `no-four-tier`. All variants use the same UIDs, 90 observed interactions, and 10 target interactions.

基于 XES3G5M 的学习者智能体模拟项目。当前实验流程为：

```text
90条真实历史
  -> Profile / IRT / mastery / short-long memory
  -> 后10题顺序模拟
  -> StudentAnswer 外部判分
  -> 与真实学生响应比较
```

`p_correct` 只作为统计基线，不直接决定 LLM 的答案。LLM 自主生成四层学习者响应：

```text
StudentAnswer
AnswerConfidence
StudentReasoning
ReasoningConfidence
```

答案正确性由程序与题目 metadata 中的标准答案比对，不再由 LLM 自己输出 Task4。

## 数据集

默认数据集：

```text
E:/yyx/KT数据集/XES3G5M/XES3G5M
```

主要文件：

```text
kc_level/train_valid_sequences.csv
kc_level/test.csv
metadata/questions.json
metadata/kc_routes_map.json
metadata/embeddings/
metadata/images/
```

## Agent4Edu 实验范式

项目只支持以下正式实验范式：

1. 合并同一 UID 的多条序列，并按时间排序。
2. 仅保留至少具有100条有效交互的学生。
3. 每名学生前90条作为真实历史。
4. 紧接着的10条作为模拟目标。
5. 90条历史同时用于 Profile、IRT、mastery 和 Memory 初始化。
6. 后10题按顺序模拟，每一步的模拟结果用于更新后续状态。

随机模拟基线：

```powershell
python scripts/evaluate.py --simulator random --source-rows 1000 --max-users 3 --save-steps
```

LLM 四层模拟：

```powershell
$env:DASHSCOPE_API_KEY=(Get-Content -LiteralPath "E:\yyx\8 Learner Simulator\key.txt" -Raw).Trim()
python scripts/evaluate.py --simulator llm --source-rows 1000 --max-users 3 --call-api --progress --include-prompt --save-steps
```

`w/o Four-tier` 消融只保留 `StudentAnswer`，其余数据、状态和外部判分流程不变：

```powershell
python scripts/evaluate.py --simulator llm --ablation no-four-tier --source-rows 1000 --max-users 3 --call-api --progress --include-prompt --save-steps
```

参数说明：

- `--max-users`：唯一模拟学生数量。
- `--source-rows`：聚合学生前最多扫描的源序列行数。
- 历史长度固定为90，目标长度固定为10，不提供旧切分开关。

查看单个四层 prompt：

```powershell
python scripts/llm_dry_run.py
```

导出完整模拟记录：

```powershell
python scripts/export_records.py --simulator random --source-rows 1000 --max-users 50
```

## 项目结构

```text
scripts/evaluate.py                         # 主评估入口
scripts/export_records.py                   # 导出逐步模拟记录
scripts/llm_dry_run.py                      # 渲染或调用单个 LLM prompt
scripts/test_agent4edu_protocol.py          # 90+10范式回归测试
scripts/test_four_tier.py                   # 四层响应模块测试
src/learner_simulator/data.py               # 数据读取、UID聚合与切分
src/learner_simulator/profile.py            # 学习者画像
src/learner_simulator/irt.py                # Rasch/1PL IRT
src/learner_simulator/memory.py             # 真实历史和模拟历史记忆
src/learner_simulator/behavior.py           # 非认知状态
src/learner_simulator/agent4edu_prompt.py   # LLM prompt
src/learner_simulator/four_tier.py          # 四层响应解析与诊断
src/learner_simulator/evaluation.py         # 评估指标
src/learner_simulator/simulators/           # 随机与 LLM 模拟器
```

## 关键字段

- `source=observed_history`：用于初始化的真实历史。
- `source=simulated`：后10题产生的模拟记录。
- `p_cognitive`：用户、题目、知识点、mastery 和 IRT 融合概率。
- `p_correct`：加入注意力、疲劳、粗心和猜测因素后的统计基线。
- `statistical_sampled_response`：按 `p_correct` 采样的随机基线。
- `simulated_response`：随机采样结果或 LLM 的 `StudentAnswer` 外部判分结果。
- `four_tier_assessment`：答案判分、两类置信度和诊断类别。

## 评估指标

- `prob_acc_at_threshold` / `prob_f1_at_threshold`：`p_correct >= threshold` 基线。
- `sample_match_acc` / `sample_f1`：最终模拟响应与真实响应的一致性。
- `llm_response_acc` / `llm_response_f1`：可外部判分的 LLM 答案指标。
- `four_tier_mean_answer_confidence`：平均答案置信度。
- `four_tier_mean_reasoning_confidence`：平均理由置信度。
- `four_tier_answer_scored_count`：可完成答案外部判分的响应数。
- `four_tier_fully_scored_count`：答案和理由均完成独立判分的响应数。

XES3G5M 没有真实学生理由和置信度，因此当前只客观判定答案层。理由层保留可插拔 evaluator 接口。

## 测试

```powershell
python scripts/test_agent4edu_protocol.py
python scripts/test_four_tier.py
```

## API 配置

`configs/llm.example.json` 使用 OpenAI-compatible 接口。真实 API key 应通过环境变量或项目外部 `key.txt` 读取，不要写入配置文件。
