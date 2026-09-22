# Full-Qwen teacher qualification and representation distillation

This experiment tests whether the full pinned Qwen3.5-4B model is a useful teacher
before copying its decisions into the fast Qwen-vision/Laya policy. It does not
assume that good vision features imply correct instruction following, or that
distillation can preserve all of a larger model's reasoning ability.

## Measured result: teacher qualification failed

Completed locally on 2026-09-22, using 96 cases per teacher configuration:

| Measure | Stitched model | Full Qwen, no thinking | Full Qwen, bounded thinking |
|---|---:|---:|---:|
| Reviewed visual state, 24 cases | 41.7% | 91.7% | 87.5% |
| Conditional instruction, 48 cases | 79.2% | 62.5% | 60.4% |
| Both opposing goals correct, 24 pairs | 66.7% | 41.7% | 37.5% |
| Instruction answer-order consistency | 91.7% | 83.3% | 75.0% |
| Recorded next-button candidate, 24 cases | 37.5% | 50.0% | 50.0% |

Each teacher configuration had one invalid final response, included as a failure.
Samples are small and correlated within recording sessions. The four-way recorded
proxy has a 25% uniform chance level, but its three-example teacher advantage over
the student is not sufficient evidence of reliable gameplay. Different student
checkpoints are used for instruction and recorded-action tasks, as detailed below.

The teacher's scene assessment sometimes changes when the goal changes, even for
the same image pair. Better standalone visual-state recognition does not establish
better goal-conditioned action selection. Neither teacher configuration passes the
registered requirements. Both export attempts were rejected before any training
target directory was created. No transfer training ran, no checkpoint was replaced
and no game inputs were sent. No conclusion is made about larger Qwen variants or
longer reasoning budgets.

The feature exporter and training recipe below are implemented with unit coverage
for gradient flow, frozen weights, named targets and rejection checks. Real-model
feature export and distillation remain unvalidated because qualification failed.
The next capability experiment needs a teacher that passes the same checks, or
verified expert decisions, before a matched distillation/control training run is
justified. Adding its current labels would not solve the instruction failure.

[Machine-readable results](teacher-audit-001.json). Complete predictions and
generated responses are retained locally in `artifacts/teacher-audit-001`.

## What is implemented

`laya_vision_stitch.teacher_audit` prepares a reproducible, label-isolated audit:

- Twelve reviewed scenes from each of training and development, two per menu
  state per game, covering Grounded, Minecraft and Raft. Complete recording
  sessions and all image hashes are separated between the two splits.
- Per scene, two visual-state questions with reversed answer order and four
  instruction cases covering both opposing goals and both answer orders.
- Twenty-four development cases asking which of four candidate button sets
  matches the player's next recorded controls. Three distractors come from the
  same game's training sessions; the correct answer position is balanced. Recent
  recorded controls are supplied to both teacher and student for this proxy.

The menu state was previously reviewed by the assistant, not annotated by an
expert player. Instructions are synthetic. Recorded next-key agreement is an
ambiguous behavior-matching proxy, not action optimality or task success. None
of these cases demonstrates Hordes combat or loot collection. The development
data was used in earlier experiments and is now used for teacher selection;
reserved wording and withheld games are not consumed.

The teacher receives only pixels, goals, control descriptions, previous actions
and named choices. Review labels, future actions, answers and provenance do not
enter its prompt. It uses the full language decoder plus vision tower, at image
width 640. The student retains its configured image width 320; the comparison
therefore tests this teacher recipe, not an isolated language-decoder ablation.
The instruction baseline is `robust-decoder-007`; the recorded-action baseline is
`gameplay-buttons-002`, the strongest small training-set button fit. State
predictions use the student's choice head; instruction predictions use its actual
button outputs. Recorded candidates are ranked by the button head's independent
Bernoulli log likelihood. These protocols differ from unconstrained live actions.

Teacher variants generate a brief evidence statement and final option, with
thinking disabled or enabled. The bounded thinking recipe reserves 128 of 384
output tokens for an answer and caps thinking at 256 tokens. This is not an
exhaustive test of Qwen with unrestricted reasoning. Invalid final answers count
as failures. An earlier interrupted 512-token run without a separate thinking
budget is retained locally as incomplete evidence, excluded from summary scores.

## Qualification

Thresholds are registered before scoring: state accuracy at least 90%, instruction
accuracy at least 85%, both opposing goals correct at least 80%, option-order
consistency at least 90% for both tasks, and an instruction advantage of at least
10 percentage points over the student. Every condition must pass before menu
distillation. Next-key agreement never qualifies gameplay supervision.

The export tool recomputes eligibility from all predictions rather than trusting
a saved boolean. Hash checks bind manifests, screenshots, teacher targets and
feature files. Training uses training scenes only. It retains whole paired scenes
whose teacher answers agree with the checked labels, with at least eight scenes
and both visual states. Incorrect responses remain in the audit evidence.

## Learned transfer path

`laya_vision_stitch.reasoning_transfer` implements:

1. Full Qwen processes a training clip and generates its own evidence/reasoning.
2. A multimodal prefill over that generated text, before the final answer letter,
   exports the final Qwen language feature and named-choice probabilities. The
   answer must agree with the checked generated answer. Human labels are not
   inserted into the teacher context.
3. A training-only linear projection maps pooled Laya encoder features into the
   teacher's feature space. Cosine loss provides representation supervision;
   state-choice KL and soft button targets provide decision supervision alongside
   the checked labels. Teacher text/features are never student inputs.
4. Only connector, small Laya LoRA adapters, relevant action-head parameters and
   the training projection can update. Hashes verify that original vision and
   Laya weights remain unchanged.
5. Export saves the existing policy without the teacher or projection. It checks
   prediction parity after reload. An equal-step supervised control omits teacher
   probabilities and feature alignment to test whether distillation adds value.

This is a bounded research recipe. Matching one feature vector is not equivalent
to reproducing Qwen's complete computation. The current pilot covers menu-state
reasoning; broad action/goal distillation still needs verified demonstrations and
held-out gameplay outcomes. No experimental policy is automatically deployed.

```text
Offline only:
pixels + goal -> full Qwen -> checked decisions + language feature
                                      |             |
                                decision loss   feature loss
                                      |             |
pixels + goal -> vision -> connector -> Laya -> action outputs
                                         |
                              training-only projection

Deployment: vision -> connector -> Laya -> actions
            (same existing policy graph; projection omitted)
```

## Reproduce locally

The pinned models and source datasets must already be present. From the repository:

```sh
.venv/bin/python -m laya_vision_stitch.teacher_audit prepare
.venv/bin/python -m laya_vision_stitch.teacher_audit student
.venv/bin/python -m laya_vision_stitch.teacher_audit teacher
.venv/bin/python -m laya_vision_stitch.teacher_audit report
.venv/bin/python -m laya_vision_stitch.teacher_audit teacher --thinking --max-tokens 384
.venv/bin/python -m laya_vision_stitch.teacher_audit report --thinking
```

Commands reject existing prediction outputs. Use a fresh `--data` directory to
repeat the entire experiment. Only after the selected teacher qualifies:

```sh
.venv/bin/python -m laya_vision_stitch.teacher_audit teacher --split train --thinking --max-tokens 384
.venv/bin/python -m laya_vision_stitch.reasoning_transfer export --thinking
.venv/bin/python -m laya_vision_stitch.reasoning_transfer train --output artifacts/reasoning-distilled-001
.venv/bin/python -m laya_vision_stitch.reasoning_transfer train --control --output artifacts/reasoning-control-001
```

The same `--thinking` selection must be used for qualification and training-label
generation. The default transfer budget is 400 updates, learning rate 0.00003,
seed 53. These are experimental settings, not validated optimal hyperparameters.
