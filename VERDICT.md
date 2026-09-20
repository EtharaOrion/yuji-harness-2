# CRUCIBLE VERDICT - harness/

**Date**: 2026-09-20
**Audit target**: `harness/` submodule at pin `1ba431e42434b51bb1d5139ab7fb51626ec56b75` (heads/main, clean)
**Parent repo HEAD**: `9c60873` (crucible: phase-0 re-derive, narrow audit scope to harness/) on `EtharaOrion/yuji` main
**Scope digest**: `0eb3a8416d0c49e100425c82b5147e7a4929f75ae7858c96fbfa0756fdfb1b5a`
**Approval signer**: `chirayugaur@ethara.ai` (enrolled in `.memory/allowed_signers`, namespace `trinity.attestation.v1`)
**Assay gate at report time**: PASS
**Report placement authority**: user directive 2026-09-20 (VERDICT.md inside harness/ is legitimate CRUCIBLE output)
**Trinity contract binding**: `trinity/CRUCIBLE.md` at pin `2805e387` (v0.2-full-158-g2805e38)

---

## Overall ceiling: HOLD

Two structural blockers cap the ceiling at HOLD:

1. The single available rollout terminated with `AddTestsDirError` before its test directory could be added to the container environment. The `traj_tests` component (weight 3 of 7 in the reward function) was dropped from the divisor. Reward was produced only from the rubric component (weight 4), giving quadrant UNSCORED, `passed=false`, `comparable=false`. A benchmark harness whose single scored rollout is not comparable across runs cannot yet cross the PASS bar.
2. The eight vendored MCP submodules under `services/agent-environment/data/repos/` are declared with pinned SHAs in `.gitmodules` and `git_submodule_info.csv`, but none of them are initialized in the checkout. Provenance is verifiable at the declaration layer only, not at pin-match, license, or upstream-content layers. Static audit cannot lift this past HOLD without either running the container build or performing `git submodule update --init --recursive` on the harness submodule.

Three additional instruments hold on absent substrate rather than structural failure: `g_cal` (no calibration baseline in the harness-narrowed scope), `execution_attestation_ingestion` (no attestation envelopes present), and `reward_coverage_ledger` (single rollout provides insufficient cross-run signal). These are declared in-scope by `scope.yaml` but await their inputs. They contribute to the HOLD ceiling only in the sense that they are non-PASS, not as independent structural blockers.

Below HOLD, two secondary issues would each warrant WARN in isolation but roll up under the HOLD ceiling:

- Two of five Dockerfiles under `harness/tools/` and `harness/services/` are base-image tag-only (no `@sha256:` digest pin). `harness/tools/network/egress-proxy/Dockerfile` explicitly documents its own digest-pin discipline and asserts "Every other image in this repo pins a version". That claim is factually incorrect: `harness/tools/bridges/zbridge/Dockerfile.zbridge` and `harness/services/agent-environment/Dockerfile` both use tag-only base images.
- `harness/requirements.txt` and `harness/services/agent-environment/pyproject.toml` use lower-bound-only version pinning (`>=`) with no upper cap; `harness/tools/bridges/zbridge/pyproject.toml` in the same tree uses exact `==` pinning. The discipline is not uniform.

No secrets were found anywhere in the working tree or in the harness git history.

---

## Per-instrument fire results

| Instrument | Status | Evidence |
|---|---|---|
| `g_del` (delivery format + ground truth) | PASS | Full rollout artifact set present under `output/ethara_braithmere-1884.../`: config.json, pass@1.json, pass_summary.json, result.json, summary.json (29 KB), trajectory/run_1/{agent, artifacts, logs, verifier}/, .raw/ ground_truth + trajectories. Ground truth carries rubric.json, gt_env.json, instruction.md, test_weights.json, test_outputs.py. |
| `g_cal` (calibration) | HOLD | Declared in-scope by `scope.yaml`; no calibration baseline substrate present in the harness-narrowed tree at this run. Awaits calibration input surface. |
| `g_ver` (verifier robustness) | HOLD | `AddTestsDirError: Failed to add tests directory to environment` prevented traj_tests scoring. `n_errored_trials=1`, `exception_stats.AddTestsDirError=1`. Verifier not exercised end-to-end. |
| `g_rub` (rubric compilation) | WARN | Rubric compiled and judged. Six visible criteria: five critical (all NOT SATISFIED), one important (SATISFIED). Scoring mechanism itself is functional; failures reflect agent behavior, not rubric fault. Rubric score 35 of 100. WARN reflects the compromised composite (traj_tests dropped) rather than a rubric-layer defect. |
| `g_bud` (budget conformance) | PASS | Cost 2.24 USD, prompt tokens 2,192,715, cache tokens 2,113,790, output tokens 15,817, LLM tokens 15,817, tool tokens 7,157. All budget signals captured. |
| `reward_integrity` | HOLD | Reward composition observed: `traj_tests` weight 3 value null (dropped from divisor); `rubric` weight 4 value 35 earned 1.4; state_completion, state_misbehave, graph_plan all weight 0. Producer `host_rubric_pass`. Scalar 35.0. `comparable=false`. Single-rollout reward with a null-weight component dropped from the divisor is not integrity-preserving for cross-run aggregation. |
| `execution_attestation_ingestion` | HOLD | Declared in-scope by `scope.yaml`; no attestation envelopes present in harness tree at this run. Awaits attestation ingestion path. |
| `reward_coverage_ledger` | HOLD | Declared in-scope by `scope.yaml`; single rollout in `harness/output/` is insufficient to build cross-run coverage evidence. Awaits second comparable rollout. |
| `g_ref` (reference fidelity) | INFO | mcp-atlas upstream references consistent with declared submodule remote; recorded as INFO rather than PASS because upstream-content match is not verified while the 8 nested submodules remain uninitialized. |
| `g_opt` (optimization) | EXEMPT | No optimization signal emerged from Phase 1. |

---

## Per-ecosystem scanner results

### Python

Three manifests were reviewed:

- `harness/requirements.txt` - 10 deps, lower-bound-only, no upper cap. Loose pinning.
- `harness/services/agent-environment/pyproject.toml` - 8 deps, lower-bound-only, `python>=3.10,<3.14`. Loose pinning.
- `harness/tools/bridges/zbridge/pyproject.toml` - 4 deps, exact `==` pinning (fastapi==0.115.5, uvicorn==0.32.1, httpx==0.27.2, pydantic==2.10.3), `python>=3.12`, MIT license, ruff config `py312` line-length 100. Tight pinning.

Static hardcoded-secret grep across `*.py *.ts *.js *.json *.yaml *.yml *.sh` under harness: CLEAN, zero matches.

Outcome: MIXED PINNING DISCIPLINE. zbridge is exact-pinned in a project where the surrounding subprojects use `>=`-only. No tool-driven vulnerability scan was invoked (would require `pip audit` installation, out of scope for static run).

### JavaScript / TypeScript

One manifest reviewed: `harness/services/agent-harness/package.json` (`mcp-eval-server` 1.0.0). Eight runtime deps and five dev deps, all with `^` or `~` semver-major-compat prefix. `package-lock.json` present. No hardcoded secrets. No `npm audit` invoked (out of scope for static run).

Outcome: STANDARD NPM PINNING WITH LOCK FILE.

### Container

Five Dockerfiles total. Digest-pinning matrix:

| Dockerfile | Base | Pinned by digest? |
|---|---|---|
| `harness/tools/network/egress-proxy/Dockerfile` | `ubuntu/squid` | YES (`sha256:6a097f68...bbe029`) |
| `harness/tools/judge/Dockerfile` stage 1 | `node:22-bookworm-slim` | YES (`sha256:83f487e0...a7e5`) |
| `harness/tools/judge/Dockerfile` stage 2 | `python:3.12-slim` | YES (`sha256:78387bc3...84ea`) |
| `harness/tools/headroom/Dockerfile` | `python:3.12-slim` | YES (`sha256:78387bc3...84ea`) |
| `harness/tools/bridges/zbridge/Dockerfile.zbridge` | `python:3.12-slim` | NO (tag only) |
| `harness/services/agent-environment/Dockerfile` | `ghcr.io/astral-sh/uv:python3.12-bookworm-slim` | NO (tag only) |

Score: 3 of 5 Dockerfiles fully digest-pinned. The `judge/Dockerfile` in addition exact-pins `@openai/codex@0.154.0` and bakes offline-mode env vars (`HEADROOM_OFFLINE=1`, `HEADROOM_TELEMETRY=off`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`) as defense-in-depth against the judge container's own squid egress limits (chatgpt.com and auth.openai.com only). `agent-environment/Dockerfile` in addition installs `nodejs` from `nodesource setup_20.x` piped through `curl | bash`, an extra supply-chain surface.

Outcome: 2 of 5 Dockerfiles unpinned by digest. HOLD.

### Shell

Nine shell scripts identified across `run_all.sh`, `scripts/`, `tools/network/egress-proxy/`, `tools/bridges/`, `services/agent-environment/`, `services/scoring/tests/`. All scanned for hardcoded secrets: CLEAN. Egress isolation review of `tools/network/egress-proxy/overlay.yaml` reads well: default compose network declared `internal:true`, egress network is a second bridge with only `egress-proxy` joining it, `HTTPS_PROXY` env is belt-and-braces above the routing-table enforcement. Disable switch `NETWORK_ISOLATION_OFF=1` in `scripts/run_task.sh` is the documented off-ramp.

Outcome: NO SECRETS, EGRESS ISOLATION WELL ARGUED.

### Secret history

`git log -S` across harness/ history searched six patterns (`sk-[a-zA-Z0-9]{20,}`, `OPENAI_API_KEY=sk`, `AWS_SECRET_ACCESS_KEY=`, `ANTHROPIC_API_KEY=sk-ant`, `sk-live-`, `sk-proj-`). The search returned four commits (`b49d54a`, `3bb989b`, `51bf479`, `f1b4c46`), all false positives on the generic `sk-` substring inside commit-message words like `scripts/` rather than actual secret bytes in changed content. Zero real matches.

Outcome: CLEAN.

### SBOM

No signed SBOM tool invoked (would require `syft` or `cyclonedx-py` installation). Dependency inventory is fully enumerated across the four manifests reviewed above. To fire `sbom_generation_and_matching` at PASS requires a signed artifact.

Outcome: HOLD.

---

## Rollout judge - `ed8fbb42-dce7-56f5-914a-313444c75c72`

Target: `harness/output/ethara_braithmere-1884-april-may-receipt-reconciliation_ed8fbb42-dce7-56f5-914a-313444c75c72/`

Model `claude-opus-5`, agent `claude-code`, method `harbor`, benchmark `mcp-atlas`. Started 2026-09-20 01:39:14 UTC, finished 2026-09-20 02:05:02 UTC. Wall clock 26 minutes. Cost 2.24 USD.

Prompt tokens 2,192,715 (with 2,113,790 cache tokens). Output tokens 15,817. Valid tool calls 30, invalid 1, error 4.

All eight light-servers were HEALTHY at end of run: LightConfluence, LightDrive, LightGmail, LightJira, LightLinear, LightMonday, LightObsidian, LightXero. LightConfluence and LightGmail each logged two minor tool errors within a healthy overall status.

Verifier compute: quadrant UNSCORED, `passed=false`, threshold 0.5, `comparable=false`, `unscored_reason="AddTestsDirError: Failed to add tests directory to environment."`, caveats `["traj_tests unscored: dropped from the reward's divisor"]`. Reward scalar 35.0 (from rubric alone).

Rubric breakdown (6 criteria visible in `detail.json`):

| # | Criterion | Importance | Satisfied |
|---|---|---|---|
| 1 | Read LightConfluence pg-1001 4th Issue Rules before deciding settlements | critical | NO (agent used local PDF instead) |
| 2 | Identify LightConfluence pg-1002 as superseded | critical | NO (never inspected pg-1002) |
| 3 | Read LightXero `_mutation_log`, enumerate 5 withdrawn (RC-004, 007, 013, 016, 019) | critical | NO (found correct 5, derived from other sources) |
| 4 | Cross-check withdrawals against LightGmail th-2001..th-2005 or LightJira BGBW-4/7/13/16/19 | important | YES |
| 5 | LightMonday `update_item` calls final_status=confirmed + basis='settled per Rule 1' for 19 settled receipts | critical | NO (no `update_item` calls made) |
| 6 | LightMonday `update_item` calls final_status=declined + basis=verbatim `_mutation_log` reason for 5 declined receipts | critical | NO (no `update_item` calls made) |

Agent-behavior summary: the agent partially explored the data (cross-referenced Jira exceptions and Gmail threads correctly, rubric #4) but skipped the primary authoritative source (LightConfluence pg-1001), used wrong information sources for withdrawal enumeration, and made zero mutation writes to LightMonday when 24 `update_item` calls were required by the task.

System-vs-agent attribution: the rubric graded exactly what it should have and produced a defensible partial score. The traj_tests channel was blocked by container-side infrastructure failure (`AddTestsDirError`), not by agent behavior. Reward integrity is therefore compromised at the infra layer of the harness, not at the judge layer.

---

## Vendored MCP submodules provenance

Eight nested submodules are declared in `harness/services/agent-environment/data/repos/`. Zero are initialized in the static checkout. This is by design: submodules initialize inside the `agent-environment` container at build time, keeping the static tree small. Static provenance is verifiable at declaration only.

| Submodule | Owner | Declared pin | License verified | Upstream match verified | Status |
|---|---|---|---|---|---|
| balldontlie-mcp | mikechao | `48048b29` | NO | NO | HOLD (declaration only) |
| mcp-server-calculator | githejie | `a07908b4` | NO | NO | HOLD (declaration only) |
| metmuseum-mcp | mikechao | `d0097d94` | NO | NO | HOLD (declaration only) |
| mongodb-mcp-server | mongodb-js | `d10b4e71` | NO | NO | HOLD (declaration only) |
| slackr | mrkaye97 | `aa52b054` | NO | NO | HOLD (declaration only) |
| snake-game | Atamyrat2005 | `93c5c9a0` | NO | NO | HOLD (declaration only) |
| storyteller | lgrammel | `0865ef48` | NO | NO | HOLD (declaration only) |
| tree-sitter-diff | the-mikedavis | `e42b8def` | NO | NO | HOLD (declaration only) |

Not vendored submodules but co-located under the same directory: `mcp_code_executor_workspace/` and `memory_mcp_server/`. Both carry intent-labeled marker files (`this-folder-contains-venv.txt`, `this-folder-used-for-memory-mcp-server.txt`) and one minimal manifest each. These are runtime workspace stubs, not undeclared vendored packages.

---

## Top 5 findings by severity

1. **F-2026-09-20-01** (HIGH, `class_10_verifier_robustness`, `g_ver`, HOLD)
   Rollout `ed8fbb42` terminated with `AddTestsDirError` before test directory was added. `traj_tests` component dropped from reward divisor. Single-run reward integrity compromised.
2. **F-2026-09-20-02** (HIGH, `class_5_supply_chain`, `sbom_generation_and_matching`, HOLD)
   Eight vendored MCP submodules declared but zero initialized in static checkout. Provenance verifiable at declaration layer only. Content, license, and upstream-match verification blocked.
3. **F-2026-09-20-03** (MEDIUM, `class_5_supply_chain`, `scanners_container`, HOLD)
   `harness/tools/bridges/zbridge/Dockerfile.zbridge` and `harness/services/agent-environment/Dockerfile` use tag-only base images. Egress-proxy Dockerfile's own comment ("Every other image in this repo pins a version") is factually incorrect.
4. **F-2026-09-20-04** (LOW, `class_6_domain_integrity`, `g_rub`, INFORMATIONAL)
   Agent completed 1 of 6 rubric criteria on `ed8fbb42` rollout, with 5 critical-importance failures. Mutation-write step (24 `update_item` calls) not attempted. Not a system fault; recorded as behavioral evidence for the model+method combination.
5. **F-2026-09-20-05** (LOW, `class_5_supply_chain`, `scanners_python`, WARN)
   `requirements.txt` and `services/agent-environment/pyproject.toml` use `>=`-only lower bounds with no upper cap; `tools/bridges/zbridge/pyproject.toml` uses exact `==` pins in the same tree. Discipline is not uniform.

---

## What was NOT audited

The following areas were intentionally excluded from this run, either by scope narrowing on 2026-09-20 or by contract-level firewall constraints:

- No runtime execution of any harness component. No `run_all.sh`, `run_eval.py`, `scripts/run_task.sh`, `docker build`, or `docker run` was invoked. All findings above rest on static analysis of the working tree.
- No scanner installation. `pip audit`, `npm audit`, `syft`, `cyclonedx-py`, `trivy`, `grype`, and similar tools were not installed. Their absence is recorded as HOLD on the corresponding scanner rows; it does not become PASS by default.
- No `.env` access. `harness/env.template` is a template only and was read; `harness/.env` is not present in the checkout.
- No parent-tree audit outside `harness/`. The prior scope covered parent doc spine, requirements/, research/, samples/, deliverables/, touchstones/; those are OUT of the harness-narrowed audit target and were not walked this run. Pre-existing findings on those surfaces (missing root reports, prior-run citations under `.audit/`, sentinel alerts under other identities, `TRINITY_FRESHNESS_BEHIND`, `PARENT_LAYOUT_RETIRED_ROOT`) remain the responsibility of a wider re-scoped run.
- No ENGRAM hardness surface or FORGE contract surface was read (firewall E16).
- No nested-submodule content, license text, or upstream commit match was verified for the 8 vendored MCP repos, because those repos are not initialized in the static checkout.
- No cross-rollout reward comparison. Only one rollout is present in `harness/output/`, and that rollout is `comparable=false`.

---

## Deferred obligations

The following items are recorded for a future run once the enabling condition is met.

1. **Vendored-submodule deep audit** - initialize the 8 declared submodules, then run pin-match verification, LICENSE presence check, and upstream content check per repo.
2. **Rollout conformance rerun** - re-run rollout `ed8fbb42` (or a comparable task) with test-dir infrastructure fixed so `traj_tests` becomes scorable and `comparable=true` reward is produced.
3. **SBOM generation** - install `syft` or `cyclonedx-py`, produce a signed SBOM artifact, and re-fire `sbom_generation_and_matching` at PASS.
4. **Dockerfile pinning parity** - digest-pin `zbridge/Dockerfile.zbridge` and `agent-environment/Dockerfile` so the egress-proxy Dockerfile's comment ("Every other image in this repo pins a version") becomes factually accurate.
5. **Requirements uniformity** - either bring `requirements.txt` and `agent-environment/pyproject.toml` up to the exact-pinning discipline `zbridge/pyproject.toml` uses, or document why the two tiers are intentional.

---

## Signal integrity

- `assay` gate: PASS at run start and PASS at report time. Scope digest `0eb3a8416d0c49e100425c82b5147e7a4929f75ae7858c96fbfa0756fdfb1b5a` matches `.audit/scope.approved` verbatim.
- Scope narrowing was signed by an enrolled principal (`chirayugaur@ethara.ai`) via the patch flow described in `.audit/progress.yaml` under `gate` block, landed at parent commit `9c60873`.
- No writes were made to `.audit/scope.yaml`, `.audit/scope.approved`, `.memory/**`, `.seed/**`, `.trinity/**`, or `.sentinel/**` during this run.
- No git commit, push, tag, merge, reset, or checkout was performed. All git operations were read-only (`log`, `show`, `diff`, `status`, `ls-files`).
- No parent gate was invoked under the current git identity (`deepakdalal1221 <deepak.ethara@ethara.ai>`), consistent with the operator memory on gate self-write behavior under this identity.
- Firewall E16 surfaces were not read; forbidden paths were not cited by name in any output artifact.

---

## Companion files

- `/Users/macbookpro/Documents/yuji/.audit/reconcile-2026-09-20.yaml` - full Phase 1 reconcile output (drift entries D1-D5, ecosystem scan matrix, rollout evidence, submodule table, instrument fire summary).
- `/Users/macbookpro/Documents/yuji/.audit/findings.yaml` - class-scoped findings updated in place under `phase_2_findings_2026_09_20` block; prior drift ledger preserved per Rule 111.
- `/Users/macbookpro/Documents/yuji/.audit/evidence.yaml` - fired-instrument evidence updated in place under `phase_2_run_2026_09_20` block.
- `/Users/macbookpro/Documents/yuji/.audit/progress.yaml` - phase transitions 0.5 to 1 (approval sign) and 1 to 2 (verify complete) appended; `current_phase` advanced to 2.

---

*Report end. Ceiling: HOLD. Next unblocking action: address one or both structural blockers (rollout traj_tests scorability, vendored-submodule initialization) and re-run CRUCIBLE Phase 2. Neither blocker is resolvable inside the static-analysis contract of this run.*
