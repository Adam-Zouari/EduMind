"""Fail-closed, blinded semantic labels; no built-in model or API client.

Inject a transport or replay saved labels. Calibration is an external reviewed
input, never produced by benchmark execution. Every transport reply is retained.
"""

import json
import subprocess
from pathlib import Path

from edumind.common.artifacts import atomic_write_json, sha256_file, stable_hash
from experiments.benchmarks.common.protocol import (
    boolean,
    integer,
    mapping,
    number,
    sequence,
    strict_object,
    string,
)

TASKS = (
    "claim_extraction",
    "full_context_support",
    "cited_subset_support",
    "correctness",
    "gold_matching",
    "relevancy",
    "semantic_equivalence",
)
LABEL_SCHEMA = "generation-judge-labels-v2"


def load_judge_contract(path):
    if path is None:
        raise ValueError("judge_calibration_pending: no frozen semantic judge contract")
    path = Path(path)
    contract = strict_object(
        json.loads(path.read_text(encoding="utf-8")),
        "judge contract",
        {
            "schema_version",
            "identity",
            "prompts",
            "rubric",
            "label_schema",
            "retry",
            "calibration",
        },
    )
    if (
        type(contract["schema_version"]) is not int
        or contract["schema_version"] != 2
        or contract["label_schema"] != LABEL_SCHEMA
    ):
        raise ValueError("Unsupported semantic judge contract/schema")
    identity = strict_object(
        contract["identity"], "judge identity", {"model_id", "revision", "decoding"}
    )
    string(identity["model_id"], "judge model ID")
    string(identity["revision"], "judge exact version")
    mapping(identity["decoding"], "judge decoding")
    prompts = strict_object(contract["prompts"], "judge prompts", {"response", "pair"})
    for value in prompts.values():
        string(value, "judge prompt")
    rubrics = strict_object(contract["rubric"], "judge rubrics", {"response", "pair"})
    for value in rubrics.values():
        string(value, "judge task rubric")
    retry = strict_object(
        contract["retry"], "judge retry", {"maximum_attempts", "timeout_seconds"}
    )
    integer(retry["maximum_attempts"], "judge maximum_attempts", minimum=1)
    number(
        retry["timeout_seconds"],
        "judge timeout_seconds",
        minimum=0,
        minimum_exclusive=True,
    )
    # JSON serializability and finite decoder fields are required throughout.
    json.dumps(contract, allow_nan=False)
    checksum = stable_hash(
        {key: value for key, value in contract.items() if key != "calibration"}
    )
    calibration_ref = strict_object(
        contract["calibration"], "calibration reference", {"path", "sha256"}
    )
    calibration_path = path.parent / string(calibration_ref["path"], "calibration path")
    if sha256_file(calibration_path) != calibration_ref["sha256"]:
        raise ValueError("Judge calibration checksum mismatch")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if (
        calibration.get("schema_version") != 2
        or calibration.get("label_schema") != LABEL_SCHEMA
        or calibration.get("split") != "development"
        or calibration.get("contract_checksum") != checksum
    ):
        raise ValueError(
            "Judge calibration must bind this exact contract and development-only data"
        )
    string(calibration.get("reviewed_by"), "calibration reviewer")
    string(calibration.get("selection_rationale"), "judge selection rationale")
    if not mapping(calibration.get("cost_review"), "judge calibration cost review"):
        raise ValueError("Judge calibration requires recorded observed cost review")
    labels_ref = strict_object(
        calibration.get("human_labels"), "human labels", {"path", "sha256"}
    )
    labels_path = calibration_path.parent / string(
        labels_ref["path"], "human label path"
    )
    if sha256_file(labels_path) != labels_ref["sha256"]:
        raise ValueError("Human calibration labels checksum mismatch")
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    if (
        labels.get("schema_version") != 2
        or labels.get("label_schema") != LABEL_SCHEMA
        or labels.get("split") != "development"
        or not labels.get("items")
    ):
        raise ValueError(
            "Human calibration set must contain reviewed development-only items"
        )
    criteria = strict_object(
        calibration.get("criteria"), "calibration criteria", set(TASKS)
    )
    for task, result in criteria.items():
        result = strict_object(result, task, {"criterion", "observed", "passed"})
        string(result["criterion"], "recorded acceptance criterion")
        number(result["observed"], "observed calibration result")
        if not boolean(result["passed"], "calibration passed"):
            raise ValueError(f"Judge failed calibration: {task}")
    return {
        **contract,
        "checksum": checksum,
        "contract_path": str(path.resolve()),
        "contract_sha256": sha256_file(path),
        "calibration_path": str(calibration_path.resolve()),
    }


class SemanticJudge:
    def __init__(
        self,
        contract_path,
        transport,
        *,
        journal_directory=None,
        evaluation_identity=None,
    ):
        self.contract = load_judge_contract(contract_path)
        self.transport = transport
        self.journal_directory = (
            Path(journal_directory) if journal_directory is not None else None
        )
        self.evaluation_identity = evaluation_identity
        self._attempted_requests = set()
        if self.journal_directory is not None:
            if not evaluation_identity:
                raise ValueError(
                    "Durable judge journal requires prediction/evaluation identity"
                )
            self.journal_directory.mkdir(parents=True, exist_ok=True)
            self.journal_identity = {
                "schema_version": 2,
                "contract_sha256": self.contract["contract_sha256"],
                "evaluation_identity": evaluation_identity,
            }
            path = self.journal_directory / "identity.json"
            if path.exists():
                if (
                    json.loads(path.read_text(encoding="utf-8"))
                    != self.journal_identity
                ):
                    raise ValueError(
                        "Judge journal prediction/evaluator identity mismatch"
                    )
            else:
                from .evidence import write_new_json

                write_new_json(path, self.journal_identity)

    def with_journal(self, directory, identity):
        return SemanticJudge(
            self.contract["contract_path"],
            self.transport,
            journal_directory=directory,
            evaluation_identity=identity,
        )

    def response(self, question, response):
        # No candidate/model/mode/seed identity, hidden reasoning, or timings.
        request = {
            "task": "response",
            "question": question["question"],
            "response": response,
            "supplied_evidence": [
                {"id": b["id"], "text": b["text"]} for b in question["blocks"]
            ],
            "cited_evidence": [
                {"id": b["id"], "text": b["text"]}
                for b in question["blocks"]
                if b["id"] in set(response["citations"])
            ],
            "accepted_answers": question["accepted_answers"],
            "required_gold_claims": question["gold_claims"],
            "verification_material": question["verification_material"],
        }
        return self._call(
            request, lambda value: validate_response_labels(value, question, response)
        )

    def pair(self, question, left, right):
        return self._call(
            {
                "task": "pair",
                "question": question["question"],
                "left": left,
                "right": right,
            },
            validate_pair_labels,
        )

    def _call(self, request, validate):
        request = {
            **request,
            "prompt": self.contract["prompts"][request["task"]],
            "rubric": self.contract["rubric"][request["task"]],
            "label_schema": LABEL_SCHEMA,
        }
        path = (
            self.journal_directory / (stable_hash(request) + ".json")
            if self.journal_directory
            else None
        )
        entry = {
            "request": request,
            "calls": [],
            "status": "pending",
            "labels": None,
            "contract_checksum": self.contract["checksum"],
        }

        def checkpoint():
            if path is not None:
                atomic_write_json(
                    path, {"entry": entry, "checksum": stable_hash(entry)}
                )

        if path is not None and path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            entry = saved["entry"]
            if (
                saved.get("checksum") != stable_hash(entry)
                or entry.get("request") != request
                or entry.get("contract_checksum") != self.contract["checksum"]
            ):
                raise ValueError("Judge journal request/reply provenance mismatch")
            # A returned raw reply is durable before validation. Recover it even
            # if interruption occurred between receipt and accepted-label commit.
            for call in reversed(entry["calls"]):
                if call.get("state") in {"received", "accepted"}:
                    raw = call["raw"]
                    try:
                        value = validate(
                            json.loads(raw) if isinstance(raw, str) else raw
                        )
                    except (ValueError, TypeError, KeyError) as exc:
                        if call["state"] == "accepted":
                            raise ValueError(
                                "Judge journal accepted reply is invalid"
                            ) from exc
                        call.update(
                            state="rejected", error=f"{type(exc).__name__}: {exc}"
                        )
                        checkpoint()
                        continue
                    if entry["status"] == "evaluated" and entry["labels"] != value:
                        raise ValueError("Judge journal accepted labels changed")
                    call.update(state="accepted", error=None)
                    entry.update(status="evaluated", labels=value)
                    checkpoint()
                    return entry
            if entry["status"] == "evaluated":
                raise ValueError("Judge journal lacks accepted raw reply")
            if stable_hash(request) in self._attempted_requests:
                return entry
        self._attempted_requests.add(stable_hash(request))
        checkpoint()
        calls = entry["calls"]
        round_id = 1 + max((call.get("round", 0) for call in calls), default=0)
        for _ in range(self.contract["retry"]["maximum_attempts"]):
            raw = None
            call = {"round": round_id, "state": "started", "raw": None, "error": None}
            calls.append(call)
            checkpoint()
            try:
                raw = self.transport(
                    request, timeout_seconds=self.contract["retry"]["timeout_seconds"]
                )
                call.update(raw=raw, state="received")
                checkpoint()
                value = json.loads(raw) if isinstance(raw, str) else raw
                value = validate(value)
            except Exception as exc:  # noqa: BLE001 - saved evaluator failure, never candidate failure
                call.update(
                    raw=getattr(exc, "raw_reply", raw),
                    state="rejected",
                    error=f"{type(exc).__name__}: {exc}",
                )
                checkpoint()
            else:
                call.update(state="accepted", error=None)
                entry.update(status="evaluated", labels=value)
                entry.pop("reason", None)
                checkpoint()
                return entry
        entry.update(
            status="unavailable", reason="judge_retries_exhausted", labels=None
        )
        checkpoint()
        return entry


def _sources(value, inventory, label):
    sources = []
    for source in sequence(value, label):
        source = strict_object(source, label, {"id", "quote"})
        if (
            source["id"] not in inventory
            or not isinstance(source["quote"], str)
            or not source["quote"]
            or source["quote"] not in inventory[source["id"]]
        ):
            raise ValueError(
                f"{label} must cite recorded source spans, not judge memory"
            )
        sources.append(source)
    return sources


def validate_response_labels(value, question, response):
    value = strict_object(value, "semantic response labels", {"claims", "relevancy"})
    blocks = {row["id"]: row["text"] for row in question["blocks"]}
    cited = {key: text for key, text in blocks.items() if key in response["citations"]}
    material = {row["id"]: row["text"] for row in question["verification_material"]}
    gold = {row["id"] for row in question["gold_claims"]}
    claims = sequence(value["claims"], "claims")
    ids, texts = set(), set()
    for claim in claims:
        claim = strict_object(
            claim,
            "atomic claim",
            {
                "id",
                "text",
                "spans",
                "supported",
                "correct",
                "support_sources",
                "verification_sources",
                "support_reason",
                "cited_supported",
                "cited_support_sources",
                "cited_support_reason",
                "correctness_reason",
                "gold_matches",
            },
        )
        claim_id, text = (
            string(claim["id"], "claim ID"),
            string(claim["text"], "claim text"),
        )
        if claim_id in ids or text.casefold() in texts:
            raise ValueError(
                "Judge claims must be atomic and semantically deduplicated"
            )
        ids.add(claim_id)
        texts.add(text.casefold())
        spans = sequence(claim["spans"], "original claim spans")
        if not spans:
            raise ValueError("Claim original spans are required")
        for span in spans:
            if not isinstance(span, list) or len(span) != 2:
                raise ValueError("Claim span must be [start,end)")
            start = integer(span[0], "claim start", minimum=0)
            integer(
                span[1], "claim end", minimum=start + 1, maximum=len(response["answer"])
            )
        supported, correct = (
            boolean(claim["supported"], "supported"),
            boolean(claim["correct"], "correct"),
        )
        support = _sources(claim["support_sources"], blocks, "support_sources")
        cited_supported = boolean(claim["cited_supported"], "cited_supported")
        cited_support = _sources(
            claim["cited_support_sources"], cited, "cited_support_sources"
        )
        verification = _sources(
            claim["verification_sources"], material, "verification_sources"
        )
        string(claim["support_reason"], "support justification")
        string(claim["cited_support_reason"], "cited support justification")
        string(claim["correctness_reason"], "correctness justification")
        if (
            (supported and not support)
            or (correct and not verification)
            or (cited_supported and not cited_support)
        ):
            raise ValueError(
                "Positive semantic labels require recorded authoritative sources"
            )
        if cited_supported and not supported:
            raise ValueError("Cited-subset support contradicts full-context support")
        matched = set()
        for match in sequence(claim["gold_matches"], "gold_matches"):
            match = strict_object(match, "gold match", {"id", "justification"})
            if match["id"] not in gold or match["id"] in matched or not correct:
                raise ValueError(
                    "Gold matches must be unique, source-correct required facts"
                )
            string(match["justification"], "gold matching justification")
            matched.add(match["id"])
    if (
        response["answer"]
        == "The provided evidence is insufficient to answer the question."
        and claims
    ):
        raise ValueError("The exact refusal has an empty factual-claim inventory")
    relevancy = strict_object(
        value["relevancy"], "relevancy", {"level", "justification"}
    )
    integer(relevancy["level"], "relevancy level", minimum=0, maximum=2)
    string(relevancy["justification"], "relevancy justification")
    return value


def validate_pair_labels(value):
    value = strict_object(
        value, "semantic pair labels", {"equivalent", "justification"}
    )
    boolean(value["equivalent"], "semantic equivalence")
    string(value["justification"], "semantic equivalence justification")
    return value


def replay_judge(contract_path, records_path):
    """Use already obtained judge replies; never initiates a network call."""
    records = strict_object(
        json.loads(Path(records_path).read_text(encoding="utf-8")),
        "judge replay",
        {
            "schema_version",
            "label_schema",
            "contract_checksum",
            "contract_sha256",
            "responses",
        },
    )
    contract = load_judge_contract(contract_path)
    if (
        type(records["schema_version"]) is not int
        or records["schema_version"] != 2
        or records["label_schema"] != LABEL_SCHEMA
        or records["contract_checksum"] != contract["checksum"]
        or records["contract_sha256"] != contract["contract_sha256"]
    ):
        raise ValueError("Saved judge labels belong to a different contract")
    mapping(records["responses"], "judge replay responses")

    def transport(request, *, timeout_seconds):
        return records["responses"][stable_hash(request)]

    return SemanticJudge(contract_path, transport)


def configured_judge(contract_path, transport_path):
    """Use an explicitly configured adapter; never choose a provider or judge.

    The command receives {identity, request} JSON on stdin and returns raw label
    JSON on stdout. The adapter is responsible for executing exactly that frozen
    identity/decoder; credentials stay in its environment, outside artifacts.
    """
    config = strict_object(
        json.loads(Path(transport_path).read_text(encoding="utf-8")),
        "judge transport",
        {"schema_version", "kind", "command"},
    )
    if (
        type(config["schema_version"]) is not int
        or config["schema_version"] != 1
        or config["kind"] != "command"
    ):
        raise ValueError("Unsupported configured judge transport")
    command = [
        string(item, "judge adapter argument")
        for item in sequence(config["command"], "judge adapter command")
    ]
    if not command:
        raise ValueError("Judge transport command is empty")
    contract = load_judge_contract(contract_path)

    def transport(request, *, timeout_seconds):
        try:
            result = subprocess.run(
                command,
                input=json.dumps(
                    {"identity": contract["identity"], "request": request},
                    allow_nan=False,
                ),
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=timeout_seconds,
                check=False,
                shell=False,
            )
        except subprocess.TimeoutExpired as exc:
            raw = (
                exc.stdout.decode("utf-8", errors="replace")
                if isinstance(exc.stdout, bytes)
                else exc.stdout
            )
            raise JudgeTransportError("Configured judge adapter timeout", raw) from exc
        if result.returncode:
            raise JudgeTransportError(
                f"Configured judge adapter exited {result.returncode}", result.stdout
            )
        return result.stdout

    return SemanticJudge(contract_path, transport)


class JudgeTransportError(RuntimeError):
    def __init__(self, message, raw_reply=None):
        self.raw_reply = raw_reply
        super().__init__(message)
