from pathlib import Path
from types import SimpleNamespace
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_cpo
import run_ipo
import run_kto
import run_orpo
import run_robust_dpo


class _FakeTrainer:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def _fake_parser(training_args):
    class _Parser:
        def __init__(self, *_args, **_kwargs):
            pass

        def parse(self):
            model_args = SimpleNamespace(model_name_or_path="policy-model")
            data_args = SimpleNamespace(dataset_mixer={"HuggingFaceH4/ultrafeedback_binarized": 1.0})
            return model_args, data_args, training_args

    return _Parser


def _patch_common(monkeypatch, module, training_args, *, datasets, model_tuple):
    captured = {}

    monkeypatch.setattr(module, "H4ArgumentParser", _fake_parser(training_args))
    monkeypatch.setattr(module, "setup_run", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "finalize_training", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(module, "logger", SimpleNamespace(info=lambda *args, **kwargs: None))

    if hasattr(module, "prepare_pairwise_preference_datasets"):
        monkeypatch.setattr(
            module,
            "prepare_pairwise_preference_datasets",
            lambda *args, **kwargs: (datasets, "tokenizer"),
        )
    if hasattr(module, "prepare_kto_datasets"):
        monkeypatch.setattr(
            module,
            "prepare_kto_datasets",
            lambda *args, **kwargs: (datasets, "tokenizer"),
        )

    monkeypatch.setattr(
        module,
        "prepare_trl_trainer_models",
        lambda *args, **kwargs: model_tuple,
    )

    return captured


def test_run_ipo_constructs_dpo_trainer_with_ipo_loss(monkeypatch):
    training_args = SimpleNamespace(
        beta=0.2,
        loss_type="sigmoid",
        do_eval=True,
        model_init_kwargs={"torch_dtype": "bfloat16"},
    )
    datasets = {"train": ["train"], "test": ["test"]}
    captured = _patch_common(
        monkeypatch,
        run_ipo,
        training_args,
        datasets=datasets,
        model_tuple=("policy-model", "ref-model", "peft-config"),
    )
    monkeypatch.setattr(run_ipo, "DPOTrainer", _FakeTrainer)

    run_ipo.main()

    trainer = captured["trainer"]
    assert training_args.loss_type == "ipo"
    assert trainer.kwargs["ref_model"] == "ref-model"
    assert trainer.kwargs["model_init_kwargs"] == {"torch_dtype": "bfloat16"}
    assert trainer.kwargs["ref_model_init_kwargs"] == {"torch_dtype": "bfloat16"}


def test_run_robust_dpo_constructs_dpo_trainer_with_robust_loss(monkeypatch):
    training_args = SimpleNamespace(
        beta=0.2,
        label_smoothing=0.15,
        loss_type="sigmoid",
        do_eval=False,
        model_init_kwargs={"torch_dtype": "bfloat16"},
    )
    datasets = {"train": ["train"]}
    captured = _patch_common(
        monkeypatch,
        run_robust_dpo,
        training_args,
        datasets=datasets,
        model_tuple=("policy-model", "ref-model", "peft-config"),
    )
    monkeypatch.setattr(run_robust_dpo, "DPOTrainer", _FakeTrainer)

    run_robust_dpo.main()

    trainer = captured["trainer"]
    assert training_args.loss_type == "robust"
    assert trainer.kwargs["ref_model"] == "ref-model"


def test_run_cpo_constructs_cpo_trainer_without_reference_model(monkeypatch):
    training_args = SimpleNamespace(
        beta=0.1,
        cpo_alpha=1.0,
        loss_type="hinge",
        do_eval=False,
        model_init_kwargs={"torch_dtype": "bfloat16"},
    )
    datasets = {"train": ["train"]}
    captured = _patch_common(
        monkeypatch,
        run_cpo,
        training_args,
        datasets=datasets,
        model_tuple=("policy-model", None, "peft-config"),
    )
    monkeypatch.setattr(run_cpo, "CPOTrainer", _FakeTrainer)

    run_cpo.main()

    trainer = captured["trainer"]
    assert training_args.loss_type == "sigmoid"
    assert "ref_model" not in trainer.kwargs


def test_run_kto_constructs_kto_trainer_with_reference_model(monkeypatch):
    training_args = SimpleNamespace(
        beta=0.1,
        desirable_weight=1.0,
        undesirable_weight=1.0,
        do_eval=True,
        model_init_kwargs={"torch_dtype": "bfloat16"},
    )
    datasets = {"train": ["train"], "test": ["test"]}
    captured = _patch_common(
        monkeypatch,
        run_kto,
        training_args,
        datasets=datasets,
        model_tuple=("policy-model", "ref-model", "peft-config"),
    )
    monkeypatch.setattr(run_kto, "KTOTrainer", _FakeTrainer)

    run_kto.main()

    trainer = captured["trainer"]
    assert trainer.kwargs["ref_model"] == "ref-model"
    assert trainer.kwargs["train_dataset"] == ["train"]


def test_run_orpo_constructs_orpo_trainer_without_reference_model(monkeypatch):
    training_args = SimpleNamespace(
        beta=0.1,
        do_eval=False,
        model_init_kwargs={"torch_dtype": "bfloat16"},
    )
    datasets = {"train": ["train"]}
    captured = _patch_common(
        monkeypatch,
        run_orpo,
        training_args,
        datasets=datasets,
        model_tuple=("policy-model", None, "peft-config"),
    )
    monkeypatch.setattr(run_orpo, "ORPOTrainer", _FakeTrainer)

    run_orpo.main()

    trainer = captured["trainer"]
    assert "ref_model" not in trainer.kwargs
    assert trainer.kwargs["model"] == "policy-model"
