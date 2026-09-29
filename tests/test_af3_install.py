"""Installation boundaries and configuration; real GPU checks are recorded separately."""
import json
from pathlib import Path
import sys
import subprocess

import pytest

from guideflip import validation as af3

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('old_override', [False, True])
def test_configuration_in_installation_directory(tmp_path, monkeypatch, old_override):
    saved = tmp_path / 'installation/.guideflip-af3.json'
    saved.parent.mkdir()
    saved.write_text(json.dumps(dict(python='/runtime/python', script='/runtime/run.py',
                                    model_dir='/weights')))
    monkeypatch.setattr(af3, 'INSTALL_CONFIG', saved)
    for key in ('CONFIG', 'PYTHON', 'SCRIPT', 'MODEL_DIR'):
        monkeypatch.delenv('GUIDEFLIP_AF3_' + key, raising=False)
    if old_override:
        monkeypatch.setenv('GUIDEFLIP_AF3_CONFIG', str(tmp_path / '.guideflip-af3.json'))
    cfg = af3.AF3Config.from_settings({}, beside=tmp_path)
    assert cfg.python == '/runtime/python' and cfg.model_dir == '/weights'
    assert not (tmp_path / '.guideflip-af3.json').exists()


def test_installation_scripts_resolve_repository_from_another_working_directory(tmp_path):
    for script, args in [('install.sh', []), ('install_af3.sh', ['--model-dir', str(tmp_path/'weights')])]:
        result = subprocess.run(['bash', str(ROOT/'installation'/script), '--dry-run', *args],
                                cwd=tmp_path, text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
        if script == 'install.sh':
            assert str(ROOT/'installation/requirements.txt') in result.stdout
            assert str(ROOT) + '[models]' in result.stdout
        else:
            assert str(ROOT/'installation/.guideflip-af3.json') in result.stdout


def test_saved_installation_paths_and_overrides(tmp_path, monkeypatch):
    saved = tmp_path / "installation.json"
    saved.write_text(json.dumps(dict(python="venv/python", script="source/run.py", model_dir="weights")))
    monkeypatch.setenv("GUIDEFLIP_AF3_CONFIG", str(saved))
    for name in ("PYTHON", "SCRIPT", "MODEL_DIR"):
        monkeypatch.delenv("GUIDEFLIP_AF3_" + name, raising=False)
    settings = tmp_path / "settings"
    cfg = af3.AF3Config.from_settings({}, beside=settings)
    assert cfg.python == str(tmp_path / "venv/python")
    assert cfg.script == str(tmp_path / "source/run.py")
    monkeypatch.setenv("GUIDEFLIP_AF3_MODEL_DIR", "/external/weights")
    cfg = af3.AF3Config.from_settings({"python": "custom/python"}, beside=settings)
    assert cfg.python == str(settings / "custom/python")
    assert cfg.model_dir == "/external/weights"


def test_missing_explicit_installation_config_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("GUIDEFLIP_AF3_CONFIG", str(tmp_path / "missing.json"))
    with pytest.raises(FileNotFoundError, match="configuration is missing"):
        af3.AF3Config.from_settings({}, beside=tmp_path)


def test_installer_refuses_unowned_and_other_checkouts(tmp_path, monkeypatch):
    source = (ROOT / "installation/install_af3.sh").read_text().split("<<'PYTHON_INSTALLER'\n", 1)[1]
    source = source.rsplit("\nPYTHON_INSTALLER", 1)[0]
    monkeypatch.setattr(sys, "argv", ["install_af3.sh", str(ROOT)])
    module = {"__name__": "installer_test"}
    exec(compile(source, "install_af3.sh", "exec"), module)
    prefix = tmp_path / "existing"
    prefix.mkdir()
    valuable = prefix / "user-file"
    valuable.write_text("preserve me")
    with pytest.raises(ValueError, match="choose a new --prefix"):
        module["check_prefix"](prefix)
    assert valuable.read_text() == "preserve me"
    marker = prefix / module["MARKER"]
    marker.write_text(json.dumps(dict(repository="/another/checkout", revision=module["REVISION"], format=1)))
    with pytest.raises(ValueError, match="another installation"):
        module["check_prefix"](prefix)


def test_disabled_af3_ignores_missing_installation(tmp_path, monkeypatch):
    monkeypatch.setenv("GUIDEFLIP_AF3_CONFIG", str(tmp_path / "missing.json"))
    assert af3.AF3Config.from_settings(None, beside=tmp_path) is None


def test_af3_addon_isolates_design_environment(tmp_path, monkeypatch):
    design = tmp_path / 'guideflip'
    prefix = tmp_path / 'af3'
    monkeypatch.setenv('CONDA_PREFIX', str(design))
    process = subprocess.run(['bash', str(ROOT / 'installation/install_af3.sh'), '--dry-run',
                              '--model-dir', str(tmp_path / 'weights'),
                              '--prefix', str(prefix)], capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    commands = [line for line in process.stdout.splitlines() if line.startswith('+ ')]
    assert all(str(design) not in line for line in commands)
    assert any(' create ' in line and str(prefix / 'toolchain') in line for line in commands)
    assert any(' sync ' in line and '--frozen' in line and '--no-editable' in line for line in commands)
    assert any(str(prefix / 'venv/bin/build_data') in line for line in commands)
    assert not prefix.exists()


@pytest.mark.parametrize('format_version', [1, 3])
def test_installer_preserves_incompatible_environment(tmp_path, monkeypatch, format_version):
    source = (ROOT / 'installation/install.sh').read_text().split("<<'PYTHON_INSTALLER'\n", 1)[1]
    source = source.rsplit('\nPYTHON_INSTALLER', 1)[0]
    monkeypatch.setattr(sys, 'argv', ['install.sh', str(ROOT)])
    module = {'__name__': 'installer_test'}
    exec(compile(source, 'install.sh', 'exec'), module)
    old = tmp_path / 'old-guideflip'
    old.mkdir()
    marker = old / module['MARKER']
    prior = json.dumps(dict(repository=str(ROOT), format=format_version, manager='conda'))
    marker.write_text(prior)
    with pytest.raises(ValueError, match='different installation'):
        module['check_environment_path'](old)
    assert marker.read_text() == prior
