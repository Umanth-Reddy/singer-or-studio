# Reproduction environment

All runs used Python 3.10.16 on CPU. No NVIDIA driver/GPU was available.

| Role | Environment | Main packages |
|---|---|---|
| Music4All preparation and metric analysis | /legacy/ps/.venv-ssl | NumPy 1.26.4, SciPy 1.15.3, PyTorch 2.5.1+cpu, SoundFile 0.14.0 |
| CVSM inference | /legacy/ps/.venv-gate1 | TensorFlow 2.15.1, NumPy 1.26.4, SciPy 1.15.3, SoundFile 0.14.0 |
| Torch / speech / music model inference and adapter fitting | /legacy/ps/claude-research/.venv | PyTorch 2.14.1, TorchAudio 2.11.0, Transformers 5.20.2, SpeechBrain 1.1.2, NumPy 2.2.6, SciPy 1.15.3 |

System tools: FFmpeg 8.0.1, 7-Zip 26.00, zipinfo, ffprobe, sha256sum. Model checkpoints were loaded locally; the runs did not download packages or weights.
