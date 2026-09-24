# Upstream attribution and usage notes

EmoStable-TTS is a portfolio-oriented research snapshot. It contains original
modifications together with code derived from or interoperating with several
upstream research projects. Model weights, datasets, generated audio and API
credentials are not distributed here.

| Component | Location | Upstream status | Treatment in this repository |
| --- | --- | --- | --- |
| EmoVoice | Core TTS design and parts of `examples/tts/` | [Project](https://github.com/yanghaha0908/EmoVoice), [paper](https://arxiv.org/abs/2504.12867). The GitHub repository did not declare a license when checked on 2026-09-24. | Attribution is preserved. No permission to reuse or redistribute EmoVoice-derived portions is granted by this repository. |
| SLAM-LLM | `src/slam_llm/` | [Upstream](https://github.com/X-LANCE/SLAM-LLM), MIT | The upstream MIT text is bundled in `licenses/SLAM-LLM-MIT.txt`. File-level notices remain intact. |
| CosyVoice | `examples/tts/utils/cosyvoice/` | [Upstream](https://github.com/FunAudioLLM/CosyVoice), Apache-2.0 | The upstream Apache-2.0 text is bundled in `licenses/CosyVoice-Apache-2.0.txt`. File-level notices remain intact. |
| Matcha-TTS | `examples/tts/utils/third_party/Matcha-TTS/` | [Upstream](https://github.com/shivammehta25/Matcha-TTS), MIT | Only the runtime subset imported by the codec path is retained. Its upstream `LICENSE` is preserved in that directory. |

External assets such as Qwen, CosyVoice checkpoints, EmoVoice datasets,
Whisper, emotion2vec and UTMOS have their own model/data terms. Users must
obtain those assets separately and comply with their respective licenses.

The repository-level `LICENSE` applies only to code and modifications owned by
the repository owner. It does not replace or broaden any upstream license.

