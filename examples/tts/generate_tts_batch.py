import os
import time
import random
import logging
import hydra
import torch
import soundfile as sf
from omegaconf import DictConfig, OmegaConf

from slam_llm.utils.model_utils import get_custom_model_factory
from slam_llm.utils.dataset_utils import get_preprocessed_dataset
from utils.codec_utils import audio_decode_cosyvoice

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _is_missing_path(path):
	return path in (None, "", "None")


def _resolve_existing_audio_path(path, dataset_config):
	if _is_missing_path(path):
		return None
	path = os.path.expanduser(str(path))
	candidates = [path]
	val_data_path = dataset_config.get("val_data_path", None) if hasattr(dataset_config, "get") else None
	if val_data_path:
		candidates.append(os.path.join(os.path.dirname(os.path.abspath(val_data_path)), path))
	candidates.append(os.path.join(PROJECT_ROOT, path))
	for cand in candidates:
		cand = os.path.abspath(cand)
		if os.path.exists(cand):
			return cand
	return os.path.abspath(path)


@hydra.main(config_name=None, version_base=None)
def main_hydra(cfg: DictConfig):
	kwargs = cfg
	log_level = getattr(logging, kwargs.get("log_level", "INFO").upper())
	logging.basicConfig(level=log_level)
	if kwargs.get("debug", False):
		import pdb;
		pdb.set_trace()
	main(kwargs)

def main(kwargs: DictConfig):
	train_config, fsdp_config, model_config, log_config, dataset_config, decode_config = kwargs.train_config, kwargs.fsdp_config, kwargs.model_config, kwargs.log_config,kwargs.dataset_config, kwargs.decode_config
	OmegaConf.set_struct(kwargs,False)
	del kwargs["train_config"]
	del kwargs["fsdp_config"]
	del kwargs["model_config"]
	del kwargs["log_config"]
	del kwargs["dataset_config"]
	del kwargs["decode_config"]
	OmegaConf.set_struct(kwargs,True)



	# Set log
	if not os.path.exists(os.path.dirname(log_config.log_file)):
		os.makedirs(os.path.dirname(log_config.log_file), exist_ok=True)
	logging.basicConfig(
		level=logging.INFO, 
		format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
		datefmt="%Y-%m-%d %H:%M:%S",
		filemode='w'
	)
	logger = logging.getLogger()  
	logger.setLevel(logging.INFO)

	file_handler = logging.FileHandler(filename=log_config.log_file, mode='w')
	file_handler.setLevel(logging.INFO)
	file_formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
	file_handler.setFormatter(file_formatter)

	logger.handlers[0].setLevel(logging.INFO)
	console_formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
	logger.handlers[0].setFormatter(console_formatter) 
	logger.addHandler(file_handler)
    
	logger.info("train_config: {}".format(train_config))
	logger.info("fsdp_config: {}".format(fsdp_config))
	logger.info("model_config: {}".format(model_config))
	logger.info(f"[CKPT] inference_load_ckpt_path={kwargs.get('ckpt_path', None)}")
	recorded_best_ckpt = kwargs.get("recorded_best_ckpt_path", None) or os.environ.get("SANITY_RECORDED_BEST_CKPT_PATH", None)
	if recorded_best_ckpt not in (None, "", "None"):
		logger.info(f"[CKPT] recorded_best_ckpt_path={recorded_best_ckpt}")
		try:
			if os.path.abspath(str(recorded_best_ckpt)) != os.path.abspath(str(kwargs.get("ckpt_path", ""))):
				logger.warning("[CKPT] [DIAGNOSIS] inference ckpt != recorded best ckpt.")
		except Exception:
			if str(recorded_best_ckpt) != str(kwargs.get("ckpt_path", "")):
				logger.warning("[CKPT] [DIAGNOSIS] inference ckpt != recorded best ckpt.")




	# Set the seeds for reproducibility
	torch.cuda.manual_seed(train_config.seed)
	torch.manual_seed(train_config.seed)
	random.seed(train_config.seed)
	
	model_factory = get_custom_model_factory(model_config, logger) # return func
	model, tokenizer = model_factory(train_config, model_config, **kwargs)
	codec_decoder = model.codec_decoder
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	model.eval()




	logger.info("dataset_config: {}".format(dataset_config))
	dataset_test = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="test",
    )
	if not (train_config.enable_fsdp or train_config.enable_ddp):
		logger.info(f"--> Test Set Length = {len(dataset_test)}")

	test_dataloader = torch.utils.data.DataLoader(
            dataset_test,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
			shuffle=False,
            batch_size=train_config.val_batch_size,
			drop_last=False,
			collate_fn=dataset_test.collator
        )

	task_type = decode_config.task_type
	code_layer = model_config.vocab_config.code_layer
	num_latency_tokens = dataset_config.num_latency_tokens
	modeling_paradigm = dataset_config.modeling_paradigm
	interleaved_text_token_num = dataset_config.interleaved_text_token_num
	interleaved_audio_token_num = dataset_config.interleaved_audio_token_num
	debug_decode_overlong_audio = bool(getattr(decode_config, "debug_decode_overlong_audio", False))

	decode_log_dir = kwargs.get('decode_log')
	output_text_only = kwargs.get('output_text_only', False)
	speech_sample_rate = kwargs.get('speech_sample_rate', 24000)
	default_audio_prompt_path = kwargs.get('audio_prompt_path', None)

	if not os.path.exists(decode_log_dir):
		os.makedirs(decode_log_dir)

	pred_path = os.path.join(decode_log_dir, "pred_text")
	gt_path = os.path.join(decode_log_dir, "gt_text")
	question_path = os.path.join(decode_log_dir, "question_text")
	generate_audio_dir = os.path.join(decode_log_dir, "pred_audio")

	tone_dir = "neutral_prompt_speech"
	tone_audio_dir = os.path.join(generate_audio_dir, tone_dir)
	if not os.path.exists(tone_audio_dir) and not (output_text_only or decode_config.decode_text_only):
		os.makedirs(tone_audio_dir)

	logger.info("decode_config: {}".format(decode_config))	
	if decode_config.do_sample:
		logger.info("Decode Strategy: Sampling")
	else:
		logger.info("Decode Strategy: Greedy")
	if decode_config.decode_text_only:
		logger.info("Decode Text Only")
	else:
		logger.info("Decode Text & Audio")
	logger.info("Decode Code Layer: {}".format(code_layer))
	logger.info("Tone for Audio Generation: {}".format(tone_dir))
	logger.info("Modeling Paradigm: {}".format(modeling_paradigm))
	debug_max_samples = int(getattr(decode_config, "debug_max_samples", 0) or 0)
	if debug_max_samples > 0:
		logger.info("[GEN_DEBUG] debug_max_samples=%d", debug_max_samples)

	if modeling_paradigm == "interleaved":
		logger.info("Interleaved Text Token Num: {}".format(interleaved_text_token_num))
		logger.info("Interleaved Audio Token Num: {}".format(interleaved_audio_token_num))





	logger.info("============== Start {task_type} Inference ==============".format(task_type=task_type))

	with open(pred_path, "w") as pred, open(gt_path, "w") as gt, open(question_path, "w") as q:
		for step, batch in enumerate(test_dataloader):
			if debug_max_samples > 0 and step >= debug_max_samples:
				logger.info("[GEN_DEBUG] debug_max_samples reached. Stop batch inference early.")
				break
			for key in batch.keys():
				batch[key] = batch[key].to(device) if isinstance(batch[key], torch.Tensor) else batch[key]

			batch_audio_prompt_path = batch["neutral_speaker_wav"][0] if "neutral_speaker_wav" in batch and len(batch["neutral_speaker_wav"]) > 0 else None
			if _is_missing_path(batch_audio_prompt_path):
				batch_audio_prompt_path = batch["target_wav"][0] if "target_wav" in batch and len(batch["target_wav"]) > 0 else None
			audio_prompt_path = batch_audio_prompt_path if not _is_missing_path(batch_audio_prompt_path) else default_audio_prompt_path
			audio_prompt_path = _resolve_existing_audio_path(audio_prompt_path, dataset_config)
			batch["audio_prompt_path"] = [audio_prompt_path]
			batch["speech_sample_rate"] = speech_sample_rate

			start_time = time.time()
			if modeling_paradigm == "parallel" or modeling_paradigm == "interleaved":
				model_outputs = model.generate(**batch, **decode_config)
			elif modeling_paradigm == "serial":
				model_outputs = model.serial_generate(**batch, **decode_config)

			if modeling_paradigm == "parallel" or modeling_paradigm == "serial":
				text_outputs = model_outputs[code_layer]
				audio_outputs = model_outputs[:code_layer]
			elif modeling_paradigm == "interleaved":
				text_outputs = model_outputs["text"]
				audio_outputs = model_outputs["audio"]
			else:
				raise NotImplementedError
			end_time_llm = time.time()
			logger.info(f"LLM Inference Time: {end_time_llm - start_time:.2f}s")
			output_text = model.tokenizer.decode(text_outputs, add_special_tokens=False, skip_special_tokens=True)
			for key, source_text, target_text, generated_text in zip(batch["keys"], batch["source_texts"], batch["target_texts"], [output_text]):
				q.write(key + "\t" + source_text + "\n")
				if "chuanxing" in dataset_config.val_data_path:
					target_text = target_text.replace('\n', '')
				gt.write(key + "\t" + target_text + "\n")
				generated_text = generated_text.replace('\n', '')
				pred.write(key + "\t" + generated_text + "\n")
				
				logger.info(f"Target Text: {target_text}")
				logger.info(f"Generated Text: {generated_text}")

			if output_text_only or decode_config.decode_text_only:
				continue
			if _is_missing_path(audio_prompt_path):
				raise RuntimeError("Missing reference wav for codec decode. Please set `neutral_speaker_wav` in jsonl or pass `++audio_prompt_path=/abs/path/ref.wav`.")
			if not os.path.exists(audio_prompt_path):
				raise FileNotFoundError(f"Reference wav not found: {audio_prompt_path}")
			
			allow_missing_eoa_decode = False
			if modeling_paradigm != "serial":
				if audio_outputs[0].shape[0] == decode_config.max_new_tokens:	# if the audio token is too long, skip (bad case)
					if debug_decode_overlong_audio:
						allow_missing_eoa_decode = True
						logger.warning(
							"Audio token reached max_new_tokens without EOA; "
							"debug_decode_overlong_audio=True so decode will use the full generated length."
						)
					else:
						logger.warning(f"Audio token is too long, skip. You can try to increase the max_new_tokens in the decode_config.")
						continue
			else:
				if audio_outputs[0]==[]:
					logger.warning(f"Text token never stop")
					continue				

			for i, key in enumerate(batch["keys"]):
				audio_tokens = [audio_outputs[layer] for layer in range(code_layer)] if code_layer > 0 else audio_outputs
				audio_hat = audio_decode_cosyvoice(
					audio_tokens,
					model_config,
					codec_decoder,
					audio_prompt_path,
					code_layer,
					num_latency_tokens,
					speed=1.0,
					allow_missing_eoa=allow_missing_eoa_decode,
				)
				if audio_hat == None:
					logger.info(f"Error in decoding {key}: eoa at start! or No eoa!")
					continue

				if key[-4:] == ".wav":
					key = key[:-4]
				end_time = time.time()
				audio_length = audio_hat.shape[1] / speech_sample_rate
				RTF = (end_time - start_time) / audio_length
				sf.write(f"{tone_audio_dir}/{key}.wav", audio_hat.squeeze().cpu().numpy(), speech_sample_rate)
				logger.info(f"Generated Audio: {tone_dir}/{key}.wav, audio length: {audio_length:.2f}s, generation time: {end_time - start_time:.2f}s, RTF: {RTF:.2f}")
				RTF_llm = (end_time_llm - start_time) / audio_length
				logger.info(f"LLM RTF: {RTF_llm:.2f}")

	logger.info("============== Inference Finished ==============")

if __name__ == "__main__":
	main_hydra()
