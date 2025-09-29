from datasets import load_dataset, get_dataset_config_names, interleave_datasets, load_dataset_builder
from .dataset import HFDiaIterDataset
import pandas as pd
from huggingface_hub import hf_hub_download


LANG_NAME_TO_CODE = {
    "dutch":      "nl",
    "french":     "fr",
    "german":     "de",
    "italian":    "it",
    "polish":     "pl",
    "portuguese": "pt",
    "spanish":    "es",
    # add more if other configs appear...
}






def load_cml_tts_streamed(dia_cfg, dac_model):
    """
    Stream all language subsets of the CML-TTS dataset in train split,
    add a `language` field, drop all except `text`, `audio`, `language`,
    and interleave them into one streaming Dataset.

    Returns:
        datasets.IterableDataset: interleaved streaming dataset
    """
    # 1) Discover all language subsets
    lang_configs = get_dataset_config_names("ylacombe/cml-tts")

    # 2) Build one streaming subset per language, with only desired columns
    streams = []
    num_ex=0
    for lang in lang_configs:
        
        iso_code = LANG_NAME_TO_CODE.get(lang, lang)
        ds_stream = load_dataset(
            "ylacombe/cml-tts",
            name=lang,
            split="train",
            streaming=True
        )

        num_ex += ds_stream.info.splits['train'].num_examples
        # keep only text, audio, and add language
        def _add_lang(ex, iso=iso_code):
            return {
                "text": ex["text"],
                "audio": ex["audio"],
                "language": iso
            }
        ds_stream = ds_stream.map(
            _add_lang,
            remove_columns=[c for c in ds_stream.column_names if c not in ["text", "audio", "language"]]
        )
        streams.append(ds_stream)

    # 3) Interleave all streams into one unified stream
    interleaved = interleave_datasets(streams, stopping_strategy="all_exhausted")
    ds = HFDiaIterDataset(interleaved, dia_cfg, dac_model)
    ds.total_examples = num_ex
    return ds






def count_tsv_rows(
    repo_id: str,
    subset: str,
    split: str = "train",
    revision: str = "main"
) -> int:
    """Download the TSV for a given subset/split and return its number of rows."""
    file_path = f"transcript/{subset}/{split}.tsv"
    try:
        local_file = hf_hub_download(
            repo_id=repo_id,
            filename=file_path,
            repo_type="dataset",
            revision=revision
        )
    except:
        print("error fetching tsv metadata")

    df = pd.read_csv(local_file, sep="\t", low_memory=False)
    return len(df)

def load_common_voice17_streamed(dia_cfg, dac_model, revision="main"):
    """
    Stream the train split of Common Voice 17 for the given language codes,
    rename `sentence`→`text`, keep only `text`, `audio`, and `language`,
    then interleave into a single streaming Dataset.

    Languages loaded: en, de, fr, es, it, nl, pl, pt, tr, hu
    """
    repo_id = "mozilla-foundation/common_voice_17_0"
    langs = ["en", "de", "fr", "es", "it", "nl", "pl", "pt", "tr", "hu"]

    streams = []
    row_counts = []

    for lang in langs:
        # 1) figure out how many rows in the TSV
        n_rows = count_tsv_rows(repo_id, lang, split="train", revision=revision)
        row_counts.append(n_rows)

        # 2) load in streaming mode
        ds_stream = load_dataset(
            repo_id,
            name=lang,
            split="train",
            streaming=True,
            revision=revision
        )

        # 3) map to desired schema
        def _prep(ex, iso=lang):
            return {
                "text": ex["sentence"],
                "audio": ex["audio"],
                "language": iso
            }

        ds_stream = ds_stream.map(
            _prep,
            remove_columns=[c for c in ds_stream.column_names if c not in ("sentence", "audio")]
        )
        streams.append(ds_stream)

    # 4) interleave: all_exhausted ⇒ max_length * num_streams
    interleaved = interleave_datasets(streams, stopping_strategy="all_exhausted")

    # 5) wrap and attach total_examples
    ds = HFDiaIterDataset(interleaved, dia_cfg, dac_model)
    ds.total_examples = max(row_counts) * len(langs)

    return ds

