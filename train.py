import pathlib
import random

import numpy as np
import pandas as pd
import seaborn as sns
import speechbrain as sb
import torch
from huggingface_hub import hf_hub_download
from hyperpyyaml import load_hyperpyyaml
from speechbrain.dataio.dataio import read_audio, write_audio
from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.dataio.encoder import CategoricalEncoder

from crybrain import CryBrain, download_data
import sys

dataset_path = "/home/fpaissan/data/cryceleb"

def create_cut_length_interval(row, cut_length_interval):
    """cut_length_interval is a tuple indicating the range of lengths we want our chunks to be.
    this function computes the valid range of chunk lengths for each audio file
    """
    # the lengths are in seconds, convert them to frames
    cut_length_interval = [round(length * 16000) for length in cut_length_interval]
    cry_length = round(row["duration"] * 16000)
    # make the interval valid for the specific sound file
    min_cut_length, max_cut_length = cut_length_interval
    # if min_cut_length is greater than length of cry, don't cut
    if min_cut_length >= cry_length:
        cut_length_interval = (cry_length, cry_length)
    # if max_cut_length is greater than length of cry, take a cut of length between min_cut_length and full length of cry
    elif max_cut_length >= cry_length:
        cut_length_interval = (min_cut_length, cry_length)
    return cut_length_interval

def get_babies_with_both_recordings(manifest_df):
    count_of_periods_per_baby = manifest_df.groupby("baby_id")["period"].count()
    baby_ids_with_recording_from_both_periods = count_of_periods_per_baby[
        count_of_periods_per_baby == 2
    ].index
    return baby_ids_with_recording_from_both_periods

def split_by_period(row, included_baby_ids):
    if row["baby_id"] in included_baby_ids:
        if row["period"] == "B":
            return "train"
        else:
            return "val"
    else:
        return "not_used"

def audio_pipeline(file_path, cut_length_interval_in_frames):
    """Load the signal, and pass it and its length to the corruption class.
    This is done on the CPU in the `collate_fn`."""
    sig = sb.dataio.dataio.read_audio(file_path)
    if cut_length_interval_in_frames is not None:
        cut_length = random.randint(*cut_length_interval_in_frames)
        # pick the start index of the cut
        left_index = random.randint(0, len(sig) - cut_length)
        # cut the signal
        sig = sig[left_index : left_index + cut_length]
    return sig

def prepare_data(dataset_path):
    download_data(dataset_path)

    # read metadata
    metadata = pd.read_csv(
        f"{dataset_path}/metadata.csv", dtype={"baby_id": str, "chronological_index": str}
    )
    train_metadata = metadata.loc[metadata["split"] == "train"].copy()
    train_metadata["cry"] = train_metadata.apply(
        lambda row: read_audio(f'{dataset_path}/{row["file_name"]}').numpy(), axis=1
    )
    # concatenate all segments for each (baby_id, period) group
    manifest_df = pd.DataFrame(
        train_metadata.groupby(["baby_id", "period"])["cry"].agg(lambda x: np.concatenate(x.values)),
        columns=["cry"],
    ).reset_index()
    # all files have 16000 sampling rate
    manifest_df["duration"] = manifest_df["cry"].apply(len) / 16000
    pathlib.Path(f"{dataset_path}/concatenated_audio_train").mkdir(exist_ok=True)
    manifest_df["file_path"] = manifest_df.apply(
        lambda row: f"{dataset_path}/concatenated_audio_train/{row['baby_id']}_{row['period']}.wav",
        axis=1,
    )
    manifest_df.apply(
        lambda row: write_audio(
            filepath=f'{row["file_path"]}', audio=torch.tensor(row["cry"]), samplerate=16000
        ),
        axis=1,
    )
    manifest_df = manifest_df.drop(columns=["cry"])
    ax = sns.histplot(manifest_df, x="duration")
    ax.set_title("Histogram of Concatenated Cry Sound Lengths")
    
    cut_length_interval = (3, 5)
    manifest_df["cut_length_interval_in_frames"] = manifest_df.apply(
        lambda row: create_cut_length_interval(row, cut_length_interval=cut_length_interval), axis=1
    )

    babies_with_both_recordings = get_babies_with_both_recordings(manifest_df)
    manifest_df["split"] = manifest_df.apply(
        lambda row: split_by_period(row, included_baby_ids=babies_with_both_recordings), axis=1
    )
    
    # each instance will be identified with a unique id
    manifest_df["id"] = manifest_df["baby_id"] + "_" + manifest_df["period"]
    manifest_df.set_index("id").to_json("manifest.json", orient="index")

    # create a dynamic dataset from the csv, only used to create train and val datasets
    dataset = DynamicItemDataset.from_json("manifest.json")
    baby_id_encoder = CategoricalEncoder()
    datasets = {}
    # create a dataset for each split
    for split in ["train", "val"]:
        # retrieve the desired slice (train or val) and sort by length to minimize amount of padding
        datasets[split] = dataset.filtered_sorted(
            key_test={"split": lambda value: value == split}, sort_key="duration"
        )  # select_n=100
        # create the baby_id_encoded field
        datasets[split].add_dynamic_item(
            baby_id_encoder.encode_label_torch, takes="baby_id", provides="baby_id_encoded"
        )
        # set visible fields
        datasets[split].set_output_keys(["id", "baby_id", "baby_id_encoded", "sig"])
    
    
    # create the signal field for the val split (no chunking)
    datasets["val"].add_dynamic_item(sb.dataio.dataio.read_audio, takes="file_path", provides="sig")
    
    # the label encoder will map the baby_ids to target classes 0, 1, 2, ...
    # only use the classes which appear in `train`,
    baby_id_encoder.update_from_didataset(datasets["train"], "baby_id")
    
    # create the signal field (with chunking)
    datasets["train"].add_dynamic_item(
        audio_pipeline, takes=["file_path", "cut_length_interval_in_frames"], provides="sig"
    )

    return datasets, baby_id_encoder

if __name__ == "__main__":
    # This flag enables the inbuilt cudnn auto-tuner
    torch.backends.cudnn.benchmark = True

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Tensorboard logging
    if hparams["use_tensorboard"]:
        from speechbrain.utils.train_logger import TensorboardLogger

        hparams["tensorboard_train_logger"] = TensorboardLogger(
            hparams["tensorboard_logs_folder"]
        )

    datasets, l_encoder = prepare_data(hparams["data_dir"])
    
    # Initialize the Brain object to prepare for training.
    crybrain = CryBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    
    # if a pretrained model is specified, load it
    if "pretrained_embedding_model" in hparams:
        sb.utils.distributed.run_on_main(hparams["pretrained_embedding_model"].collect_files)
        hparams["pretrained_embedding_model"].load_collected(device=run_opts['device'])
    
    crybrain.fit(
        epoch_counter=crybrain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["val"],
        train_loader_kwargs=hparams["train_dataloader_options"],
        valid_loader_kwargs=hparams["val_dataloader_options"],
    )

