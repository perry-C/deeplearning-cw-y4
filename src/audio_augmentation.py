
# For audio augmentation
from random import randint
from itertools import islice
import muda
# For format coneversion
import jams
# Import librosa for spectrogram making
import librosa
# Wav files chunking
from pydub import AudioSegment
# Serialization of data
import pickle

import math
import numpy as np
import sys
import dataset
import torch
import random
import os

# ==========================================================================
#                       AUDIO AUGMENTATION
#       Time Stretching: slowing down or speeding up the
#       original audio sample while keeping the same pitch
#       information. Time stretching was applied using the
#       multiplication factors 0.5, 0.2 for slowing down and 1.2,
#       1.5 for increasing the tempo.
#
#       Pitch Shifting: raising or lowering the pitch of an audio
#       sample while keeping the tempo unchanged. The applied
#       pitch shifting lowered and raised the pitch by 2 and 5
#       semitones.
#
#       For each deformation three segments have been randomly
#       chosen from the audio content. The combinations of the
#       two deformations with four different factors each resulted
#       thus in 48 additional data instances per audio file.
# ==========================================================================


# Augment previously segmented audios
def augment_audio(in_dir_name, out_dir_name, music_classes):

    # All transformations used for constructing a deformation pipeline
    pitch_deforms = [
        ("PitchShift(-2)", muda.deformers.PitchShift(-2)),
        ("PitchShift(-5)", muda.deformers.PitchShift(-5)),
        ("PitchShift(2)", muda.deformers.PitchShift(2)),
        ("PitchShift(5)", muda.deformers.PitchShift(5))
    ]

    # time_deforms = [
    #     ("TimeStretch(0.2)", muda.deformers.TimeStretch(0.2)),
    #     ("TimeStretch(0.5)", muda.deformers.TimeStretch(0.5)),
    #     ("TimeStretch(1.2)", muda.deformers.TimeStretch(1.2)),
    #     ("TimeStretch(1.5)", muda.deformers.TimeStretch(1.5))]

    # all_deforms = [[pd, td]
    #                for pd in pitch_deforms for td in time_deforms]

    # Make folders to store augmented audio files
    if os.path.exists(out_dir_name) == False:
        os.mkdir(out_dir_name)

    for c in music_classes:
        # This step is for the splitting of data into train and val set later
        # Label each wave file's name with either _val or _train, randomly
        # the percentage of train against val should be 1 : 3
        test_labels = ["_train"] * 300
        train_labels = ["_val"] * 100
        train_test_labels = train_labels + test_labels
        random.shuffle(train_test_labels)

        in_class_dir_name = f"{in_dir_name}/{c}"
        out_class_dir_name = f"{out_dir_name}/{c}"
        if os.path.exists(out_class_dir_name) == False:
            # Make folders to store files for each class
            os.mkdir(out_class_dir_name)
            # List all file names under the specific class folder
            for input_file in os.listdir(in_class_dir_name):
                input_file_name = os.fsdecode(input_file)

                # Apply all possible combinations of augmentations
                jam_in = muda.load_jam_audio(
                    jams.JAMS(), f"{in_class_dir_name}/{input_file_name}")

                # keeping only file name and separate away file extension
                only_file_name = os.path.splitext(input_file_name)[0]
                output_file_name = f"{out_class_dir_name}/{only_file_name}"
                output_file_name += train_test_labels.pop()
                for i, deform in enumerate(pitch_deforms):
                    pipeline = muda.Pipeline(steps=[deform])
                    for j, jam_out in enumerate(pipeline.transform(jam_in)):
                        audio_out = "{}.{}.{}".format(
                            output_file_name, i+j, "wav")
                        muda.save(audio_out,
                                  "{}.{}.{}".format(
                                      output_file_name, i+j, "jams"),
                                  jam_out, strict=False)


# Create_mel_spectrogram from the augmented audio file
def create_data_from_audio(in_dir_name, out_dir_name, music_classes):

    # (file name string, mel spectrogram tensor, class label, array of raw data)
    aug_train_dataset = []
    aug_val_dataset = []

    audio_length = 30000
    segment_length = 930
    hop_size = 470
    time_range_list = sliding_window_partition(
        audio_length, segment_length, hop_size)

    enum_classes = [(c, i) for i, c in enumerate(music_classes)]

    if os.path.exists(out_dir_name) == False:
        os.mkdir(out_dir_name)
    # Make folders to store segmented audio files
    for c, ci in enum_classes:
        in_class_dir_name = f"{in_dir_name}/{c}"
        out_class_dir_name = f"{out_dir_name}/{c}"
        if os.path.exists(out_class_dir_name) == False:
            # Make folders to store files for each class
            os.mkdir(out_class_dir_name)
            # List all audio file names under the specific class folder
            for input_audio in os.listdir(in_class_dir_name):
                input_audio_name = os.fsdecode(input_audio)
                # keeping only file name and separate away file extension
                extension_name = os.path.splitext(input_audio_name)[1]
                # Ignore the jams file
                if extension_name == ".jams":
                    continue
                only_file_name = os.path.splitext(input_audio_name)[0]
                # For the 63 chunks of each 30-seconds-augmented-audio-file, pick 15 random ones
                for i in range(15):
                    # Pick an index between 0 and 62
                    tr_index_list = list(range(len(time_range_list)))
                    random.shuffle(tr_index_list)
                    # Ensure we can never pick the same index twice
                    tr_index = tr_index_list.pop()
                    audio_in = AudioSegment.from_wav(
                        f"{in_class_dir_name}/{input_audio_name}")
                    audio_chunk = audio_in[time_range_list[tr_index]
                                           [0]:time_range_list[tr_index][1]]
                    audio_chunk_path = f"{out_class_dir_name}/{only_file_name}.{i}.wav"
                    audio_chunk.export(audio_chunk_path, format="wav")
                    mel = create_mel_spectrogram(audio_chunk_path)
                    # I really dont care about [0] and [3], so pass as empty
                    if "train" in audio_chunk_path:
                        aug_train_dataset.append(("", mel, ci, []))
                    if "val" in audio_chunk_path:
                        aug_val_dataset.append(("", mel, ci, []))

    return (aug_train_dataset, aug_val_dataset)


def create_mel_spectrogram(audio_chunk_path):
    segment_duration = 0.93
    hop_length = 512
    mel_height = 80
    mel_width = 80

    sample_rate = mel_width * hop_length // segment_duration

    y, sample_rate = librosa.load(audio_chunk_path, sr=sample_rate)
    y = librosa.util.fix_length(y, size=40449)
    # The height, i.e., your frequency resolution, only depends on the number of mel bands you decide to use.
    # You can manipulate it by passing the n_mels parameter to the melspectrogram function.
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sample_rate,
        win_length=1024,
        hop_length=hop_length,
        n_mels=mel_height,
    )
    mel = torch.reshape(torch.tensor(mel), (1, 80, 80))
    return mel


# This auxiliary function helps with partitioning each audio file into small chunks
def sliding_window_partition(audio_length, segment_size, hop_size):
    time_range_list = []
    t1 = 0
    t2 = 0
    while t2 < audio_length:
        # Semgent size is 0.97 s == 970 ms
        t2 = t1 + segment_size
        time_range_list.append((t1, t2))
        t1 += hop_size

    # Return a list of tuples, each is a time range (t1, t2)
    return time_range_list


def main():
    orig_audio_dir = "../data/audio_files"
    aug_audio_dir = "../data/augmented_audio_files"
    seg_audio_dir = "../data/segmented_audio_files"

    #  10 music genres (blues, disco...)
    music_classes = tuple([name for name in sorted(os.listdir(
        orig_audio_dir)) if os.path.isdir(os.path.join(orig_audio_dir, name))])

    augment_audio(orig_audio_dir, aug_audio_dir, music_classes)

    aug_train_dataset, aug_val_dataset = create_data_from_audio(
        aug_audio_dir, seg_audio_dir, music_classes)

    train_dataset = dataset.GTZAN("../data/train.pkl").dataset
    val_dataset = dataset.GTZAN("../data/val.pkl").dataset

    print(len(aug_train_dataset))
    print(len(aug_val_dataset))

    train_dataset.append(aug_train_dataset)
    val_dataset.append(aug_val_dataset)

    print(len(train_dataset))
    print(len(val_dataset))

    with open("../data/train_aug.pkl", "wb") as aug_train:
        pickle.dump(train_dataset, aug_train)

    with open("../data/val_aug.pkl", "wb") as aug_val:
        pickle.dump(train_dataset, aug_val)


if __name__ == "__main__":
    main()
