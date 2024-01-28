import os
import symusic
import numpy as np

switcher = {
    0: "C",
    1: "C#",
    2: "D",
    3: "D#",
    4: "E",
    5: "F",
    6: "F#",
    7: "G",
    8: "G#",
    9: "A",
    10: "A#",
    11: "B",
}

def convert_pitch_to_name_with_octave(pitch):
    if pitch < 0 or pitch > 127:
        return None
    name = pitch % 12

    name = switcher.get(name, None)
    octave = pitch // 12 - 2
    return str(name) + str(octave)

def single_midi_to_features(path: str):
    score = symusic.Score(path)
    # convert score to pianoroll
    score = score.resample(
        tpq=6
    )
    pianorolls = score.tracks[0].pianoroll(modes=["onset", "frame"], pitchRange=(0, 128), encodeVelocity=True)
    pianorolls = pianorolls.transpose([1, 0, 2])
    # store pianoroll as json
    pianorolls = pianorolls.tolist()

    return pianorolls

def get_midi_files(directory):
    midi_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".mid"):
                midi_files.append(os.path.join(root, file))
    return midi_files

def load_all_midis():
    # concate all pianorolls
    result = np.zeros((128, 2, 1))
    midi_paths = get_midi_files("dataset/src_001")
    i = 0
    # for path in midi_paths:
    #     i = i + 1
    #     print(f"Loading {i}/{len(midi_paths)}: {path}")
    #     pianorolls = single_midi_to_features(path)
    #     result = np.concatenate((result, pianorolls), axis=2)
    #     # add 1 EOF Token
    #     result = np.concatenate((result, np.zeros((128, 2, 1))), axis=2)
    path = "dataset/src_001/0.mid"
    pianorolls = single_midi_to_features(path)
    result = np.concatenate((result, pianorolls), axis=2)
    print(result.shape)
    return result.tolist()
    
    

