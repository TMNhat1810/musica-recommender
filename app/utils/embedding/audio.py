import librosa
import torch
from app.configs import device
import torch.nn.functional as F
from torchaudio.pipelines import WAV2VEC2_ASR_BASE_960H


__w2v_model = WAV2VEC2_ASR_BASE_960H.get_model().to(device)


def load_waveform(file_path: str, sr=16000):
    y, _ = librosa.load(file_path, sr=sr)
    y_trimmed, _ = librosa.effects.trim(y)
    waveform = torch.from_numpy(y_trimmed).unsqueeze(0)
    return waveform


def chunk_waveform(waveform, chunk_size=16000 * 10):
    return [
        waveform[:, i : i + chunk_size] for i in range(0, waveform.size(1), chunk_size)
    ]


def w2v_feature_extract(file_path: str):
    waveform = load_waveform(file_path)
    features = []

    with torch.inference_mode():
        for chunk in chunk_waveform(waveform):
            if chunk.shape[1] < 400:
                continue
            chunk = chunk.to(device)
            fx = __w2v_model(chunk)[0]
            pooled = fx.mean(dim=0).cpu()
            features.append(pooled.mean(dim=1))

    torch.cuda.empty_cache()

    return torch.cat(features, dim=0)


def pad_or_truncate(
    tensor: torch.Tensor, target_len: int, pad_value: float = 0.0
) -> torch.Tensor:
    current_len = tensor.shape[0]

    if current_len > target_len:
        return tensor[:target_len]
    elif current_len < target_len:
        pad_len = target_len - current_len
        return F.pad(tensor, (0, pad_len), value=pad_value)
    else:
        return tensor


def extract_audio_embedding(audio):
    fx = w2v_feature_extract(audio)
    return pad_or_truncate(fx, 20000).tolist()
