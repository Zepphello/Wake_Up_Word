import wave
import torchaudio
import torch
from neuralnet.dataset import get_featurizer
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class WakeWordEngine:
    def __init__(self, model_file, chunk=1024, sensitivity=20):
        self.chunk = chunk
        self.model = torch.jit.load(model_file)
        self.model.eval().to('cpu')
        self.featurizer = get_featurizer(sample_rate=16000)
        self.sensitivity = sensitivity

    def save(self, waveforms, fname="wakeword_temp"):
        wf = wave.open(fname, "wb")
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b''.join(waveforms))
        wf.close()
        return fname

    def predict(self, audio):
        with torch.no_grad():
            fname = self.save(audio)
            waveform, sr = torchaudio.load(fname)
            mfcc = self.featurizer(waveform).transpose(1, 2).transpose(0, 1)
            out = self.model(mfcc)
            pred = torch.round(torch.sigmoid(out))
            return pred.item()

    def run(self):
        dir_name = "./scripts/data/test/not_marvin"
        files_list = sorted(os.listdir(dir_name))

        count = 0
        sum_result_list = []
        index_result_list = []

        for file_index, file in enumerate(files_list[:10000]):
            wave_path = os.path.join(dir_name, file)
            wf = wave.open(wave_path, 'rb')

            sum_result = 0
            len_data = 1
            audio_q = []

            while len_data != 0:
                data = wf.readframes(self.chunk)
                len_data = len(data)
                audio_q.append(data)
                sum_result += self.predict(audio_q)

            sum_result_list.append(sum_result)
            if len(sum_result_list) > 3:
                sum_result_list.pop(0)

            if sum(sum_result_list) > self.sensitivity:
                index_result_list.append(file_index)
                if len(index_result_list) > 2:
                    index_result_list.pop(0)

                if (index_result_list[-1] - index_result_list[0]) == 1:
                    index_result_list = []
                    continue

                print(file)
                print(sum(sum_result_list))
                print(sum_result_list)
                print()

                count += 1

        print('Total 1:', count)


if __name__ == "__main__":
    class Args:
        def __init__(self,
                     model_file: str = None,
                     sensitivty: int = 20):
            self.model_file = model_file
            self.sensitivty = sensitivty


    args = Args(model_file='./scripts/data/checkpoint/wakeword_opt.pt')

    for chunk in [1024]:
        print('chunk:', chunk)
        wakeword_engine = WakeWordEngine(args.model_file, chunk, args.sensitivty)
        wakeword_engine.run()


