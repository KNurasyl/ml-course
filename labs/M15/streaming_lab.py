"""Actual incremental Vosk recognition and deterministic audio-turn cancellation."""
from __future__ import annotations
from dataclasses import dataclass, field
import array
import json
from pathlib import Path
import sys
import wave
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from projects.interview_trainer.speech import synthesize_wav


@dataclass
class AudioQueue:
    active_turn: int = 0
    queue: list = field(default_factory=list)
    dropped: int = 0
    def begin(self, turn):
        if turn <= self.active_turn: raise ValueError('turn must increase')
        self.active_turn=turn
        self.dropped+=len(self.queue);self.queue.clear()
    def receive(self, turn, chunk):
        if turn != self.active_turn: self.dropped+=1;return False
        self.queue.append((turn,chunk));return True
    def cancel(self):
        self.active_turn+=1;self.dropped+=len(self.queue);self.queue.clear()
    def pop(self):
        while self.queue:
            turn,chunk=self.queue.pop(0)
            if turn==self.active_turn:return chunk
            self.dropped+=1
        return None


def stream_recognize(path, pause_ms=500, chunk_ms=100):
    from vosk import Model, KaldiRecognizer, SetLogLevel
    SetLogLevel(-1)
    model=Model(str(ROOT/'models/vosk-model-small-ru-0.22'))
    recognizer=KaldiRecognizer(model,16000)
    events=[];silence=0;heard=False;sample_offset=0
    with wave.open(str(path),'rb') as stream:
        if (stream.getnchannels(),stream.getsampwidth(),stream.getframerate())!=(1,2,16000):
            raise ValueError('mono PCM16 16k required')
        while raw:=stream.readframes(16000*chunk_ms//1000):
            samples=array.array('h',raw)
            rms=(sum(int(v)*int(v) for v in samples)/max(1,len(samples)))**.5
            duration_ms=len(samples)/16
            sample_offset+=len(samples)
            silence=silence+duration_ms if rms<300 else 0
            heard=heard or rms>=300
            if recognizer.AcceptWaveform(raw):
                value=json.loads(recognizer.Result()).get('text','')
                events.append({'at_ms':sample_offset/16,'kind':'recognizer_final','text':value})
            else:
                value=json.loads(recognizer.PartialResult()).get('partial','')
                events.append({'at_ms':sample_offset/16,'kind':'partial','text':value})
            if heard and silence>=pause_ms:
                value=json.loads(recognizer.FinalResult()).get('text','')
                events.append({'at_ms':sample_offset/16,'kind':'endpoint_final','text':value,'pause_ms':silence})
                recognizer=KaldiRecognizer(model,16000);heard=False;silence=0
        events.append({'at_ms':sample_offset/16,'kind':'eof_final','text':json.loads(recognizer.FinalResult()).get('text','')})
    return events


def run():
    out=ROOT/'artifacts/M15-stream';out.mkdir(parents=True,exist_ok=True)
    paths=[synthesize_wav(text,out/f'fragment-{i}.wav') for i,text in enumerate(
        ['Я разработал сервис.','Проверил повторную отправку запроса.'])]
    combined=out/'two-turns.wav'
    with wave.open(str(combined),'wb') as target:
        target.setparams((1,2,16000,0,'NONE','not compressed'))
        for p in paths:
            with wave.open(str(p),'rb') as source:target.writeframes(source.readframes(source.getnframes()))
            target.writeframes(b'\x00\x00'*16000)
    events=stream_recognize(combined)
    assert any(e['kind']=='partial' for e in events)
    assert any(e['kind']=='endpoint_final' for e in events)
    gate=AudioQueue();gate.begin(1)
    assert gate.receive(1,paths[0].read_bytes())
    gate.cancel()
    assert not gate.receive(1,paths[1].read_bytes())
    assert gate.pop() is None
    gate.begin(3);assert gate.receive(3,paths[1].read_bytes())
    assert gate.pop()==paths[1].read_bytes()
    report={'events':events,'cancelled_chunks_dropped':gate.dropped,'live_microphone':False,
        'scope':'Real SAPI WAV and incremental Vosk. AudioQueue simulates delivery/playback admission; it does not play sound. Energy threshold is an educational baseline, not production VAD.'}
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report

if __name__=='__main__': print(json.dumps(run(),ensure_ascii=False,indent=2))
