"""Controlled inner pause in a real synthesized phrase and two streaming passes."""
from pathlib import Path
import array,json,sys,wave
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from projects.interview_trainer.speech import synthesize_wav
from labs.M15.streaming_lab import stream_recognize,run
def trim(path):
    with wave.open(str(path),'rb') as audio:raw=audio.readframes(audio.getnframes())
    samples=array.array('h',raw);active=[i for i,x in enumerate(samples) if abs(x)>500]
    if not active:raise ValueError('silent source')
    return samples[max(0,active[0]-160):min(len(samples),active[-1]+160)].tobytes()
def main():
    run();out=ROOT/'artifacts/M15-hw';out.mkdir(parents=True,exist_ok=True)
    pieces=[trim(synthesize_wav(text,out/f'part-{i}.wav')) for i,text in enumerate(['Я разработал','новый сервис заказов.'])]
    path=out/'internal-pause.wav'
    with wave.open(str(path),'wb') as audio:
        audio.setparams((1,2,16000,0,'NONE','not compressed'))
        audio.writeframes(pieces[0]);audio.writeframes(b'\x00\x00'*5600)
        audio.writeframes(pieces[1]);audio.writeframes(b'\x00\x00'*16000)
    report={'inserted_pause_ms':350,'trim_threshold':500,'reports':{},'scope':'SAPI sources trimmed around energy, not a microphone benchmark; ASR may change text.'}
    for pause in (300,500):
        events=stream_recognize(path,pause_ms=pause,chunk_ms=50)
        report['reports'][str(pause)]={'events':events,'endpoints':sum(e['kind']=='endpoint_final' for e in events)}
    assert all(r['endpoints']>=1 for r in report['reports'].values())
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report
if __name__=='__main__': print(json.dumps(main(),ensure_ascii=False,indent=2))
