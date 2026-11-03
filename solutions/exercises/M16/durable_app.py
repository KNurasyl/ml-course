"""Capstone key: single-process durable adapter for the existing local web app.

No model or grading policy changes. Failed writes leave the accepted in-memory
state intact. A new process verifies snapshots before serving any sessions.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from projects.interview_trainer import app
from projects.interview_trainer.core import Interview, QUESTIONS
from solutions.exercises.M16.persistence import save, restore


def snapshot(interview):
    return {'mode': interview.mode, 'turn': interview.turn,
            'topic': interview.topic_index, 'clarifications': int(interview.clarification),
            'history': deepcopy(interview.history)}


def configure(directory):
    """Restore first, then install an adapter. Call once before starting HTTP."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    class DurableInterview(Interview):
        def __post_init__(self):
            super().__post_init__()
            save(directory / f'{self.id}.json', snapshot(self))

        def commit(self, *args, **kwargs):
            # HTTP owns app.WORK: no second writer can inspect a partial turn.
            previous = deepcopy(self.__dict__)
            try:
                result = super().commit(*args, **kwargs)
                save(directory / f'{self.id}.json', snapshot(self))
                return result
            except Exception:
                self.__dict__.clear()
                self.__dict__.update(previous)
                raise

    recovered = {}
    for path in directory.glob('*.json'):
        if len(path.stem) != 32 or any(c not in '0123456789abcdef' for c in path.stem):
            raise ValueError('unexpected session filename')
        state = restore(path)
        # Recompute the state from accepted answers using this version's policy.
        # A policy/topic-order change requires a migration or explicit refusal.
        interview = Interview(state['mode'], id=path.stem)
        for row in state['history']:
            if not isinstance(row, dict):
                raise ValueError('invalid history row')
            question, _ = interview.prepare(row['answer'], row['turn'])
            if question != row['question'] or interview.topic_index != row['topic_index']:
                raise ValueError('topic policy changed; migration required')
            interview.commit(row['answer'], row['feedback'], row['sources'], row['seconds'],
                             row.get('generation'), row.get('assessment'))
        if snapshot(interview) != state:
            raise ValueError('snapshot and accepted history disagree')
        # Bypass the constructor to avoid overwriting a verified snapshot.
        durable = object.__new__(DurableInterview)
        durable.__dict__.update(interview.__dict__)
        recovered[durable.id] = durable
    app.Interview = DurableInterview
    app.SESSIONS.clear()
    app.SESSIONS.update(recovered)
    return recovered


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--state-dir', type=Path, default=ROOT / 'artifacts/M16-durable/sessions')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    configure(args.state_dir)
    server = app.ThreadingHTTPServer(('127.0.0.1', args.port), app.Handler)
    print(f'http://127.0.0.1:{args.port}; restored={len(app.SESSIONS)}; Ctrl+C to stop', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
