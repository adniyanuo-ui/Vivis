"""Estimate word and clause times locally against a supplied Chinese script."""
import argparse
import difflib
import json
import re
from pathlib import Path

from faster_whisper import WhisperModel

parser = argparse.ArgumentParser()
parser.add_argument('audio')
parser.add_argument('text')
parser.add_argument('output')
args = parser.parse_args()
root = Path(__file__).resolve().parent.parent
text = Path(args.text).read_text().strip()

def norm(value):
    return ''.join(re.findall(r'[\u4e00-\u9fffA-Za-z0-9]', value)).lower()

model = WhisperModel(str(root / 'assets/models/whisper-base'), device='cpu',
                     compute_type='int8', cpu_threads=8, local_files_only=True)
segments, _ = model.transcribe(args.audio, language='zh', beam_size=5,
    word_timestamps=True, initial_prompt=text, vad_filter=False,
    condition_on_previous_text=False)
words, segment_rows, recognized, char_times = [], [], [], []
for segment in segments:
    segment_rows.append({'start': segment.start, 'end': segment.end, 'text': segment.text})
    for word in segment.words or []:
        words.append({'word': word.word, 'start': word.start, 'end': word.end,
                      'probability': word.probability})
        chars = norm(word.word)
        for i, char in enumerate(chars):
            recognized.append(char)
            char_times.append((word.start + (word.end-word.start)*i/len(chars),
                               word.start + (word.end-word.start)*(i+1)/len(chars)))
reference, decoded, mapping = norm(text), ''.join(recognized), {}
for block in difflib.SequenceMatcher(None, reference, decoded, autojunk=False).get_matching_blocks():
    for j in range(block.size):
        mapping[block.a+j] = block.b+j
coverage = len(mapping)/len(reference)
if coverage < .85:
    raise ValueError(f'Insufficient transcript agreement: {coverage:.3f}: {decoded}')
clauses, offset = [], 0
for phrase in re.split(r'[，。！？；——]+', text):
    clean = norm(phrase)
    if not clean:
        continue
    matched = [char_times[mapping[i]] for i in range(offset, offset+len(clean)) if i in mapping]
    if len(matched)/len(clean) < .6:
        raise ValueError('Uncertain clause timing: '+phrase)
    clauses.append({'text': phrase, 'start': min(t[0] for t in matched),
                    'end': max(t[1] for t in matched),
                    'matched_character_fraction': len(matched)/len(clean)})
    offset += len(clean)
report = {'method': 'local Whisper word timing estimates matched to supplied text; not ground-truth forced alignment',
          'audio': str(Path(args.audio).resolve()), 'reference': str(Path(args.text).resolve()),
          'recognized_text': decoded, 'matched_character_fraction': coverage,
          'segments': segment_rows, 'words': words, 'content_phrases': clauses}
Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
print(json.dumps({'coverage': coverage, 'clauses': clauses}, ensure_ascii=False, indent=2))
