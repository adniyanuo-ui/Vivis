"""Estimate local word timings, then match them to the supplied Chinese script."""
import difflib,json,re
from pathlib import Path
from faster_whisper import WhisperModel
root=Path(__file__).resolve().parent.parent
text=(root/'5.txt').read_text().strip()
def norm(s):return ''.join(re.findall(r'[\u4e00-\u9fffA-Za-z0-9]',s)).lower()
model=WhisperModel(str(root/'assets/models/whisper-base'),device='cpu',compute_type='int8',cpu_threads=8,local_files_only=True)
segments,info=model.transcribe(str(root/'assets/audio/wz_v3.0.wav'),language='zh',beam_size=5,word_timestamps=True,initial_prompt=text,vad_filter=False,condition_on_previous_text=False)
words=[];segment_rows=[]
for s in segments:
 segment_rows.append({'start':s.start,'end':s.end,'text':s.text})
 for w in s.words or []:words.append({'word':w.word,'start':w.start,'end':w.end,'probability':w.probability})
recognized=[];char_times=[]
for w in words:
 chars=norm(w['word']);count=len(chars)
 for i,c in enumerate(chars):
  recognized.append(c);char_times.append((w['start']+(w['end']-w['start'])*i/count,w['start']+(w['end']-w['start'])*(i+1)/count))
reference=norm(text);decoded=''.join(recognized);mapping={}
for block in difflib.SequenceMatcher(None,reference,decoded,autojunk=False).get_matching_blocks():
 for j in range(block.size):mapping[block.a+j]=block.b+j
coverage=len(mapping)/len(reference)
assert coverage>0.85,('Transcript alignment is too uncertain',coverage,decoded)
phrases=['北京翌脑科技有限公司','专注于人工智能仿生安全与具身智能领域','是全球领先的生物特征安全蓝军服务商','我们以仿生安全攻防技术为核心能力','同时面向各垂直细分领域','提供智能仿生交互机器人的定制化解决方案']
rows=[]
for phrase in phrases:
 clean=norm(phrase);a=reference.index(clean);b=a+len(clean)
 matched=[char_times[mapping[i]] for i in range(a,b) if i in mapping]
 assert len(matched)/len(clean)>0.6
 rows.append({'text':phrase,'start':min(t[0] for t in matched),'end':max(t[1] for t in matched),'matched_character_fraction':len(matched)/len(clean)})
keywords=['北京翌脑科技有限公司','人工智能','仿生安全','具身智能','全球领先','核心能力','定制化解决方案']
protected=[]
for keyword in keywords:
 clean=norm(keyword);offset=0
 while (a:=reference.find(clean,offset))>=0:
  matched=[char_times[mapping[i]] for i in range(a,a+len(clean)) if i in mapping]
  if matched:protected.append({'text':keyword,'start':min(t[0] for t in matched),'end':max(t[1] for t in matched)})
  offset=a+len(clean)
report={'method':'faster-whisper base local word timing estimates, matched to 5.txt; not ground-truth forced alignment','reference':'5.txt','audio':'wz.m4a','recognized_text':decoded,'matched_character_fraction':coverage,'segments':segment_rows,'words':words,'content_phrases':rows,'protected_keywords':protected}
(root/'outputs/wz_content_alignment.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'coverage':coverage,'phrases':rows,'keywords':protected},ensure_ascii=False,indent=2))
