"""Read-only, bounded projection of the current manager's local event journal."""
import json
import time
from datetime import datetime
from pathlib import Path

from vivis.paths import ROOT


def interaction_snapshot(root=ROOT):
    result = dict(available=False, status='交互管理器未运行', entries=[])
    try:
        state = json.loads((root / 'outputs/runtime/program_status.json').read_text())
        if not isinstance(state, dict):
            raise ValueError('Invalid manager state')
        session = Path(state['output']).resolve()
        session.relative_to((root / 'outputs/p1').resolve())
        path = (session / 'events.jsonl').resolve()
        path.relative_to(session)
        age = time.monotonic() - float(state['updated_monotonic'])
        active = 0 <= age < 5 and state.get('phase') not in ('stopped', 'fault')
        modes = dict(standby='待机', interaction='交互态', task='任务执行', preparing='准备中', voice_recovering='语音恢复中')
        result.update(available=True, active=active, session=session.name,
                      status=modes.get(state.get('phase'), state.get('phase', '未知')) + ('' if active else '（已停止或状态过期）'),
                      self_check=state.get('self_check', {}).get('last_message', ''))
        with path.open('rb') as stream:
            stream.seek(0, 2)
            start = max(0, stream.tell() - 512 * 1024)
            stream.seek(start)
            if start:
                stream.readline()  # Discard a potentially partial first record.
            lines = stream.read(512 * 1024).splitlines(keepends=True)
        offset = time.time() - time.monotonic()
        for line in lines:
            if not line.endswith(b'\n'):
                continue  # Writer may be appending this record now.
            try:
                event = json.loads(line)
                if not isinstance(event, dict):
                    continue
                kind = event.get('kind')
                detail = event.get('detail') or {}
                if not isinstance(detail, dict):
                    continue
                role, text = '系统', ''
                if kind == 'cloud_transcript':
                    role = {'user': detail.get('identity_label') or 'other-other', 'assistant': '小帅'}.get(detail.get('role'), '系统')
                    text = detail.get('text', '')
                elif kind == 'preparation':
                    text = event.get('last_message', '')
                elif kind in ('start', 'worker_ready'):
                    text = ('准备：' if kind == 'start' else '状态：') + modes.get(event.get('mode'), str(event.get('mode', ''))) + (' · ' + event['program'] if event.get('program') else '')
                elif kind in ('cloud_error', 'speech_fault', 'mouth_error', 'voice_warning', 'voice_health'):
                    text = detail.get('message') or detail.get('error') or detail.get('reason') or ('语音已恢复' if detail.get('available') else '语音暂不可用')
                elif kind == 'intent':
                    role, text = '控制识别', event.get('text', '')
                elif kind == 'closed':
                    text = '交互管理器已退出' + ('：' + event['error'] if event.get('error') else '')
                if isinstance(text, str) and text.strip():
                    stamp = datetime.fromtimestamp(float(event['monotonic']) + offset).strftime('%H:%M:%S')
                    result['entries'].append(dict(time=stamp, role=role, text=text[:4000]))
            except (ValueError, TypeError, KeyError, OverflowError, OSError):
                continue
        result['entries'] = result['entries'][-100:]
        try:
            identity_path=(session/'voice/identity.json').resolve()
            identity_path.relative_to(session)
            with identity_path.open('rb') as stream:
                identity=json.loads(stream.read(8192))
            if active and 0<=time.monotonic()-identity['observed_monotonic']<6:
                result['identity_status']='身份（逻辑-实时）：'+(identity.get('identity_label') or 'other-other')
        except (OSError, ValueError, TypeError, KeyError):
            pass
    except (OSError, ValueError, TypeError, KeyError):
        return dict(available=False, status='暂无可读取的交互日志', entries=[])
    return result
