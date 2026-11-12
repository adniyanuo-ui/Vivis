"""Bounded, best-effort worker failure details for the journal and UI.

This describes observed errors; it does not change stop/recovery policy or
classify a communication timeout as proof of a broken motor or power supply.
"""
import json
import math
from pathlib import Path
import re
import signal


def clean(value,limit=1000):
    return re.sub(r'[\x00-\x1f\x7f]+',' ',str(value)).strip()[:limit]


def read_object(path):
    try:
        with Path(path).open('rb') as f:data=f.read(16*1024*1024+1)
        if len(data)>16*1024*1024:return {}
        value=json.loads(data)
        return value if isinstance(value,dict) else {}
    except (OSError,ValueError,UnicodeError,RecursionError):return {}


def exception_tail(path):
    try:
        with Path(path).open('rb') as f:
            f.seek(0,2);f.seek(max(0,f.tell()-32768));lines=f.read().decode('utf-8',errors='replace').splitlines()
    except OSError:return ''
    # Do not copy arbitrary stdout, source-code traceback lines, or conversation.
    for line in reversed(lines):
        if re.match(r'^[\w.]*(?:Error|Exception|Interrupt|Exit):\s*',line):return clean(line)
        if line.startswith(('SELF-CHECK FAIL ','PREPARE FAILED ')):return clean(line)
    return ''


def finite(value):return type(value) in (int,float) and math.isfinite(value)


def describe_failure(folder,returncode,program=None):
    folder=Path(folder);playback=read_object(folder/'playback_log.json')
    summary=read_object(folder/'summary.json');preparation=read_object(folder/'preparation.json')
    raw=exception_tail(folder.with_suffix('.log'))
    if not raw:raw=clean(summary.get('error') or playback.get('error') or preparation.get('error') or '')
    code='WORKER_EXIT';message='子程序异常退出，未取得具体异常；请查看详细日志'
    if 'Playback watchdog:' in raw:
        code='CONTROL_WATCHDOG';message='控制目标更新间隔超过100毫秒保护阈值'
    elif 'Feedback watchdog:' in raw:
        code='FEEDBACK_TIMEOUT';message='舵机有效反馈超时'
    elif 'Standby pursuit target exceeds command limits' in raw or 'Upstream trajectory reference exceeds motion limits' in raw:
        code='MOTION_TARGET_LIMIT';message='软件规划目标超出指令范围；检查目标生成过程及原始异常'
    elif 'Serial device is missing:' in raw:
        code='SERIAL_MISSING';message='USB串口设备不存在，请检查连接'
    elif 'access denied:' in raw or 'Permission denied' in raw:
        code='DEVICE_PERMISSION';message='设备访问权限不足'
    elif 'status 0x' in raw or 'status error 0x' in raw:
        code='SERVO_STATUS';message='舵机返回状态报警；ID和状态码见原始异常'
    elif 'read timeout' in raw or 'startup communication recovery failed' in raw:
        code='SERIAL_TIMEOUT';message='舵机通信读取超时或恢复失败'
    elif 'corrupt response' in raw or 'Damaged/missing feedback' in raw:
        code='SERIAL_DATA';message='舵机反馈缺失或校验失败'
    elif 'critical temperature' in raw:
        code='TEMPERATURE';message='温度反馈触发停止条件'
    elif any(text in raw for text in ('home did not settle','home position outside','device limits changed','startup corridor')):
        code='HOME_CHECK';message='回中或限位检查未通过'
    elif any(text in raw for text in ('Cannot open camera','视觉服务启动失败','Camera','camera')):
        code='VISION';message='相机或视觉子程序异常'
    elif 'Audio callback fault' in raw or 'audio' in raw.lower():
        code='AUDIO';message='音频子程序异常'
    elif raw:code='PROGRAM_EXCEPTION';message='程序执行异常'
    if returncode<0:
        code='WORKER_SIGNAL'
        try:name=signal.Signals(-returncode).name
        except ValueError:name=str(-returncode)
        message='子程序被信号终止：'+name
    label={'standby':'接待待机','interaction':'交互态'}.get(program,program or folder.name)
    result=dict(code=code,program=clean(label),returncode=returncode,message=message,
                raw_error=raw,log_path=str(folder.with_suffix('.log')),evidence=[])
    interval=playback.get('watchdog_interval_s')
    if finite(interval):
        result['watchdog_interval_ms']=interval*1000
        result['evidence'].append(f'最后控制目标更新间隔={interval*1000:.1f}毫秒')
    holds=playback.get('feedback_rescue_events')
    if code in ('CONTROL_WATCHDOG','FEEDBACK_TIMEOUT') and isinstance(holds,list) and holds and isinstance(holds[-1],dict):
        last=holds[-1];ids=last.get('critical_ids',[]);ages=last.get('feedback_ages_s',{})
        if isinstance(ids,list) and isinstance(ages,dict):
            readings=[f'ID{sid}={ages[str(sid)]*1000:.1f}毫秒' for sid in ids[:30]
                      if type(sid) is int and finite(ages.get(str(sid)))]
            result['evidence'].append('最近一次主动暂停发位置、尝试恢复反馈时：'+('，'.join(readings) or '反馈接近超时'))
    watchdog=playback.get('feedback_watchdog')
    if isinstance(watchdog,dict):
        for sid,state in list(watchdog.items())[:30]:
            if isinstance(state,dict) and finite(state.get('valid_feedback_age_s')):
                result['evidence'].append(f'ID{clean(sid,4)}有效反馈已间隔{state["valid_feedback_age_s"]*1000:.1f}毫秒')
    # Only the most recent attempts are relevant; do not flood the journal
    # with historical recovered errors or whole sensor/vision dumps.
    diagnostics=playback.get('serial_diagnostics')
    failures=diagnostics.get('read_failures') if isinstance(diagnostics,dict) else None
    if code in ('CONTROL_WATCHDOG','FEEDBACK_TIMEOUT','SERIAL_TIMEOUT','SERIAL_DATA') and isinstance(failures,list):
        recent=list(dict.fromkeys(clean(row.get('error',''),240) for row in failures[-3:] if isinstance(row,dict)))
        if recent:result['evidence'].append('末尾读取记录：'+'；'.join(recent))
    return result


def failure_lines(report):
    lines=[f'节目故障[{report["code"]}] {report["program"]}：{report["message"]}（退出码{report["returncode"]}）']
    if report['raw_error']:lines.append('原始异常：'+report['raw_error'])
    lines.extend('故障上下文：'+line for line in report['evidence'])
    lines.append('节目已停止；管理器将退出。详细日志：'+report['log_path'])
    return lines
