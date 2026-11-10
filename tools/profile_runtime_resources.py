#!/usr/bin/env python3
"""Sample Orin resources around an explicit command; never choose a robot/task.

Uses only the standard library. tegrastats and /proc are read at one-second
intervals. The child retains its own startup, motion, alarm and stop behavior.
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time


def proc_stat(pid):
    try:
        fields=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
        return dict(pid=pid,ppid=int(fields[1]),ticks=int(fields[11])+int(fields[12]),
                    born=int(fields[19]),rss_kib=int(fields[21])*os.sysconf('SC_PAGE_SIZE')/1024)
    except (OSError,ValueError,IndexError):return None


def process_tree(root):
    rows={}
    for path in Path('/proc').iterdir():
        if path.name.isdigit():
            item=proc_stat(int(path.name))
            if item:rows[item['pid']]=item
    selected={root}
    while True:
        expanded=selected|{pid for pid,item in rows.items() if item['ppid'] in selected}
        if expanded==selected:break
        selected=expanded
    return [rows[pid] for pid in sorted(selected) if pid in rows]


def memory_info():
    return {key:int(value.split()[0]) for line in Path('/proc/meminfo').read_text().splitlines()
            for key,value in [line.split(':',1)]}


def cpu_info():
    result={}
    for line in Path('/proc/stat').read_text().splitlines():
        fields=line.split()
        if not fields or not re.fullmatch(r'cpu\d*',fields[0]):continue
        values=list(map(int,fields[1:9]))  # guest time is already in user/nice.
        result[fields[0]]=(sum(values),values[3]+values[4])
    return result


def proportional_memory(pid):
    try:
        fields={line.split(':',1)[0]:int(line.split()[1])
                for line in (Path('/proc')/str(pid)/'smaps_rollup').read_text().splitlines()[1:]
                if ':' in line}
        return fields.get('Pss')
    except (OSError,ValueError,IndexError):return None


def role(pid):
    try:command=(Path('/proc')/str(pid)/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
    except OSError:return 'exited'
    if 'a2f_chunk_probe' in command:return 'a2f'
    if 'vivis.vision.service' in command:return 'vision'
    if 'vivis.interaction' in command:return 'controller'
    return 'other'


def parse_tegra(line):
    values={}
    for name,pattern in [('gpu_pct',r'GR3D_FREQ\s+(\d+)%'),('ram_mb',r'RAM\s+(\d+)/'),
                         ('ram_total_mb',r'RAM\s+\d+/(\d+)MB'),('swap_mb',r'SWAP\s+(\d+)/'),
                         ('cpu_temp_c',r'\bcpu@([\d.]+)C'),('soc_temp_c',r'\bsoc0@([\d.]+)C')]:
        match=re.search(pattern,line)
        if match:values[name]=float(match.group(1))
    values['rails_mw']={m[0]:float(m[1]) for m in re.findall(r'\b(VDD_\w+|VIN_\w+)\s+(\d+)mW',line)}
    return values


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--baseline',type=float,default=10.)
    parser.add_argument('--tail',type=float,default=10.)
    parser.add_argument('--timeout',type=float,default=240.)
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args(argv)
    command=args.command[1:] if args.command[:1]==['--'] else args.command
    if not command or not 0<=args.baseline<=60 or not 0<=args.tail<=60 or not 1<=args.timeout<=3600:
        parser.error('Provide an explicit command and valid baseline/tail/timeout')
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();observer_start=time.process_time();shutdown=False
    def interrupt(*_):
        nonlocal shutdown
        shutdown=True
    previous={sig:signal.signal(sig,interrupt) for sig in (signal.SIGINT,signal.SIGTERM)}
    files={name:(args.output/name).open('w') for name in ('samples.jsonl','tegrastats.log','application.log','events.jsonl')}
    tegra=child=None;threads=[];latest={};events=[];failure=None;app_exit=None;app_start=None
    metadata=dict(command=command,started_utc=datetime.now(timezone.utc).isoformat(),interval_s=1.,
                  cpu_count=os.cpu_count(),baseline_s=args.baseline,tail_s=args.tail,timeout_s=args.timeout)
    def event(kind,**fields):
        value=dict(kind=kind,monotonic=time.monotonic(),**fields)
        events.append(value);files['events.jsonl'].write(json.dumps(value)+'\n');files['events.jsonl'].flush()
    def tegra_reader():
        nonlocal latest
        for line in tegra.stdout:
            stamp=time.monotonic();files['tegrastats.log'].write(line);files['tegrastats.log'].flush()
            latest={**parse_tegra(line),'observed_monotonic':stamp}
    def app_reader():
        for line in child.stdout:
            files['application.log'].write(line);files['application.log'].flush()
            stripped=line.strip()
            if any(tag in stripped for tag in ('SELF-CHECK','HOME ','WARMUP','PREPARE COMPLETE',' START runtime-A2F','TELEMETRY')):
                event('application',message=stripped);print(stripped,flush=True)
    old_cpu=cpu_info();old_time=started;old_process={};count=0
    try:
        tegra=subprocess.Popen(['tegrastats','--interval','1000'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,start_new_session=True)
        threads.append(threading.Thread(target=tegra_reader,daemon=True));threads[-1].start()
        event('baseline_start')
        next_sample=started
        while True:
            now=time.monotonic()
            if shutdown:break
            if child is None and now-started>=args.baseline:
                child=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,start_new_session=True)
                app_start=time.monotonic();event('application_start',pid=child.pid)
                threads.append(threading.Thread(target=app_reader,daemon=True));threads[-1].start()
            if child is not None and app_exit is None and child.poll() is not None:
                app_exit=time.monotonic();event('application_exit',returncode=child.returncode)
            if app_start is not None and app_exit is None and now-app_start>args.timeout:
                raise TimeoutError('Application exceeded measurement timeout')
            if app_exit is not None and now-app_exit>=args.tail:break
            if now<next_sample:
                time.sleep(min(.05,next_sample-now));continue
            current=cpu_info();elapsed=max(.001,now-old_time);busy={}
            for name,(total,idle) in current.items():
                previous_cpu=old_cpu.get(name,(total,idle));dt=total-previous_cpu[0]
                busy[name]=100*(dt-(idle-previous_cpu[1]))/dt if dt>0 else 0.
            process=[];fresh={}
            for item in process_tree(child.pid) if child is not None and app_exit is None else []:
                key=(item['pid'],item['born']);delta=max(0,item['ticks']-old_process.get(key,0));fresh[key]=item['ticks']
                process.append(dict(pid=item['pid'],role=role(item['pid']),cpu_pct_one_core=100*delta/os.sysconf('SC_CLK_TCK')/elapsed,
                                    rss_kib=item['rss_kib'],pss_kib=proportional_memory(item['pid'])))
            mem=memory_info();record=dict(monotonic=now,elapsed_s=now-started,
                stage='baseline' if child is None else 'post' if app_exit else 'application',
                system_cpu_pct=busy['cpu'],core_cpu_pct={k:v for k,v in busy.items() if k!='cpu'},
                mem_total_kib=mem['MemTotal'],mem_available_kib=mem['MemAvailable'],
                mem_used_available_kib=mem['MemTotal']-mem['MemAvailable'],
                swap_used_kib=mem['SwapTotal']-mem['SwapFree'],processes=process,
                tree_cpu_pct_one_core=sum(p['cpu_pct_one_core'] for p in process),
                tree_pss_kib=sum(p['pss_kib'] or 0 for p in process),
                pss_complete=all(p['pss_kib'] is not None for p in process),tegra=latest.copy())
            files['samples.jsonl'].write(json.dumps(record)+'\n');files['samples.jsonl'].flush()
            old_time=now;old_cpu=current;old_process=fresh;count+=1;next_sample+=1.
            while next_sample<=now:next_sample+=1.
    except Exception as exc:
        failure=str(exc);raise
    finally:
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGINT)
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.terminate()
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:child.kill();child.wait()
        if tegra is not None:
            tegra.terminate()
            try:tegra.wait(timeout=3)
            except subprocess.TimeoutExpired:tegra.kill();tegra.wait()
        for thread in threads:thread.join(timeout=3)
        metadata.update(samples=count,application_pid=child.pid if child else None,
            application_start_monotonic=app_start,application_exit_monotonic=app_exit,
            application_returncode=child.returncode if child else None,error=failure,cancelled=shutdown,
            elapsed_s=time.monotonic()-started,observer_cpu_seconds=time.process_time()-observer_start)
        (args.output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
        for file in files.values():file.close()
        for sig,handler in previous.items():signal.signal(sig,handler)
    return 130 if shutdown else child.returncode if child is not None else 1


if __name__=='__main__':raise SystemExit(main())
