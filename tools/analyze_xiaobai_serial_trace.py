#!/usr/bin/env python3
"""Offline validation of application-level Xiaobai serial traces; no hardware IO."""
import argparse
import collections
import json
from pathlib import Path


def analyze(trace):
    groups=[]
    for event in trace['events']:
        if event['direction']=='tx':
            groups.append(dict(tx=event['hex'],t_ns=event['t_ns'],
                               written=event['written'],rx=''))
        elif event['direction']=='rx' and groups:
            groups[-1]['rx']+=event['hex']
    counts=collections.Counter();bad_ids=collections.Counter();reads=[];bad_tx=[]
    for group in groups:
        tx=bytes.fromhex(group['tx']);counts['tx_packets']+=1
        valid_tx=(len(tx)>=6 and tx[:2]==b'\xff\xff' and tx[3]==len(tx)-4
                  and sum(tx[2:])%256==255 and group['written']==len(tx))
        if not valid_tx:bad_tx.append(group)
        if len(tx)!=8 or tx[4]!=2:continue
        counts['reads']+=1;rx=bytes.fromhex(group['rx'])
        group.update(id=tx[2],address=tx[5],requested=tx[6],size=len(rx),
                     checksum_ok=len(rx)>=6 and sum(rx[2:])%256==255,
                     header_ok=len(rx)>=6 and rx[:3]==tx[:3],
                     length_ok=len(rx)==tx[6]+6 and rx[3]==len(rx)-4)
        valid=all(group[k] for k in ['checksum_ok','header_ok','length_ok'])
        if not valid:
            counts['bad_read_replies']+=1;bad_ids[tx[2]]+=1
        else:
            group['status']=rx[4]
            if rx[4]:counts['valid_nonzero_status']+=1
            if tx[5]==56 and tx[6]==15:
                group.update(temperature=rx[12],voltage_raw=rx[11])
                if rx[12]>=80:counts['valid_temperature_ge80']+=1
        reads.append(group)
    counts['bad_tx_packets']=len(bad_tx)
    return dict(method='Concatenate RX chunks until next TX; validate exact response. No byte repair. Application capture does not prove electrical bus waveform.',
                counts=dict(counts),bad_reply_ids=dict(bad_ids),uart_delta=trace.get('uart_delta'),
                dropped_events=trace['dropped_events'],bad_tx=bad_tx,reads=reads)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace',type=Path)
    args=parser.parse_args()
    result=analyze(json.loads(args.trace.read_text()))
    out=args.trace.with_name('packet_analysis.json')
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['reads','bad_tx']},indent=2))
