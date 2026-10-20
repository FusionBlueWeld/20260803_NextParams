"""Live progress artifact for this local verification run; exits at 18 summaries."""
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[2]/'v005/trials'
NAMES={'cut_cu':'Cu切断','cut_al':'Al切断','form_cu':'Cu成形','form_al':'Al成形','assembly':'組立','welding':'溶接'}


def main():
    previous=-1
    while True:
        completed=0
        text=['# 電気接続・単一工程検証の進捗','',
              '途中の最良値を最適到達と解釈しない。完了欄の初到達は、終了後の全格子監査による。',
              f"更新時刻：{time.strftime('%Y-%m-%d %H:%M:%S')}",'',
              '|工程|seed 42|seed 0|seed 1|','|---|---|---|---|']
        for stage,title in NAMES.items():
            cells=[]
            for seed in [42,0,1]:
                folder=ROOT/f'trial_interconnect_stage_{stage}_s{seed}_20261010_focus_on'
                try:
                    if (folder/'summary.json').exists():
                        r=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
                        cells.append(f"{r['stop_status']} {r['measurements']}条件／初到達 {r['first_optimum_measurement'] or 'なし'}")
                        completed+=1
                    elif (folder/'ledger.json').exists():
                        rows=json.loads((folder/'ledger.json').read_text(encoding='utf-8'))
                        cells.append(f"計算中 {rows[-1]['total_measurements']}条件")
                    else: cells.append('未開始／準備中')
                except (OSError,json.JSONDecodeError): cells.append('更新中')
            text.append(f"|{title}|{'|'.join(cells)}|")
        text += ['',f'完了：{completed}/18試行。最終検証結果はREPORT.mdに保存する。']
        out=ROOT/'_interconnect_stage_20261010_focus_on/PROGRESS.md'
        temp=out.with_suffix('.tmp');temp.write_text('\n'.join(text)+'\n',encoding='utf-8');temp.replace(out)
        if completed!=previous: print(json.dumps({'completed':completed,'target':18}),flush=True);previous=completed
        if completed==18: break
        time.sleep(20)


if __name__=='__main__': main()
