"""Full-model serve check of the NVFP4-W4A8 plugin.

Derived from the 2026-11-24 handoff worker (evidence/worker.py) with identical serve flags,
probes, 262080-token capacity request and 1000-request/300 s burst, so the result is
directly comparable to earlier NVFP4_A16 Marlin and MXFP4_A8 serve runs.
Changes: variants come from SERVE_VARIANTS; evidence root is new; fixtures are read from
the old root; VLLM_PLUGINS loads super_nvfp4_w4a8.
"""
import asyncio, hashlib, importlib.util, json, math, os, signal, subprocess, sys
import threading, time, traceback
from pathlib import Path

ROOT=Path('/cache/probe-evidence/nvfp4-w4a8-serve')
FIXROOT=Path('/cache/probe-evidence/longload-topk-20261124')
BF16='/cache/models/BF16/2dc98e2afe4face0e4ce40972a915c45368bd34a'
NVFP4='/cache/models/NVFP4/ff433f5493e25d631c9f12b5d55c674229923d02'
BASE='http://127.0.0.1:8000'
BURST_SECONDS=300
QUESTIONS=[('Chỉ trả lời kết quả: 17 + 25 bằng bao nhiêu?','42'),
 ('Only output the number: 12 multiplied by 13.','156'),
 ('Chỉ ghi số: 144 chia 12 bằng bao nhiêu?','12'),
 ('Only output the number: What is 19 squared?','361'),
 ('Chỉ ghi số: một hình chữ nhật dài 15, rộng 8 có diện tích bao nhiêu?','120'),
 ('Only output the number: 250 minus 87.','163'),
 ('Chỉ ghi số: 7 nhân 8 cộng 9 bằng bao nhiêu?','65'),
 ('Only output the number: a dozen plus two dozen.','36')]

def command(model,extra):
    eager=[] if '--no-enforce-eager-placeholder' in extra else ['--enforce-eager']
    return [sys.executable,'-m','vllm.entrypoints.cli.main','serve',model,
      '--served-model-name','super-probe','--trust-remote-code',
      '--tensor-parallel-size','1','--data-parallel-size','1','--pipeline-parallel-size','1',
      '--gpu-memory-utilization','0.96','--max-model-len','262144',
      '--max-num-seqs','2048','--max-num-batched-tokens','8192',
      '--kv-cache-dtype','fp8','--mamba-ssm-cache-dtype','float32',
      '--enable-chunked-prefill','--no-enable-prefix-caching',*eager,
      '--max-logprobs','20','--port','8000',*[e for e in extra if e!='--no-enforce-eager-placeholder']]


def cpu():
    import importlib.metadata as md
    result={'status':'FAIL','packages':{p:md.version(p) for p in ('vllm','torch','flashinfer-python')}}
    try:
        from vllm.entrypoints.openai.completion.protocol import CompletionRequest
        result['completion_logprob_fields']={k:str(v.annotation) for k,v in CompletionRequest.model_fields.items() if 'logprob' in k or 'echo' in k}
        assert 'logprobs' in CompletionRequest.model_fields
        assert Path(BF16,'config.json').exists() and Path(NVFP4,'config.json').exists()
        cfg=json.loads(Path(BF16,'config.json').read_text())
        result['model_config']={k:v for k,v in cfg.items() if ('mamba' in k or 'head' in k or 'layer' in k or 'intermediate' in k or 'position' in k) and k!='quantization_config'}
        # CLI help itself constructs a CUDA DeviceConfig in this image and is
        # not a valid CPU-only gate. Check the exact EngineArgs schema instead.
        import dataclasses
        from vllm.engine.arg_utils import EngineArgs
        fields={f.name for f in dataclasses.fields(EngineArgs)}
        for field in ('max_model_len','max_num_seqs','max_num_batched_tokens','max_logprobs',
                      'mamba_ssm_cache_dtype','enable_chunked_prefill','kv_cache_dtype'):
            assert field in fields,field
        result['engine_argument_schema_verified']=True
        spec=importlib.util.spec_from_file_location('prep','/probe/prepare_long.py')
        prep=importlib.util.module_from_spec(spec);spec.loader.exec_module(prep)
        result['fixture_manifest']=prep.prepare()
        result['status']='CPU_PASS'
    except Exception: result['error']=traceback.format_exc()
    return result

def jsonl(path,value):
    with path.open('a') as f: f.write(json.dumps(value,ensure_ascii=False)+'\n')

def metric_values(text):
    result={}
    for line in text.splitlines():
        if not line or line.startswith('#'): continue
        key=line.split('{',1)[0].split(' ',1)[0]
        if key.startswith('vllm:') and any(s in key for s in (
            'num_requests_running','num_requests_waiting','kv_cache_usage','gpu_cache_usage',
            'prompt_tokens_total','generation_tokens_total','num_preemptions_total',
            'request_success_total','request_queue_time_seconds','request_prefill_time_seconds')):
            try: result[key]=result.get(key,0)+float(line.rsplit(' ',1)[1])
            except ValueError: pass
    return result

def topk_probe(folder,tokenizer):
    import requests
    from concurrent.futures import ThreadPoolExecutor
    tasks=[]
    for q,answer in QUESTIONS:
        ids=tokenizer.apply_chat_template([{'role':'user','content':q}],tokenize=True,
                         add_generation_prompt=True,enable_thinking=False,return_dict=False)
        if hasattr(ids,'keys'): ids=ids['input_ids']
        if ids and isinstance(ids[0],list): ids=ids[0]
        assert ids and isinstance(ids[0],int),type(ids)
        tasks.append((q,answer,ids,''))
    arithmetic=tasks[6]
    # Each comparison uses an identical prefix across formats and batch sizes.
    # Additional fixed continuations are diagnostic, not free-running sequence alignment.
    for continuation in ('6','7','65','71'):
        tasks.append((arithmetic[0],None,arithmetic[2]+tokenizer.encode(continuation,add_special_tokens=False),continuation))
    rows=[]
    def one(task):
        question,answer,ids,continuation=task
        start=time.monotonic()
        r=requests.post(BASE+'/v1/completions',json={'model':'super-probe','prompt':ids,
          'max_tokens':1 if continuation else 8,'temperature':0,'logprobs':20},timeout=120)
        r.raise_for_status();body=r.json();choice=body['choices'][0]
        logs=choice.get('logprobs') or {}
        return {'question':question,'answer':answer,'fixed_continuation':continuation,
          'prefix_sha256':hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
          'prefix_tokens':len(ids),'elapsed_s':time.monotonic()-start,
          'text':choice['text'],'exact_match':None if answer is None else choice['text'].strip()==answer,
          'first_token_topk_probability_mass':sum(math.exp(p) for p in (logs.get('top_logprobs') or [{}])[0].values()),
          'response':body}
    # Warm C8 then repeat both C1/C8; top-k never enabled for the load benchmark.
    with ThreadPoolExecutor(max_workers=8) as pool: list(pool.map(one,tasks[:8]))
    for repeat in range(3):
        for concurrency in (1,8):
            with ThreadPoolExecutor(max_workers=concurrency) as pool: batch=list(pool.map(one,tasks))
            for row in batch:
                row.update(repeat=repeat,concurrency=concurrency);jsonl(folder/'topk.jsonl',row)
            rows.extend(batch)
    return {'requests':len(rows),'numeric_scores':[
      {'repeat':r,'concurrency':c,'correct':sum(x['exact_match'] is True for x in rows if x['repeat']==r and x['concurrency']==c),
       'n':8} for r in range(3) for c in (1,8)],
      'scope':'first-token same-prefix raw logprobs; no full-vocabulary KL claim'}

async def load_burst(folder,fixtures):
    import aiohttp
    start=time.monotonic();gate=asyncio.Event();records=[];sent=0
    connector=aiohttp.TCPConnector(limit=1100,limit_per_host=1100)
    timeout=aiohttp.ClientTimeout(total=None,sock_connect=30,sock_read=None)
    async with aiohttp.ClientSession(connector=connector,timeout=timeout,
                                    read_bufsize=65536) as session:
        async def one(i):
            nonlocal sent
            fixture=fixtures[i%len(fixtures)]
            payload={'model':'super-probe','prompt':fixture['tokens'],'max_tokens':2048,
                     'temperature':0,'stream':True,'stream_options':{'include_usage':True}}
            # Serialize before releasing the barrier, so no semaphore limits the 1000 clients.
            encoded=json.dumps(payload).encode()
            await gate.wait()
            row={'request_id':i,'source_row':fixture['row'],'input_tokens_submitted':len(fixture['tokens']),
                 'prefix_sha256':fixture['sha256'],'start_s':time.monotonic()-start,'status':'SUBMITTING',
                 'text':'','event_times_s':[],'usage':{}}
            records.append(row)
            sent+=1
            try:
                async with session.post(BASE+'/v1/completions',data=encoded,
                   headers={'Content-Type':'application/json','X-Request-Id':f'burst-{i}'}) as resp:
                    row['http_status']=resp.status
                    row['http_response_s']=time.monotonic()-start
                    if resp.status!=200: raise RuntimeError((await resp.text())[:600])
                    row['status']='STREAMING';buffer=b''
                    async for chunk in resp.content.iter_any():
                        buffer+=chunk
                        while b'\n' in buffer:
                            line,buffer=buffer.split(b'\n',1)
                            if not line.startswith(b'data: '): continue
                            data=line[6:].strip()
                            if data==b'[DONE]': continue
                            event=json.loads(data)
                            if event.get('usage'): row['usage']=event['usage']
                            for choice in event.get('choices',[]):
                                piece=choice.get('text') or ''
                                if piece:
                                    at=time.monotonic()-start
                                    if 'first_s' not in row: row['first_s']=at
                                    row['event_times_s'].append(at);row['text']+=piece
                                if choice.get('finish_reason'): row['finish_reason']=choice['finish_reason']
                    row['status']='COMPLETED'
            except asyncio.CancelledError:
                row['status']='CANCELLED_WINDOW_END';raise
            except Exception as exc: row['status']='ERROR';row['error']=str(exc)
            finally:
                row['end_s']=time.monotonic()-start
                jsonl(folder/'requests.jsonl',row)
        tasks=[asyncio.create_task(one(i)) for i in range(1000)]
        # Let all coroutines prepare their payloads and reach the shared barrier.
        await asyncio.sleep(0)
        start=time.monotonic();gate.set()
        (folder/'burst-start.json').write_text(json.dumps({'wall_utc_s':time.time(),'count':1000,
          'window_s':BURST_SECONDS,'prefix_cache':False,'topk':False,'max_output':2048,
          'total_input_tokens_submitted':sum(len(fixtures[i%len(fixtures)]['tokens']) for i in range(1000))}))
        done,pending=await asyncio.wait(tasks,timeout=BURST_SECONDS)
        for task in pending: task.cancel()
        await asyncio.gather(*pending,return_exceptions=True)
        for task in done: task.result()
    return {'attempted':sent,'http_200':sum(r.get('http_status')==200 for r in records),
      'completed':sum(r['status']=='COMPLETED' for r in records),
      'cancelled':sum(r['status']=='CANCELLED_WINDOW_END' for r in records),
      'errors':sum(r['status']=='ERROR' for r in records),'measurement_wall_s':time.monotonic()-start,
      'output_tokens_completed':sum(r['usage'].get('completion_tokens',0) for r in records),
      'input_tokens_completed':sum(r['usage'].get('prompt_tokens',0) for r in records),
      'scope':'finite 5-minute burst; partial output and in-flight prefill also recorded by server counters'}

def gpu(attempt):
    import requests
    from transformers import AutoTokenizer
    folder=ROOT/attempt;folder.mkdir(parents=True,exist_ok=False)
    result={'status':'RUNNING','attempt':attempt,'start_utc_s':time.time(),'models':[],
            'call_id':os.environ.get('PROBE_MODAL_CALL_ID')}
    result['device']=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total','--format=csv,noheader'],text=True)
    fixtures=json.loads((FIXROOT/'fixtures.json').read_text())
    tokenizer=AutoTokenizer.from_pretrained(BF16,local_files_only=True,trust_remote_code=True)
    proc=None
    def save(): (folder/'result.json').write_text(json.dumps(result,indent=2))
    try:
        variants={'NVFP4_W4A8':(NVFP4,['--quantization','super_nvfp4_w4a8']),
                  'NVFP4_A16':(NVFP4,['--moe-backend','marlin'])}
        for name in os.environ.get('SERVE_VARIANTS','NVFP4_W4A8').split(','):
            path,extra=variants[name]
            if os.environ.get('SERVE_CUDA_GRAPHS')=='1': extra=extra+['--no-enforce-eager-placeholder']
            sub=folder/name;sub.mkdir()
            row={'format':name,'command':command(path,extra),'status':'STARTING','start_utc_s':time.time()}
            result['models'].append(row);save()
            log=(sub/'server.log').open('w')
            proc=subprocess.Popen(row['command'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
              text=True,env={**os.environ,'VLLM_PLUGINS':'super_nvfp4_w4a8','SUPER_W4A8_CHECK':'1'},start_new_session=True)
            def collect(p=proc,f=log,label=name):
                for line in p.stdout:
                    f.write(line);f.flush()
                    print(label,line.rstrip(),flush=True)
            reader=threading.Thread(target=collect,daemon=True);reader.start()
            ready=False
            for _ in range(210):
                if proc.poll() is not None: break
                try:
                    if requests.get(BASE+'/health',timeout=2).status_code==200: ready=True;break
                except requests.RequestException: pass
                time.sleep(4)
            if not ready:
                row['status']='ENGINE_INITIALIZATION_FAILED';row['returncode']=proc.poll();save()
                # Preserve failure at the requested config, do not silently lower max-num-seqs.
            else:
                row['ready_utc_s']=time.time();row['ready_wall_s']=row['ready_utc_s']-row['start_utc_s'];save()
                row['topk']=topk_probe(sub,tokenizer);save()
                # Explicit long-context acceptance request before queue stress.
                capacity_ids=json.loads((FIXROOT/'capacity_tokens.json').read_text())
                t=time.monotonic()
                cap=requests.post(BASE+'/v1/completions',json={'model':'super-probe','prompt':capacity_ids,
                     'temperature':0,'max_tokens':1},timeout=180)
                row['capacity_probe']={'input_tokens':len(capacity_ids),'wall_s':time.monotonic()-t,
                  'http_status':cap.status_code,'response':cap.json()};save()
                stop=threading.Event()
                def monitor():
                    while not stop.is_set():
                        entry={'wall_utc_s':time.time()}
                        try:
                            response=requests.get(BASE+'/metrics',timeout=5)
                            entry['metrics']=metric_values(response.text)
                            jsonl(sub/'metrics-raw.jsonl',{'wall_utc_s':entry['wall_utc_s'],'text':response.text})
                            entry['smi']=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,power.draw','--format=csv,noheader'],text=True,timeout=5).strip()
                        except Exception as exc: entry['error']=str(exc)
                        jsonl(sub/'timeline.jsonl',entry);stop.wait(2)
                mon=threading.Thread(target=monitor,daemon=True);mon.start()
                try: row['burst']=asyncio.run(load_burst(sub,fixtures))
                finally: stop.set();mon.join(timeout=15)
                row['status']='MEASURED'
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try: proc.wait(timeout=25)
                except subprocess.TimeoutExpired: os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            reader.join(timeout=5);log.close();proc=None;save()
        result['status']='COMPLETE_ATTEMPT'
    except Exception:
        result['status']='FAIL';result['error']=traceback.format_exc();print(result['error'],flush=True)
    finally:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid,signal.SIGTERM)
            try: proc.wait(timeout=20)
            except subprocess.TimeoutExpired: os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        result['end_utc_s']=time.time();save()
    return result

if __name__=='__main__':
    def terminate(signum,frame): raise TimeoutError('Bounded native deadline; cleanup server')
    signal.signal(signal.SIGTERM,terminate)
    stage,attempt,target=sys.argv[1:]
    r=cpu() if stage=='cpu' else gpu(attempt)
    Path(target).write_text(json.dumps(r,indent=2))
