"""선택 과제 1: decoder block + NumPy 수동 역전파 + PyTorch CUDA 학습.

노트북에서 run_optional_gpt(CORPUS, HOLDOUT, ...) 호출.
GEN_NUMPY_STEPS=150, GEN_GPU_STEPS=500 환경 변수로 학습 길이 조정.
NumPy 구현은 autograd를 사용하지 않는다. CUDA 구현과 동일한 초기 모수로
forward/gradient를 비교한 다음 두 모델을 각각 학습한다.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


def layer_norm(x, gamma, beta, eps=1e-5):
    inv = 1 / np.sqrt(x.var(axis=-1, keepdims=True) + eps)
    normalized = (x-x.mean(axis=-1, keepdims=True))*inv
    return normalized*gamma+beta, (normalized, inv, gamma)


def layer_norm_backward(grad, cache):
    normalized, inv, gamma = cache
    axes = tuple(range(grad.ndim-1))
    dxhat = grad*gamma
    dx = inv*(dxhat-dxhat.mean(axis=-1, keepdims=True)
              -normalized*(dxhat*normalized).mean(axis=-1, keepdims=True))
    return dx, (grad*normalized).sum(axis=axes), grad.sum(axis=axes)


def gelu(x):
    u = np.sqrt(2/np.pi)*(x+.044715*x**3)
    return .5*x*(1+np.tanh(u))


def gelu_backward(grad, x):
    t = np.tanh(np.sqrt(2/np.pi)*(x+.044715*x**3))
    derivative = .5*(1+t)+.5*x*(1-t*t)*np.sqrt(2/np.pi)*(1+3*.044715*x*x)
    return grad*derivative


class NumpyGPT:
    """학습 가능한 위치 임베딩, pre-LN, causal MHA, residual, GELU FFN, weight tying."""

    def __init__(self, vocab_size, context=32, width=32, heads=4, seed=42):
        if width % heads:
            raise ValueError('width must be divisible by heads')
        self.context, self.width, self.heads = context, width, heads
        r = np.random.default_rng(seed)
        self.params = {
            'E': r.normal(0, .02, (vocab_size, width)),
            'P': r.normal(0, .02, (context, width)),
            'qkv_w': r.normal(0, 1/np.sqrt(width), (width, 3*width)),
            'qkv_b': np.zeros(3*width),
            'out_w': r.normal(0, 1/np.sqrt(width), (width, width)),
            'out_b': np.zeros(width),
            'ff1_w': r.normal(0, 1/np.sqrt(width), (width, 4*width)),
            'ff1_b': np.zeros(4*width),
            'ff2_w': r.normal(0, 1/np.sqrt(4*width), (4*width, width)),
            'ff2_b': np.zeros(width),
        }
        for name in ['ln1', 'ln2', 'lnf']:
            self.params[name+'_g'] = np.ones(width)
            self.params[name+'_b'] = np.zeros(width)

    def forward(self, ids):
        p = self.params
        B, T = ids.shape
        if not 0 < T <= self.context:
            raise ValueError('sequence length outside context')
        h, dh = self.heads, self.width//self.heads
        x = p['E'][ids]+p['P'][None, :T]
        a, ln1 = layer_norm(x, p['ln1_g'], p['ln1_b'])
        qkv = a @ p['qkv_w']+p['qkv_b']
        q, k, v = [z.reshape(B,T,h,dh).transpose(0,2,1,3) for z in np.split(qkv,3,axis=-1)]
        scores = q @ k.swapaxes(-2,-1)/np.sqrt(dh)
        mask = np.triu(np.ones((T,T), dtype=bool), 1)
        scores = np.where(mask, -np.inf, scores)
        exp = np.exp(scores-scores.max(axis=-1,keepdims=True))
        attn = exp/exp.sum(axis=-1,keepdims=True)
        merged = (attn @ v).transpose(0,2,1,3).reshape(B,T,self.width)
        residual = x+merged @ p['out_w']+p['out_b']
        b, ln2 = layer_norm(residual, p['ln2_g'], p['ln2_b'])
        pre = b @ p['ff1_w']+p['ff1_b']
        activated = gelu(pre)
        residual2 = residual+activated @ p['ff2_w']+p['ff2_b']
        final, lnf = layer_norm(residual2, p['lnf_g'], p['lnf_b'])
        logits = final @ p['E'].T
        cache = (ids, a, ln1, q, k, v, attn, merged, b, ln2, pre, activated, final, lnf)
        return logits, cache

    def loss_and_grad(self, ids, targets):
        logits, cache = self.forward(ids)
        ids, a, ln1, q, k, v, attn, merged, b, ln2, pre, activated, final, lnf = cache
        B, T, V = logits.shape
        p = self.params
        g = {key: np.zeros_like(value) for key,value in p.items()}
        shifted = logits-logits.max(axis=-1,keepdims=True)
        logp = shifted-np.log(np.exp(shifted).sum(axis=-1,keepdims=True))
        bi, ti = np.arange(B)[:,None], np.arange(T)[None,:]
        loss = float(-logp[bi,ti,targets].mean())
        dlogits = np.exp(logp)
        dlogits[bi,ti,targets] -= 1
        dlogits /= B*T
        g['E'] += dlogits.reshape(-1,V).T @ final.reshape(-1,self.width)
        dz = dlogits @ p['E']
        dz, g['lnf_g'], g['lnf_b'] = layer_norm_backward(dz,lnf)

        def linear_backward(dy, x, weight_name, bias_name):
            g[weight_name] += x.reshape(-1,x.shape[-1]).T @ dy.reshape(-1,dy.shape[-1])
            g[bias_name] += dy.sum(axis=(0,1))
            return dy @ p[weight_name].T

        dr = dz.copy()  # FFN residual shortcut
        da = linear_backward(dz,activated,'ff2_w','ff2_b')
        dpre = gelu_backward(da,pre)
        db = linear_backward(dpre,b,'ff1_w','ff1_b')
        db, g['ln2_g'], g['ln2_b'] = layer_norm_backward(db,ln2)
        dr += db
        dx = dr.copy()  # attention residual shortcut
        dm = linear_backward(dr,merged,'out_w','out_b')
        h, dh = self.heads, self.width//self.heads
        dm = dm.reshape(B,T,h,dh).transpose(0,2,1,3)
        dv = attn.swapaxes(-2,-1) @ dm
        dattn = dm @ v.swapaxes(-2,-1)
        ds = attn*(dattn-(dattn*attn).sum(axis=-1,keepdims=True))
        dq = ds @ k/np.sqrt(dh)
        dk = ds.swapaxes(-2,-1) @ q/np.sqrt(dh)
        dqkv = np.concatenate([z.transpose(0,2,1,3).reshape(B,T,self.width)
                               for z in (dq,dk,dv)],axis=-1)
        da = linear_backward(dqkv,a,'qkv_w','qkv_b')
        da, g['ln1_g'], g['ln1_b'] = layer_norm_backward(da,ln1)
        dx += da
        np.add.at(g['E'],ids.reshape(-1),dx.reshape(-1,self.width))
        g['P'][:T] += dx.sum(axis=0)
        return loss,g


def make_torch_gpt(reference, device, dtype=None):
    import torch
    import torch.nn.functional as F

    dtype = torch.float32 if dtype is None else dtype

    class TorchGPT(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.ParameterDict({key: torch.nn.Parameter(
                torch.tensor(value,device=device,dtype=dtype)) for key,value in reference.params.items()})

        def forward(self, ids):
            p = self.p
            B,T = ids.shape
            d,h = reference.width,reference.heads
            dh = d//h
            if T > reference.context:
                raise ValueError('context exceeded')
            def norm(x,name):
                return F.layer_norm(x,(d,),p[name+'_g'],p[name+'_b'],eps=1e-5)
            x = F.embedding(ids,p['E'])+p['P'][None,:T]
            a = norm(x,'ln1')
            qkv = a @ p['qkv_w']+p['qkv_b']
            q,k,v = [z.reshape(B,T,h,dh).transpose(1,2) for z in qkv.chunk(3,dim=-1)]
            scores = q @ k.transpose(-2,-1)/(dh**.5)
            causal = torch.ones(T,T,device=ids.device,dtype=torch.bool).triu(1)
            weights = torch.softmax(scores.masked_fill(causal,float('-inf')),dim=-1)
            merged = (weights @ v).transpose(1,2).contiguous().reshape(B,T,d)
            x = x+merged @ p['out_w']+p['out_b']
            a = norm(x,'ln2')
            hidden = F.gelu(a @ p['ff1_w']+p['ff1_b'],approximate='tanh')
            x = x+hidden @ p['ff2_w']+p['ff2_b']
            return norm(x,'lnf') @ p['E'].T

    return TorchGPT()


def gradient_checks(device):
    """실행 시 finite difference + 전 모수의 autograd gradient 비교. 학습 전 검증."""
    import torch
    import torch.nn.functional as F

    model = NumpyGPT(9,context=4,width=8,heads=2,seed=5)
    r = np.random.default_rng(2)
    ids,targets = r.integers(0,9,(2,2,4))
    loss,grads = model.loss_and_grad(ids,targets)
    for name in model.params:
        index = tuple(0 for _ in model.params[name].shape)
        original = model.params[name][index]
        eps = 1e-5
        try:
            model.params[name][index] = original+eps
            plus = model.loss_and_grad(ids,targets)[0]
            model.params[name][index] = original-eps
            minus = model.loss_and_grad(ids,targets)[0]
        finally:
            model.params[name][index] = original
        np.testing.assert_allclose((plus-minus)/(2*eps),grads[name][index],atol=2e-6,rtol=2e-4,
                                   err_msg=f'finite difference {name}')
    other = make_torch_gpt(model,device,dtype=torch.float64)
    xt = torch.tensor(ids,device=device,dtype=torch.long)
    yt = torch.tensor(targets,device=device,dtype=torch.long)
    logits = other(xt)
    torch_loss = F.cross_entropy(logits.reshape(-1,9),yt.reshape(-1))
    torch_loss.backward()
    np.testing.assert_allclose(torch_loss.item(),loss,atol=1e-9)
    for name in grads:
        np.testing.assert_allclose(other.p[name].grad.cpu().numpy(),grads[name],atol=2e-9,rtol=2e-6,
                                   err_msg=f'autograd {name}')
    edited = ids.copy(); edited[:,3] = (edited[:,3]+1)%9
    np.testing.assert_allclose(model.forward(ids)[0][:,:3],model.forward(edited)[0][:,:3],atol=1e-12)
    print('NumPy manual backward: finite differences, CUDA autograd, causality checks passed')


def sample_batch(data, batch_size, context, rng):
    if len(data) <= context:
        raise ValueError('corpus split must be longer than context')
    starts = rng.integers(0,len(data)-context,size=batch_size)
    x = np.stack([data[s:s+context] for s in starts])
    y = np.stack([data[s+1:s+context+1] for s in starts])
    return x,y


def numpy_adam_train(model,train_ids,steps=150,batch_size=8,lr=.003,seed=7):
    r = np.random.default_rng(seed)
    m = {k:np.zeros_like(v) for k,v in model.params.items()}
    v = {k:np.zeros_like(x) for k,x in model.params.items()}
    history = []
    for step in range(1,steps+1):
        x,y = sample_batch(train_ids,batch_size,model.context,r)
        loss,grads = model.loss_and_grad(x,y)
        norm = np.sqrt(sum(np.sum(g*g) for g in grads.values()))
        scale = min(1.,1./(norm+1e-12))
        for key in model.params:
            g = grads[key]*scale
            m[key] = .9*m[key]+.1*g
            v[key] = .999*v[key]+.001*g*g
            model.params[key] -= lr*(m[key]/(1-.9**step))/(np.sqrt(v[key]/(1-.999**step))+1e-8)
        if not np.isfinite(loss):
            raise FloatingPointError(f'NumPy loss at step {step}: {loss}')
        history.append(loss)
        if step == 1 or step%50 == 0:
            print(f'NumPy manual step={step}/{steps} loss={loss:.4f}',flush=True)
    return history


def torch_train(model,train_ids,context,device,steps=500,batch_size=32,lr=.003,seed=7):
    import torch
    import torch.nn.functional as F

    rng = np.random.default_rng(seed)
    optimizer = torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.01)
    use_amp = device.type == 'cuda'
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler('cuda',enabled=use_amp and amp_dtype == torch.float16)
    history = []
    model.train()
    for step in range(1,steps+1):
        x,y = sample_batch(train_ids,batch_size,context,rng)
        xt = torch.tensor(x,device=device,dtype=torch.long)
        yt = torch.tensor(y,device=device,dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type,dtype=amp_dtype,enabled=use_amp):
            logits = model(xt)
            loss = F.cross_entropy(logits.reshape(-1,logits.shape[-1]),yt.reshape(-1))
        if not torch.isfinite(loss).item():
            raise FloatingPointError(f'CUDA loss at step {step}: {loss.item()}')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        scaler.step(optimizer); scaler.update()
        history.append(loss.item())
        if step == 1 or step%50 == 0:
            print(f'{device} GPT step={step}/{steps} loss={loss.item():.4f}',flush=True)
    return history,optimizer


def evaluate_numpy(model,data):
    # 토큰 중복 없이 next-token loss를 계산하고 마지막 짧은 구간도 평가한다.
    loss_sum, count = 0.,0
    for s in range(0,len(data)-1,model.context):
        end = min(s+model.context,len(data)-1)
        logits,_ = model.forward(data[None,s:end])
        target = data[s+1:end+1]
        shifted = logits[0]-logits[0].max(axis=-1,keepdims=True)
        logp = shifted-np.log(np.exp(shifted).sum(axis=-1,keepdims=True))
        loss_sum -= logp[np.arange(len(target)),target].sum()
        count += len(target)
    return float(loss_sum/count)


def evaluate_torch(model,data,context,device):
    import torch
    import torch.nn.functional as F

    total,count = 0.,0
    model.eval()
    with torch.inference_mode():
        for s in range(0,len(data)-1,context):
            end = min(s+context,len(data)-1)
            ids = torch.tensor(data[None,s:end],device=device,dtype=torch.long)
            target = torch.tensor(data[s+1:end+1],device=device,dtype=torch.long)
            logits = model(ids)[0]
            total += F.cross_entropy(logits,target,reduction='sum').item()
            count += len(target)
    return total/count


def generate_numpy(model,prompt,stoi,chars,n_new=140,seed=8,temperature=.8):
    rng = np.random.default_rng(seed)
    ids = [stoi[c] for c in prompt]
    for _ in range(n_new):
        logits = model.forward(np.array([ids[-model.context:]]))[0][0,-1]/temperature
        p = np.exp(logits-logits.max()); p /= p.sum()
        ids.append(int(rng.choice(len(chars),p=p)))
    return ''.join(chars[i] for i in ids)


def generate_torch(model,prompt,stoi,chars,context,device,n_new=140,seed=8,temperature=.8):
    import torch

    generator = torch.Generator(device=device).manual_seed(seed)
    ids = [stoi[c] for c in prompt]
    model.eval()
    with torch.inference_mode():
        for _ in range(n_new):
            x = torch.tensor([ids[-context:]],device=device,dtype=torch.long)
            logits = model(x)[0,-1]/temperature
            ids.append(torch.multinomial(torch.softmax(logits,dim=-1),1,generator=generator).item())
    return ''.join(chars[i] for i in ids)


def run_optional_gpt(corpus,holdout,output_dir,device):
    import matplotlib.pyplot as plt
    import torch

    out = Path(output_dir); out.mkdir(exist_ok=True,parents=True)
    numpy_steps = int(os.environ.get('GEN_NUMPY_STEPS','150'))
    gpu_steps = int(os.environ.get('GEN_GPU_STEPS','500'))
    if min(numpy_steps,gpu_steps) < 1:
        raise ValueError('step counts must be positive')
    gradient_checks(device)
    chars = sorted(set(corpus+holdout))
    stoi = {c:i for i,c in enumerate(chars)}
    ids = np.array([stoi[c] for c in corpus],dtype=np.int64)
    split = int(.9*len(ids))
    train_ids,val_ids = ids[:split],ids[split:]
    test_ids = np.array([stoi[c] for c in holdout],dtype=np.int64)
    reference = NumpyGPT(len(chars))
    # 두 학습은 동일 초기 상태에서 독립적으로 시작. NumPy는 수동 gradient 학습 증명용.
    gpu_model = make_torch_gpt(reference,device)
    numpy_history = numpy_adam_train(reference,train_ids,steps=numpy_steps)
    gpu_history,optimizer = torch_train(gpu_model,train_ids,reference.context,device,steps=gpu_steps)
    metrics = {}
    for name,data in [('train',train_ids),('validation',val_ids),('holdout',test_ids)]:
        metrics[f'numpy_{name}_ppl'] = float(np.exp(evaluate_numpy(reference,data)))
        metrics[f'cuda_{name}_ppl'] = float(np.exp(evaluate_torch(gpu_model,data,reference.context,device)))
    prompt = '거대 언어 모델은 '
    samples = {'numpy':generate_numpy(reference,prompt,stoi,chars),
               'cuda':generate_torch(gpu_model,prompt,stoi,chars,reference.context,device)}
    for name,text in samples.items():
        print(f'[{name}] {text}')
    print(metrics)
    print('작은 말뭉치의 1블록 교육용 모델이다. 홀드아웃은 주제·문구가 유사하므로 일반 언어 능력을 뜻하지 않는다.')
    np.savez(out/'gpt_numpy.npz',**reference.params)
    config = dict(vocab_size=len(chars),context=reference.context,width=reference.width,heads=reference.heads)
    torch.save({'model':gpu_model.state_dict(),'optimizer':optimizer.state_dict(),
                'config':config,'chars':chars,'steps':gpu_steps,
                'seed':42},out/'gpt_cuda.pt')
    result = {'metrics':metrics,'samples':samples,'config':config,'chars':chars,
              'numpy_loss':numpy_history,'cuda_loss':gpu_history,
              'numpy_steps':numpy_steps,'cuda_steps':gpu_steps,'device':str(device)}
    (out/'gpt_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    fig,axes = plt.subplots(1,2,figsize=(10,3.5))
    axes[0].plot(numpy_history); axes[0].set(title='NumPy manual backward',xlabel='Step',ylabel='CE')
    axes[1].plot(gpu_history); axes[1].set(title=f'PyTorch {device} + AMP',xlabel='Step',ylabel='CE')
    fig.tight_layout(); fig.savefig(out/'gpt_training.png',dpi=160); plt.show()
    return result
