import json
from collections import Counter

def get_weight_style(d):
    wf = d.get('w_formal', 0)
    ws = d.get('w_scratchpad', 0)
    wk = d.get('w_socratic', 0)
    if wf > 0.8: return 'formal_pure'
    if ws > 0.8: return 'scratchpad_pure'
    if wk > 0.8: return 'socratic_pure'
    if wf > 0.4 and ws > 0.4: return 'formal_scratchpad_mix'
    if wf > 0.4 and wk > 0.4: return 'formal_socratic_mix'
    if ws > 0.4 and wk > 0.4: return 'scratchpad_socratic_mix'
    return 'other_mix'

def get_text_style(resp):
    resp = resp.lower()
    if 'let ' in resp or 'substituting ' in resp or 'equation ' in resp:
        return 'formal'
    if '1. ' in resp or '2. ' in resp or 'step-by-step' in resp:
        return 'scratchpad'
    if 'okay,' in resp or 'let\'s break' in resp:
        return 'socratic'
    return 'unknown'

def analyze_teacher(file_path):
    counts = Counter()
    with open(file_path) as f:
        for i, line in enumerate(f):
            if i >= 500: break
            counts[get_weight_style(json.loads(line))] += 1
    return counts

def analyze_student(file_path):
    counts = Counter()
    with open(file_path) as f:
        for i, line in enumerate(f):
            if i >= 500: break
            counts[get_text_style(json.loads(line)['response'])] += 1
    return counts

if __name__ == '__main__':
    t_file = 'data/teacher_holdout/teacher_gemma-3-12b-it_gsm8k_base_caa_mixed.jsonl'
    s_file = 'data/test/student_Qwen2.5-3B_base_tch_gsm8k_caa.jsonl'
    
    t_dist = analyze_teacher(t_file)
    print('Teacher Weight Distribution (first 500):', t_dist)
    
    s_dist = analyze_student(s_file)
    print('Student Text Style Distribution (first 500):', s_dist)
