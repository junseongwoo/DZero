# DOT 언어를 이미지로 변환하려면 
# dot 명령을 수동으로 실행해야하니 번거롭다! 
# -> dot 명령까지 한 번에 해주는 함수 만드려고함 

import os 
import subprocess

def plot_dot_graph(output, verbose=True, to_file='graph.png'):
    dot_graph = get_dot_graph(output, verbose)

    # 1. dot 데이터를 파일에 저장
    tmp_dir = os.path.join(os.path.expanduser('~'), '.dezero')
    if not os.path.exists(tmp_dir): # ~/.dezero 디렉터리가 없다면 새로 생성 
        os.mkdir(tmp_dir)
    graph_path = os.path.join(tmp_dir, 'tmp_graph.dot')

    with open(graph_path, 'w') as f :
        f.write(dot_graph)

    # 2. dot 명령 호출 
    extension = os.path.splitext(to_file)[1][1:] # 확장자(png, pdf 등)
    cmd = 'dot {} -T {} -o {}'.format(graph_path, extension, to_file)
    subprocess.run(cmd, shell=True) # 파이썬에서 외부 프로그램을 호출하기 위해 사용

# 계산 그래프에서 DOT 언어로 변환 
# Variable 인스턴스를 건네면 인스턴스의 내용을 DOT 언어로 바꿔
# 문자열로 반환 
def _dot_var(v, verbose=False):
    dot_var = '{} [label="{}", color=orange, style=filled]\n'

    name = '' if v.name is None else v.name
    if verbose and v.data is not None:
        if v.name is not None:
            name += ': '
        name += str(v.shape) + ' ' + str(v.dtype)
    
    # id 함수 : 주어진 객체의 id를 반환
    #          - 객체 id는 다른 객체와 중복되지 않기 때문에 노드의 id로 사용하기에 적합
    # fotmat 메서드 : 문자열에 등장하는 {} 부분을 메서드 인수로 건넨 객체로 차례로 바꿔줌
    return dot_var.format(id(v), name)

# Dezero 함수를 DOT 언어로 변환하는 함수 
def _dot_func(f):
    dot_func = '{} [labe="{}", color=lightblue, style=filled, shape=box]\n'
    txt = dot_func.format(id(f), f.__class__.__name__)

    dot_edge = '{} -> {}\n'
    for x in f.inputs:
        txt += dot_edge.format(id(x), id(f))

    for y in f.outputs:
        txt += dot_edge.format(id(f), id(y()))
    return txt

# 계산 그래프를 DOT 언어로 변환 
def get_dot_graph(output, verbose=True):
    txt = '' 
    funcs = []
    seen_set = set() 

    def add_func(f):
        if f not in seen_set:
            funcs.append(f)
            seen_set.add(f)

    add_func(output.creator)
    txt += _dot_var(output, verbose)

    while funcs:
        func = funcs.pop()
        txt += _dot_func(func)
        for x in func.inputs:
            txt += _dot_var(x, verbose)

            if x.creator is not None:
                add_func(x.creator)

    return 'digraph g {\n' + txt + '}'

 