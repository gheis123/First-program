#거스름돈
n=1260
count=0
array=[500,100,50,10]
for coin in array:
    count+=n//coin #해당 화폐로 거슬러 줄 수 있는 동전 세기
    n%=coin
print(count)

#1이 될때까지
#1. N에서 1을 뺀다
#2. N을 K로 나눈다.
#ex) N=17, K=4 : 3번
#1<=N<=100,000 2<=K<=100,000
#N에 대해 최대한 많이 나누기를 수행하면 된다.
n,k=map(int,input().split())
result=0

while True:
    target=(n//k)*k #N이 K로 나눠떨어지는 수가 될때까지 
    result+=(n-target)
    n=target
    #N이 K보다 작을 때(더이상 나눌수 없을 때) 반복문 
    if n<k:
        break
    result+=1
    #K로 나누기.
    n//=k
#마지막으로 남은 수에 대해 1씩 빼기.
result+=(n-1)
print(result)

#02984=>((((0+2)*9)*8)*4)=576 -->'+' or '*'
data=input()
#첫 번째 문자를 숫자로 변경하여 대입
result=int(data[0])

for i in range(1,len(data)):
    #두수중 하나라도 1 or 0인경우, 곱하기 보다는 더하기
    num=int(data[i])
    if num<=1 or result<=1:
        result+=num
    else:
        result*=num

print(result)

#모험가길드
#공포도가 X인 모험가는 반드시 X명 이상구성한 모험가 그룹에 참여가능
#여행 떠날 수 있는 그룹수의 최대값
#2 3 1 2 2
#1 2 3<->2 2
#몇명의 모험가는 마을에 그대로 남아있어도 된다.
#오름차순 정렬 이후 공포도가 가장 낮은 모험가부터 확인.
n=int(input())
data=list(map(int,input().split()))
data.sort()

result=0 #총 그룹수
count=0 #현재 그룹에 포함된 모험가의 수

for i in data: #공포도를 낮은 것부터 하나씩 확인
    count+=1 #현재 그룹에 해당 모험가를 포함시키기
    if count>=i: #현재 그룹에 포함된 모험가의 수가 현재 공포도 이상이라면, 그룹 결성
        result+=1#총 그룹수 증가시키기
        count=0 #현재 그룹에 포함된 모험가의 수 초기화.
print(result)

#구현(Implementation)
#구현이란, 머릿속에 있는 알고리즘을 소스코드로 바꾸는 과정
#풀이를 떠올리기는 쉬우나, 소스코드로 옮기기 어려움.

#examples.
#실수연산을 다루고, 특정 소수점까지 출력하는 문제
#문자열을 특정 기능에 따라 끊어 처리하는 문제
#적절 라이브러리를 찾아 사용해야하는 문제
#알고리즘은 간단한데 코드가 지나칠 만큼 길어지는 문제

for i in range(5):
    for j in range(5):
        print('(',i,'j',')',end=' ')
    print()


#동북서남
dx=[0,-1,0,1]
dy=[1,0,-1,0]
x,y=2,2
for i in range(4):
    nx=x+dx[i]
    ny=y+dy[i]
    print(nx,ny)

#여행가 A는 N*N 정사각형 공간위에 서있다.
#가장 왼쪽 위 좌표는 (1,1)이며, 가장 오른쪽아래는 (N,N)이다.
#L: 왼쪽으로 한칸 이동
#R: 오른쪽으로 한칸 이동
#U: 위로 한칸 이동
#D: 아래로 한칸 이동
#공간을 벗어나는 움직임은 무시된다.
n=int(input())
x,y=1,1
plans=input().split()

dx=[0,0,-1,1]
dy=[-1,1,0,0]
move_types=['L','R','U','D']

for plan in plans:
    for i in range(len(move_types)):
        if plan==move_types[i]:
            nx=x+dx[i] #이동후의 좌표 구하기.
            ny=y+dy[i]
    if nx<1 or ny<1 or nx>n or ny>n: #공간 밖 무시
        continue
    x,y=nx,ny

print(x,y)

#정수 N이 입력되면 00시 00분 00초부터 N시 59분 59까지의 모든 시각중
#3이 하나라도 포함되는 모든 경우의 수를 구하는 프로그램
#00시 00분 03초 세야함 
#00시 02분 55초 세지않아야함.
#가능한 모든 경우의 수: 24*60*60=86,400가지의 경우의수.
#완전탐색(Brute Forcing)
#가능한 모든 경우의 수를 검사하는 방식이다.
h=int(input())
count=0
for i in range(h+1):
    for j in range(60):
        for k in range(60):
            if '3' in str(i)+str(j)+str(k):
                count+=1
print(count)

#왕실의 나이트
#a b c d e f g h 
#1 ..... 8 (세로선)
#a1==>2가지

input_data=input()
row=int(iput_data[1])
column=nt(ord(input_data[0]))-int(ord('a'))+1
steps=[(-2,-1),(-1,-2),(1,-2),(2,-1),(2,1),(1,2),(-1,2),(-2,1)]
result=0
for step in steps:
    next_row=row+step[0]
    next_column=colun+step[1]

    if next_row >=1 and next_row<=8 and next_column>=1 and next_column<=8:
        result+=1
print(result)

#K1KA5CB7->ABCKK13 출력하라.
data=input()
result=[]
value=0

for x in data: #문자를 하나씩 확인
    if x.isalpha(): #알파벳인 경우 결과리스트 삽입
        result.append(x)
    else:
        value+=int(x) #숫자는 따로 더하기
result.sort()
#알파벳을 오름차로 정렬하기

#숫자가 하나라도 존재하는 경우 가장 뒤에 삽입.
if value!=0:
    result.append(str(value))

#최종결과 출력(리스트를 문자열로 변환하여 출력)
print(''.join(result))
            
#DFS, BFS
#탐색이란 많은 양의 데이터 중에서 원하는 데이터 찾는 과정
#코딩에서 매우 많이 등장하므로 반드시 숙지해야한다.

#스택: 먼저 들어온 데이터가 나중에 나가는 형식의 자료구조
#입구와 출구가 동일한 형태로 스택 시각화(박스쌓기)
#삽입과 삭제!
#+5 +2 +3 +7 - +1 +4 -
#5237 ->523 ->52314 ->5231    <= 입구 및 출구의 방향.

stack=[]
stack.append(5)
stack.append(2)
stack.append(3)
stack.append(7)
stack.pop()
stack.append(1)
stack.append(4)
stack.pop()

print(stack[::-1]) #최상단 원소부터 출력 [1,3,2,5]
print(stack) #최하단 원소부터 출력. [5,2,3,1]

#큐: 먼저 들어온 데이터가 먼저 나가는 형식의 자료구조
#큐는 입구와 출구가 모두 뚫려있는 터널과 같은 형태로 시각화
#+5 +2 +3 +7 - +1 +4 -
#7325 ->732->41732->4173

from collections import deque #시간적인 유익성.
queue=deque()

queue.append(5)
queue.append(2)
queue.append(3)
queue.append(7)
queue.popleft()
queue.append(1)
queue.append(4)
queue.popleft()

print(queue) #deque([3,7,1,4])
queue.reverse()
print(queue) #deque([4,1,7,3])


#재귀함수: 자기자신을 다시 호출하는 함수.
def recursive_function(i):
    if i==100:
        return
    print(i,"재귀 함수에서",i+1,"번째 재귀함수 호출!")
    recursive_function(i+1)
    print(i,"번째 재귀함수를 종료합니다.")

recursive_function(1)
#재귀 함수의 종료 조건을 반드시 명시해줘야 한다.

#factorial
def factorial_iterative(n):
    result=1
    for i in range(1,n+1):
        result*=i
    return result

def factorial_recursive(n):
    if n<=1:
        return 1
    return n*factorial_recursive(n-1)

print("반복적으로 구현:",factorial_iterative(5))
print("재귀적으로 구현:",factorial_recursive(5))

#최대공약수 계산(유클리드 호제법)
#A,B에 대해 A를 B로 나눈나머지를 R이라고 하면
#A,B의 최대공약수는 B,R의 최대공약수와 같다.
#ex: GCD(192,162)=>(162,30)=>(30,12)=>(12,6) : 6이 최대공약수
def gcd(a,b):
    if a%b==0:
        return b
    else:
        return gcd(b,a%b)

print(gcd(192,162))

#모든 재귀함수는 반복문을 이용하여 동일기능 구현 가능하다
#재귀 함수가 반복문보다 유리한 경우도 있고 불리할 수도 있음.
#스택 사용해야 할 경우, 구현상 스택 라이브러리 대신 재귀함수 이용이 많음.
#(컴퓨터가 함수를 연속으로 호출시, 컴퓨터 메모리 내부 스택 프레임에 쌓여서)

#DFS(Depth-First-Search)
#스택 자료구조 혹은 재귀를 사용
#탐색 시작 노드를 스택에 삽입하고 방문처리함
#스택의 최상단 노드에 방문하지 않은 인접한 노드가 하나라도 있으면
#그 노드를 스택에 넣고 방문 처리함. 
#방문하지 않은 인접노드가 없으면 스택에서 최상단 노드를 꺼냄
#더이상 2번의 과정을 수행할 수 없을 때까지 반복한다.

def dfs(graph,v,visited):
    visited[v]=True #현재 노드를 방문처리.
    print(v,end=' ')
    #현재 노드와 연결된 다른 노드를 재귀적으로 방문.
    for i in graph[v]:
        if not visited[i]:
            dfs(graph,i,visited)

graph=[ #각 노드가 연결된 정보를 표현(2차원 리스트)
    [],
    [2,3,8],
    [1,7],
    [1,4,5],
    [3,5],
    [3,4],
    [7],
    [2,6,8],
    [1,7]
]
#각 노드가 방문된 정보를 표현(1차원 리스트)
visited=[False]*9
#정의된 DFS 함수 호출
dfs(graph,1,visited)

#BFS(Breadth-First Search)
#너비 우선 탐색: 가까운 노드부터 우선적 탐색하는 알고리즘
#큐 자료구조를 사용
#탐색 시작 노드를 큐에 삽입후, 방문처리
#큐에서 노드를 꺼낸뒤 해당 노드의 인접 노드중 방문하지 않은 노드를
#모두 큐에 삽입하고 방문처리함.
#더이상 2번의 과정 수행할 수 없을 때까지 반복한다.

from collections import deque

def bfs(graph,start,visited):
    queue=deque([start]) #큐 구현 위해 deque 라이브러리 사용
    visited[start]=True #현재 노드를 방문처리
    while queue:#큐가 빌때까지 반복
        v=queue.popleft() 
        print(v,end=' ')
        for i in graph[v]:
            if not visited[i]: #아직 방문하지 않은 인접 원소를 큐에 삽입.
                queue.append(i)
                visited[i]=True
graph=[ #각 노드가 연결된 정보를 표현(2차원 리스트)
    [],
    [2,3,8],
    [1,7],
    [1,4,5],
    [3,5],
    [3,4],
    [7],
    [2,6,8],
    [1,7]
]
visited=[False]*9 #각 노드의 방문 정보를 표현함(1차원 리스트)
bfs(graph,1,visited) #1 2 3 8 7 4 5 6

#음료수 얼려먹기
#00110
#00011
#11111
#00000  
#-->아이스크림의 개수: 3개가 된다(이 예시에서)
#구멍이 뚫린 부분은 0, 칸막이가 존재하는 부분은 1로 표시된다.

#N(세로) M(가로) 1<=N,M<=1,000
#4 5
#00110
#00011
#11111
#00000

#DFS,BFS로 해결가능.
#특정 지점 주변 상하좌우를 살펴본 뒤에 주변 지점중 값이 '0'이면서 아직 방문하지 않은 지점있으면 해당 지점 방문
#방문한 지점에서 다시 상하좌우를 살펴보면서 방문을 진행하는 과정 반복하면, 
#연결된 모든 지점 방문가능
#모든 노드에 대해 1~2 번의 과정 반복->방문하지 않은 지점의수를 카운팅하자!

def dfs(x,y):#dfs로 특정 노드 방문하고 연결된 모든 노드들도 방문
    if x<=-1 or x>=n or y<=-1 or y>=m:
        #주어진 범위를 벗어나는 경우 즉시 종료
        return False
    #현재 노드를 아직 방문하지 않았다면
    if graph[x][y]==0: #해당 노드 방문 처리
        graph[x][y]=1
        #상하좌우의 위치들도 모두 재귀적 호출
        dfs(x-1,y)
        dfs(x,y-1)
        dfs(x+1,y)
        dfs(x,y+1)
        return True
    return False

n,m=map(int,input().split())
graph=[]
#2차원 리스트의 맴 정보 입력 받기.
for i in range(n):
    graph.append(list(map(int,input())))
result=0
#모든 노드(위치)에 대하여 음료수 채우기
for i in range(n):
    for j in range(m):
        #현재 위치에서 DFS 수행.
        if dfs(i,j)==True:
            result+=1
print(result) #정답 출력하기.

#미로 탈출: NxM 크기의 직사각형 형태 미로에 갇힘
#처음 위치는 (1,1)이며 미로의 출구는 (N,M)에 존재 한칸씩 이동가능함.
#괴물이 있으면0, 괴물이 없으면 1
#미로는 반드시 탈출가능 형태 제시
#움직여야 하는 최소의 칸개수는?
#시작칸과 마지막칸을 모두 포함하여 계산한다.

#BFS는 시작지점에서 가까운 노드부터 차례대로 그래프의 모든 노드를 탐색
#상하좌우 연결 모든 노드로의 거리가 1로 동일함
#(1,1) 지점부터 BFS 수행하여 모든 노드의 최단 거리값을 기록하기
#110
#010
#011

def bfs(x,y):
    queue=deque()
    queue.append(x,y)

    #큐가 빌때까지 반복
    while queue:
        x,y=queue.popleft() 
        #현 위치에서 4가지 방향으로의 위치 확인
        for i in range(4):
            nx=x+dx[i]
            ny=y+dy[i]
            #미로 찾기 공간 벗어나면 무시하기
            if nx<0 or nx>=n or ny<0 or ny>=m:
                continue
            #벽인 경우 무시
            if graph[nx][ny]==0:
                continue
            #해당 노드를 처음 방문하는 경우에만 최단거리 기록
            if graph[nx][ny]==1:
                graph[nx][ny]=graph[x][y]+1
                queue.append((nx,ny))
         #가장 오른쪽 아래까지의 최단 거리반환
        return graph[n-1][m-1]

from collections import deque
n,m=map(int,input().split())
graph=[]
for i in range(n):
    graph.append(list(map(int,input())))

#이동할 네가지 방향 정의(상하좌우)
dx=[-1,1,0,0]
dy=[0,0,-1,1]
print(bfs(0,0))


#정렬이란 데이터를 특정 기준에 따라 순서대로 나열하는 것
#적절한 정렬 알고리즘이 공식처럼 사용된다.

#선택정렬: 처리되지 않은 데이터중 가장 작은 데이터를 선택해
#맨 앞에 있는 데이터와 바꿈.
array=[7,5,9,0,3,1,6,2,4,8]
for i in range(len(array)):
    min_index=i #가장 작은 원소의 인덱스
    for j in range(i+1,len(array)):
        if array[min_index]>array[j]:
            min_index=j
    array[i],array[min_index]=array[mun_index],array[i] #swap

print(array) #O(N^2)
#0 1 2 3 4 5 6 7 8 9

#삽입정렬: 처리되지 않은 데이터를 하나씩 골라 적절한 위치에 삽입
#선택 정렬에 비해 구현 난이도가 높지만, 일반적으로 더 효율적!

array=[7,5,9,0,3,1,6,2,4,8]
for i in range(1,len(array)):
    for j in range(i,0,-1): #인덱스 i부터 1까지 1씩 감소하며 repeat
        if array[j]<array[j-1]: #한 칸씩 왼쪽 이동
            array[j],array[j-1]=array[j-1],array[j]
        else: #자기보다 작은 데이터를 만나면 그 위치에서 멈춘다.
            break
print(array) #시간복잡도 O(N^2)

#퀵 정렬: 기준 데이터 설정하고 그 기준보다 큰 데이터와 작은 데이터 위치 변경
#가장 많이 사용됨
#병합 정렬과 더불어 대부분 프로그래밍 언어의 정렬 라이브러리의 근간이 됨
#가장 기본적인 퀵 정렬은 첫번째 데이터를 기준 데이터(Pivot)으로 설정한다.
#피벗을 기준으로 데이터 묶음을 나누는 작업을 Divide라고 한다.
#O(NlogN)
#이미 정렬된 배열에 대해서 퀵을 수행하면 최악의 경우 O(N^2)
array=[5,7,9,0,3,1,6,2,4,8]
def quict_sort(array,start,end):
    if start>=end: #원소가 1개인 경우 종료
        return
    pivot=start #피벗은 첫번째 원소
    left=start+1
    right=end
    while(left<=right):
        #피벗보다 큰 데이터를 찾을때까지 반복
        while(left<=end and array[left]<=array[pivot]):
            left+=1
        #피벗보다 작은 데이터를 찾을때까지 반복
        while(right>start and array[right] >=array[pivot]):
            right-=1
        if (left>right): #엇갈리면 작은 데이터와 피벗 교체
            array[right],array[pivot]=array[pivot],array[right]
        else: #엇갈리지 않으면 작은데이터와 큰 데이터 교체
            array[left],array[right]=array[right],array[left]
        quict_sort(array,start,right-1)
        quict_sort(array,right+1,end)
quick_sort(array,0,len(array)-1)
print(array)

def quick_sort1(array):
    if len(array)<=1:
        return array
    pivot=array[0] #피벗은 첫번째 원소
    tail=array[1:] #피벗을 제외한 리스트
    left_side=[x for x in tail if x<=pivot] #분할된 왼쪽부분
    right_side=[x for x in tail if x>pivot] #분할된 오른쪽 부분
    #분할이후 왼쪽부분과 오른쪽부분에서 각각 정렬수행후
    #전체의 리스트를 반환한다.
    return quick_sort1(left_side)+[pivot]+quick_sort1(right_side)
print(quick_sort1(array))

#계수정렬: 특정 조건이 부합할때 사용: 매우 fast
#계수 정렬은 데이터의 크기 범위가 제한되어 정수형태 표현 가능할 때
#데이터의 개수가 N, 데이터(양수) 중 최대값이 K일때 O(N+K) 최악이라도
#정렬할 데이터: 7 5 9 0 3 1 6 2 9 1 4 8 0 5 2
#인덱스와 개수 counting
array=[7,5,9,0,3,1,6,2,9,1,4,8,0,5,2]
count=[0]*(max(array)+1) #모든 범위를 포함하는 배열선언(모든 값 0 초기화)
for i in range(len(array)):
    count[array[i]]+=1 # 각 데이터에 해당하는 인덱스 값 증가
for i in range(len(array)): #리스트에 기록된 정렬 정보 확인
    for j in range(count[i]):
        print(i, end= ' ') #띄어쓰기를 구분으로 등장 횟수만큼 인덱스 출력

#계수정렬은 동일 값을 갖는 데이터가 여러개 등장시 효과적이다.
#O(N+K) 시간복잡도와 공간복잡도 모두!

#선택정렬: 아이디어가 매우 간단
#삽입정렬: 데이터가 거의 정렬되어 있을때 매우 빠름
#퀵정렬: 대부분의 경우 가장 적합하며, 충분히 빠르다
#계수정렬:데이터의 크기가 한정되어 있는 경우에만 사용가능(매우 빠름)

from random import randint
import time

array=[]
for _ in range(10000):
    array.append(randint(1,100))
start_time=time.time()
for i in range(len(array)):
    min_index=i
    for j in range(i+1,len(array)):
        if array[min_index]>array[j]:
            min_index=j
        array[i],array[min_index]=array[index],array[i]

end_time=time.time()
print(end_time-start_time) #수행시간 출력

# 두 배열의 원소교체
# A배열의 모든 원소합을 최대로 만드는 것이 목표이다.
# 매번 배열 A에서 가장 작은 원소를 골라 배열 B의 가장 큰 원소와 교체
# 가장 먼저 배열 a,b가 주어지면 오름차순 정렬하고 b는 내림차순으로
# 이후 두 배열의 원소를 첫번째 인덱스부터 차례로 확인하며
# A의 원소가 B의 원소보다 작을때에만 교체 수행
# 100,000=>NlogN needed!
n,k=map(int,input().split())
a=list(map(int,input().split()))
b=list(map(int,input().split()))

a.sort()
b.sort(reverse=True)

for i in range(k):
    if a[i]<b[i]:
        a[i],b[i]=b[i],a[i]
    else:
        break
print(sum(a))

#이진 탐색 알고리즘
#정렬된 리스트에서 탐색 범위를 절반씩 좁혀가며 데이터 탐색하는 방법
#이진탐색은 시작점, 끝점, 중간점을 이용하여 탐색 범위 설정함.
#순차탐색: 리스트 안에 있는 특정한 데이터를 찾기 위해 앞에서부터 데이터를 하나씩 확인하는 방법
#O(logN)을 보장함.
#이진탐색: 재귀적 구현
def binary_search(array,target,start,end):
    if start>end:
        return None
    mid=(start+end)//2 
    #찾은 경우 중간의 인덱스 반환
    if array[mid]==target:
        return mid
    #중간점의 값보다 찾고자 하는 값이 작은 경우 왼쪽 확인
    elif array[mid]>target:
        return binary_search(array,target,start,mid-1)
    #중간점의 값보다 찾고자 하는 값이 큰 경우 오른쪽 확인
    else:
        return binary_search(array,target,mid+1,end)

#n(원소의 개수)과 target(찾고자하는 값)을 입력 받기
n,target=list(map(int,input().split()))
#전체 원소 입력 받기
array=list(map(int,input().split()))
#이진 탐색 수행 결과 출력
result=binary_search(array,target,0,n-1)
if result==None:
    print("원소가 존재하지 않습니다")
else:
    print(result+1)

#파이썬 이진 탐색 라이브러리
from bisect import bisect_left,bisect_right
a=[1,2,4,4,8]
x=4
print(bisect_left(a,x)) #정렬된 순서를 유지하며 배열 a에 x를 삽입할 가장 왼쪽 인덱스
print(bisect_right(a,x)) #가장 오른쪽 인덱스를 반환.

#값이 특정 범위에 속하는 데이터개수 구하기(정렬된 상태를 기준으로 하여)
from bisect import bisect_left,bisect_right

def count_by_range(a,left_value,right_value):
    right_index=bisect_right(a,right_value)
    left_index=bisect_left(a,left_value)
    return right_index-leftleft_index

a=[1,2,3,3,3,3,4,4,8,9]
print(count_by_range(a,4,4)) #값이 4인 데이터 개수 출력하기.
print(count_by_range(a,-1,3)) #값이 [-1,3] 범위에 있는 데이터 개수 출력.
 
#파라메트릭 서치(이진탐색을 이용하여 효과적으로 해결할 수 있음)
#최적화 문제를 결정문제로 바꿔 해결하는 기법(예 혹은 아니요)

#떡볶이 떡 만들기
#떡의 총 길이는 절단기로 잘라서 맞춰줌.
#높이가 H보다 긴떡은 H위의 부분은 잘릴것이고 낮은떡은 안잘림
#예를 들어 높이가 19,14,10,17cm 떡이 나란히 있고 
#절단기 높이를 15cm로 지정하면, 자른뒤의 높이는 
#15,14,10,15가 되고, 잘린 떡의 길이는 차례대로 4,0,0,2cm이다.
#손님은 6cm만큼을 가져가게 된다.
#손님이 왔을 때 요청한 총 길이가 M일 때, 적어도 M만큼의 떡을 얻기 위해
#절단기에 설정할 수 있는 높이의 최대값을 구하는 프로그램??

#현재 이 높이로 자르면 조건을 만족할 수 있는가?
#조건의 만족 여부에 따라 탐색 범위를 좁혀서 해결할 수 있음.
#절단기의 높이는 0~1e9임

n,m=list(map(int,input().split()))
#떡개수n, 요청한 떡 길이m
#각 떡의 개별 높이 정보를 입력
array=list(map(int,input().split()))

#이진 탐색을 위한 시작점과 끝점 설정
start=0
end=max(array)

#이진 탐색 수행(반복적)
result=0
while(start<=end):
    total=0
    mid=(start+end)//2
    for x in array:
        #잘랐을떄의 떡의 양 계산
        if x>mid:
            total+=x-mid
        #떡의 양이 부족한 경우 더 많이 자르기(왼쪽 부분 탐색)
        if total<m:
            end=mid-1
        #떡의 양이 충분한 경우 덜 자르기(오른쪽 부분 탐색)
        else:
            result=mid #최대한 덜 잘랐을때가 정답이므로, 여기에서 result에 기록
            start=mid+1
    #정답출력
    print(result)

#정렬된 배열에서 특정 수의 개수 구하기
#{1,1,2,2,2,2,3}이 있을 때 x=2라면, 4를 출력하기
#O(logN)으로 설계할 것
#N은 1~1,000,000
#수열이 오름차순 정렬되어 있고, 만일 없다면 -1을 출력하기
#일반적으로 Linear Search는 시간 초과판정을 받게 된다.
#특정 값이 등장하는 첫번째 위치와 마지막 위치를 찾으면 됨.
from bisect import bisect_left,bisect_right

#값이 [left_value,right_value]인 데이터의 개수를 반환하는 함수
def count_by_range(array,left_value,right_value):
    right_index=bisect_right(array,right_value)
    left_index=bisect_left(array,left_value)
    return right_index-left_index

n,x=map(int,input().split())
array=list(map(int,input().split()))

#값이 [x,x] 범위에 있는 데이터의 개수 계산
count=count_by_range(array,x,x)
#값이 x인 원소가 존재하지 않는다면
if count==0:
    print(-1)
else: #값이 x인 원소가 존재한다면
    print(count)

#다이나믹 프로그래밍
#메모리를 적절히 사용하여 수행시간 효율성을 비약적 향상 시키는 방법
#이미 계산된 결과는 별도의 메모리 영역에 저장하여 다시 계산하지 않도록
#탑다운과 보텀업으로 구성된다.

#동적계획법: 프로그램이 실행되는 도중 실행에 필요한 메모리를 할당하는 기법

#최적 부분 구조(Optimal Substructure):
#큰 문제를 작은 문제로 나눌 수 있으며, 작은 문제의 답을 모아 큰 문제 해결

#중복되는 부분 문제(Overlapping Subproblem)
#동일한 작은 문제를 반복적으로 해결해야함.

#피보나치 수열 문제(점화식: 인접한 항들 사이의 관계식)
#1 1 2 3 5 8 13 21 34 55 89
#프로그래밍에서는 이러한 수열을 배열이나 리스트를 이용하여 표현한다.

def fibo(x): #단순 재귀 소스코드
    if x==1 or x==2:
        return 1
    return fibo(x-1)+fibo(x-2)
print(fibo(4)) #3
#단순 재귀함수는 지수시간 복잡도를 가지게 된다
#N이 조금만 커져도 기하급수적으로 커지게 된다.
#f(30): 약 10억가량의 연산을 수행해야한다.

#최적 부분 구조: 큰 문제를 작은 문제로 나눌 수 있음
#중복되는 부분 문제: 동일한 작은 문제를 반복적으로 해결한다.

#피보나치의 효율적 해법: 다이나믹 프로그래밍
#Memoization: 한 번 계산된 결과를 메모리 공간에 메모함(메모이제이션)
#같은 문제를 다시 호출시, 메모했던 결과를 그대로 가져옴
#값을 기록해 놓는다는 점에서 Caching이라고도 한다.

#top down(하향식)->재귀적 호출
#bottom up(상향식)->결과 저장용 리스트는 DP 테이블이라고 함
#엄밀하게 메모이제이션은 이전 계선 결과를 일시적으로 기록해 놓는 개념
#다이나믹 프로그래밍에 국한된 개념만은 아니다.


#피보나치 수열
#한번 계산된 결과를 메모이제이션하기 위한 리스트 초기화
d=[0]*100 #Memoization.

#피보나치 함수를 재귀함수로 구현(탑다운 다이나믹 프로그래밍)
def fibo(x):
    if x==1 or x==2:
        return 1
    #이미 계산한적 있는 문제라면 그대로 반환
    if d[x]!=0:
        return d[x]
    #아직 계산하지 않은 문제라면 점화식에 따라 피보나치 결과 반환
    d[x]=fibo(x-1)+fibo(x-2)
    return d[x]

print(fibo(99)) #218922995834555169026

#bottom up method
d=[0]*100

d[1]=1
d[2]=1
n=99

for i in range(3,n+1):
    d[i]=d[i-1]+d[i-2]
print(d[n]) # 답 출력하기.

#다이나믹 프로그래밍과 분할 정복의 차이점은 부분문제의 중복이다.
#다이나믹 프로그래밍 문제에서는 각 부분 문제들이 서로 영향을 미치며 부분문제가 중복된다.
#분할 정복 문제에서는 동일한 부분 문제가 반복적으로 계산되지 않는다.

#분할 정복(퀵 정렬)
#한번 기준원소가 자리를 변경해서 자리를 잡으면 그 기준 원소의 위치는 바뀌지 않는다.
#분할 이후에 해당 피벗을 다시 처리하는 부분 문제는 호출하지 않는다.
#동일 부분 문제가 반복적으로 계산되어지지 않는다.

#주어진 문제가 다이나믹 프로그래밍 유형인가?
#그리디, 구현, 완전탐색등의 아이디어로 문제를 해결할 수 있는가?
#재귀함수로 비효율적인 완전 탐색프로그램을 작성한후(탑다운)
#작은 문제에서 구한 답이 큰 문제에서 그대로 사용될 수 있으면
#코드를 개선하는 방법을 사용하자
#일반적인 코테에서는 기본 유형의 다이나믹 프로그래밍 문제가 출제됨.

#개미전사: 개미전사는 메뚜기 식량창고를 몰래 공격하려한다
#식량 창고는 일직선으로 이어져 있다.
#각 식량창고에는 정해진 수의 식량을 저장하고 있으며
#개미 전사는 식량 창고를 선택적으로 약탈하여 식량을 빼앗으려 한다.
#이때 메뚜기 정찰병들은 일직선상에 존재하는 식량창고 중 
#서로 인접한 식량창고가 공격받으면 바로 알아차린다.
#따라서 개미전사가 정찰병에게 들키지 않고, 식량창고를 약탈하기 위해
#최소한 한칸 이상 떨어진 식량 창고를 약탈해야 한다.

#ai=max(ai-1,ai-2+ki) #ki: i번째 식량창고에 있는 식량의 양

n=int(input())
#모든 식량 정보 입력 받기
array=list(map(int,input().split()))

#앞서 계산된 결과를 저장하기 위한 DP 테이블 초기화
d=[0]*100
#다이나믹 프로그래밍 진행(보텀업)
d[0]=array[0]
d[1]=max(array[0],array[1])
for i in range(2,n):
    d[i]=max(d[i-1],d[i-2]+array[i])
#계산된 결과 출력
print(d[n-1])

#1로 만들기: X가 5로 나눠떨어지면 5로 나눈다
#X가 3으로 나눠떨어지면 3으로 나눈다
#X가 2로 나눠떨어지면, 2로 나눈다
#X에서 1을 뺀다.
#26->25->5->1 (최소3번의 연산)

#1<=X<=30,000
#최적부분 구조와 중복되는 부분 문제를 만족한다.
#단순 그리디로 해결하기는 어려움.
#ai=min(ai-1,ai/2,ai/3,ai/5)+1
#단 1을 빼는 연산을 제외하고는 해당 수로 나눠떨어질 때에 한해 점화식 적용 가능

x=int(input())

d=[0]*30001 #앞서 계산된 결과를 저장하기 위한 DP 테이블 초기화

#다이나믹 프로그래밍 진행(보텀업)
for i in range(2,x+1):
    #현재의 수에서 1을 빼는 경우
    d[i]=d[i-1]+1
    #현재의 수가 2로 나눠떨어지는 경우
    if i%2==0:
        d[i]=min(d[i],d[i//2]+1)
    #현재의 수가 3으로 나누어 떨어지는 경우
    if i%3==0:
        d[i]=min(d[i],d[i//3]+1)
    #현재의 수가 5로 나누어 떨어지는 경우
    if i%5==0:
        d[i]=min(d[i],d[i//5]+1)

print(d[x])

#효율적인 화폐구성: M원을 만들기 위한 최소한의 화폐개수를 출력하라
#ai-k를 만드는 방법 존재하면 ai=min(ai,ai-k+1)
#ai-k를 만드는 방법이 존재하지 않으면 ai=INF
#ai: 금액 i를 만들수 있는 최소한의 화폐개수
#k: 각 화폐의 단위
#리스트를 지속적으로 갱신함.
n,m=map(int,input().split())
#N개의 화폐 단위 정보를 입력받기
array=[]
for i in range(n):
    array.append(int(input()))

#한번 계산된 결과를 저장하기 위한 DP 테이블 초기화
d=[10001]*(m+1)

#다이나믹 프로그래밍 진행(보텀업)
d[0]=0
for i in range(n):
    for j in range(arary[i],m+1):
        if d[j-array[i]]!=10001: #i-k원을 만드는 방법 존재할때
            d[j]=min(d[j],d[j-array[i]]+1)

#계산된 결과 출력
if d[m]==10001: #최종적으로 M원을 만드는 방법이 없는 경우
    print(-1)
else:
    print(d[m])

#금광 문제
#n x m 크기의 금광이 있는데, 금광은 1x1 크기의 칸으로 나눠져있따.
#각 칸은 특정 크기의 금이 들어있다.
#채굴자는 첫번째 열부터 출발하여 금을 캐기 시작한다.
#맨 처음에는 첫번째 열의 어느행에서든 출발할 수 있다
#이후, m-1번에 걸쳐서 매번 오른쪽 위, 오른쪽, 오른쪽 아래 3가지중 하나의
#위치로 이동해야한다.
#결과적으로 채굴자가 얻을 수 있는 금의 최대 크기를 출력하는 프로그램을 작성하라

#왼쪽 위에서 오는 경우, 왼쪽 아래에서 오는 경우, 왼쪽에서 오는 경우
#dp[i][j]=array[i][j]+max(dp[i-1][j-1],dp[i][j-1],dp[i+1][j-1])
#이때 리스트의 범위를 벗어나지 않는지 체크해야한다.
#바로 DP 테이블에 초기 데이터를 담아 다이나믹 프로그래밍을 적용할 수 있음.


#테스트 케이스(Test case)입력
for tc in range(int(input())):
    #금광 정보 입력
    n,m=map(int,input().split())
    array=list(map(int,input().split()))
    #다이나믹 프로그래밍을 위한 2차원 DP 테이블 초기화
    dp=[]
    index=0
    for i in range(n):
        dp.append(array[index:index+m])
        index+=m
    #다이나믹 프로그래밍 진행
    for j in range(1,m):
        for i in range(n):
            #왼쪽위에서 오는 경우
            if i==0: left_up=0
            else: left_up=dp[i-1][j-1]
            #왼쪽 아래에서 오는 경우
            if i==n-1:left_down=0
            else: left_down=dp[i+1][j-1]
            #왼쪽에서 오는 경우
            left=dp[i][j-1]
            dp[i][j]=dp[i][j]+max(left_up,left_down,left)
    result=0
    for i in range(n):
        result=max(result,dp[i][m-1])
    print(result)


#병사 배치하기:N명의 병사가 무작위 나열
#각 병사는 특정값의 전투력 보유하고 있음
#전투력이 높은 병사가 앞쪽에 오도록 내림차순 배치
#배치과정에서 특정한 위치에 있는 병사를 열외시키는 방법을 이용
#그러면서도 병사의 수가 최대가 되도록
# 15 11 4 8 5 2 4->15 11 8 5 4 (최대 5명)
#N<=2000 O(N^2)으로 설계할 것.

#가장 긴 증가하는 부분 수열(Longest Increasing Subsequence: LIS)
#가장 긴 감소하는 부분 수열을 찾는 문제로 치환할 수 있음.

#가장 긴 증가하는 부분 수열알고리즘을 쓰자.
#D[i]=max(D[i],D[j]+1) if array[j]<array[i]

n=int(input())
array=list(map(int,input().split()))
#순서를 뒤집어 '최장 증가 부분 수열' 문제로 변환한다.
array.reverse()

#다이나믹 프로그래밍을 위한 1차원 DP 테이블 초기화
dp=[1]*n
#가장 긴 증가하는 부분 수열(LIS)알고리즘 수행
for i in range(1,n):
    for j in range(0,i):
        if array[j]<array[i]:
            dp[i]=max(dp[i],dp[j]+1)
#열외해야 하는 병사의 최소 수를 출력한다.
print(n-max(dp))

#최단 경로 문제: 가장 짧은 경로를 찾는 알고리즘
#한 지점에서 다른 한 지점까지의 최단 경로
#한 지점에서 다른 모든 지점까지의 최단 경로
#모든 지점에서 다른 모든 지점까지의 최단 경로
#각 지점은 그래프에서 노드로 표현
#지점 간 연결된 도로는 그래프에서 간선으로 표현


#다익스트라 최단 경로 알고리즘 개요
#특정 노드에서 출발하여 다른 모든 노드로 가는 최단 경로 계산
#음의 간선이 없을때 정상적 동작(현실 세계의 도로는 음의 간선으로 표현X)
#그리디 알고리즘으로 분류: 매 상황에서 가장 비용이 적은 노드를 선택하여
#임의의 과정을 반복한다.

#출발 노드를 설정한다
#최단 거리 테이블 초기화
#방문하지 않은 노드중에서 최단거리가 가장 짧은 노드를 설정
#해당 노드를 거쳐 다른 노드로 가는 비용을 계산하여 최단 거리 테이블 갱신
#위 과정에서 3~4번 반복한다.

#그리디 알고리즘: 매 상황에서 방문하지 않은 가장 비용 적은 노드 선택
#단계를 거치며, 한번 처리된 노드의 최단 거리는 고정됨
#한단계당 하나의 노드에 대한 최단 거리를 확실히 찾을 수 있음
#테이블에 각 노드까지의 최단 거리 정보가 저장됨
#완벽한 형태의 최단 경로를 구하려면 소스에 추가 구현 필요

import sys
input=sys.stdin.readline
INF=int(1e9) #무한을 의미하는 값으로 10억을 설정

#노드의 개수, 간선의 개수를 입력받기
n,m=map(int,input().split())
#시작 노드 번호를 입력받기
start=int(input())
#각 노드에 연결되어 있는 노드에 대한 정보를 담는 리스트를 만들기
graph=[[] for i in range(n+1)]
#방문한 적이 있는지 체크하는 목적의 리스트를 만들기
visited=[False]*(n+1)
#최단 거리 테이블을 모두 무한으로 초기화
distance=[INF]*(n+1)

#모든 간선 정보를 입력받기
for _ in range(m):
    a,b,c=map(int,input().split())
    #a번 노드에서 b번 노드로 가는 비용이 c라는 의미
    graph[a].append((b,c))
#방문하지 않은 노드 중에서, 가장 최단 거리가 짧은 노드의 번호를 반환
def get_smalledst_node():
    min_value=INF
    index=0 #가장 최단 거리가 짧은 노드(인덱스)
    for i in range(1,n+1):
        if distance[i]<min_value and not visited[i]:
            min_value=distance[i]
            index=i
        return index

def dijkstra(start):
    #시작 노드에 대해서 초기화
    distance[start]=0
    visited[start]=True
    for j in graph[start]:
        distance[j[0]]=j[1]
    #시작 노드를 제외한 전체 n-1개의 노드에 대해 반복
    for i in range(n-1):
        #현재 최단 거리가 가장 짧은 노드를 꺼내서, 방문처리
        now=get_smallest_node()
        visited[now]=True
        #현재 노드와 연결된 다른 노드 확인
        for j in graph[now]:
            cost=distance[now]+j[1]
            #현재 노드를 거쳐서 다른 노드로 이동하는 거리가 더 짧을 때
            if cost<distance[j[0]]:
                distance[j[0]]=cost

    #다익스트라 알고리즘을 수행
    dijkstra(start)
    #모든 노드로 가기 위한 최단 거리를 출력
    for i in range(1,n+1):
        #도달 불가능한 경우, 무한(INFINITY)이라고 출력한다
        if distance[i]==INF:
            print("INFINITY")
        #도달 할 수 있는 경우 거리를 출력한다.
        else:
            print(distance[i])

#하지만 노드의 개수가 10,000개를 넘어간다면???
#우선순위 큐(Priority Queue)
#우선 순위가 가장 높은 데이터를 가장 먼저 삭제함.
#stack: 가장 나중에 삽입된 데이터가 먼저 추출
#queue: 가장 먼저 삽입된 데이터
#우선순위큐: 우선순위가 가장 높은 데이터가 먼저 출력됨.

#최소힙과 최대힙으로 구성된다.

#list->삽입시간(1), 삭제시간(N)
#heap->삽입시간(logN), 삭제시간(logN)

import heapq

#최소힙
#오름차순 힙 정렬(Heap Sort)
def heapsort(iterable):
    h=[]
    result=[]
    #모든 원소를 차례대로 힙에 삽입
    for value in iterable:
        heapq.heappush(h,value)
    #힙에 삽입된 모든 원소를 차례대로 꺼내 담기
    for i in range(len(h)):
        result.append(heapq.heappop(h))
    return result
result=heapsort([1,3,5,7,9,2,4,6,8,0])
print(result) # [0,1,2,3,4,5,6,7,8,9]

#최대힙
import heapq
def heapsort(iterable):
    h=[]
    result=[]
    #모든 원소를 차례대로 힙에 삽입
    for value in iterable:
        heapq.heappush(h,-value)
    #힙에 삽입된 모든 원소를 차례대로 꺼내 담기
    for i in range(len(h)):
        result.append(-heapq.heappop(h))
    return result

result=heapsort([1,3,5,7,9,2,4,6,8,0])
print(result) #[9,8,7,6,5,4,3,2,1,0]

#단계마다 방문하지 않은 노드중 최단거리가 가장 짧은 노드를 선택하기 위해
#heap 자료구조를 이용한다.

#다익스트라 알고리즘: 개선된 구현 방법
import heapq
import sys
input=sys.stdin.readline
INF=int(1e9)

n,m=map(int(input().split()))
#시작 노드 번호를 입력받기
start=int(input())
#각 노드에 연결되어 있는 노드에 대한 정보를 담는 리스트 만들기
graph=[[] for i in range(n+1)]
#최단 거리 테이블을 모두 무한으로 초기화
distance=[INF]*(n+1)

#모든 간선 정보를 입력받기
for _ in range(m):
    a,b,c=map(int,input().split())
    #a번 노드에서 b번 노드로 가는 비용이 c라는 의미
    graph[a].append((b,c))

def dijkstra(start):
    q=[]
    #시작 노드로 가기 위한 최단 경로는 0으로 설정하여 큐에 삽입
    heapq.heappush(q,(0,start))
    distance[start]=0
    while q: #큐가 비어있지 않으면,
        #가장 최단 거리가 짧은 노드에 대한 정보 꺼내기
        dist,now=heapq.heappop(q)
        #현재 노드가 이미 처리된 적이 있는 노드라면 무시
        if distance[now]<dist:
            continue
        #현재 노드와 연결된 다른 인접한 노드들을 확인
        for i in graph[now]:
            cost=dist+i[1]
            #현재 노드를 거쳐서, 다른 노드로 이동하는 거리가 더 잛은 경우
            if cost<distance[i[0]]:
                distance[i[0]]=cost
                heapq.heappush(q,(cost,i[0]))

#다익스트라 알고리즘을 실행
dijkstra(start)

#모든 노드로 가기 위한 최단 거리를 출력
for i in range(1,n+1):
    #도달 할 수 없는 경우, 무한(INFINITY)이라고 출력
    if distance[i]==INF:
        print("INFINITY")
    else:
        print(distance[i]) # 도달 할 수 있는 경우 거리를 출력한다.

#최대 간선의 개수 E ->ElogE만큼의 시간 복잡도를 가진다.
#(중복 간선을 포함하지 않는 경우이다)

#플로이드 워셜 알고리즘.
#모든 노드에서 다른 모든 노드까지의 최단 경로를 모두 계산
#단계별로 거쳐가는 노드를 기준으로 알고리즘 수행
#매 단계마다 방문하지 않은 노드 중 최단거리를 갖는 노드찾는 과정 필요 X
#플로이드 워셜은 2차원 테이블에 최단 거리 정보를 저장한다.
#플로이드 워셜 알고리즘은 다이나믹 프로그래밍 유형에 속한다.

#Dab=min(Dab,Dak+Dkb)
#a~b로 가는 최단거리보다, a에서 k를 거쳐 b로 가는거리가 더 짧은지 검사한다.

INF=int(1e9) #무한을 의미하는 값으로 10억을 설정

#노드의 개수 및 간선 개수 입력받기
n=int(input())
m=int(input())
#2차원 리스트(그래프 표현)를 만들고, 무한으로 초기화
graph=[[INF]*(n+1) for _ in range(n+1)]

#자기자신에서 자기자신으로 가는 비용은 0으로 초기화
for a in range(1,n+1):
    for b in range(1,n+1):
        if a==b:
            graph[a][b]=0

#각 간선에 대한 정보를 입력받아, 그 값으로 초기화
for _ in range(m):
    #A에서 B로 가는 비용은 C라고 설정
    a,b,c=map(int,input().split())
    graph[a][b]=c

#점화식에 따라 플로이드 워셜 알고리즘을 수행
for k in range(1,n+1):
    for a in range(1,n+1):
        for b in range(1,n+1):
            graph[a][b]=min(graph[a][b],graph[a][k]+graph[k][b])

#수행된 결과를 출력
for a in range(1,n+1):
    for b in range(1,n+1):
        #도달 불가능한 경우, 무한이라고 출력
        if graph[a][b]==INF:
            print("INFINITY",end=" ")
        else: #도달 가능한 경우 거리를 출력
            print(graph[a][b],end=" ")
    print()
#시간 복잡도는 N^3이다.

#전보: 어떤 나라에는 N개의 도시가 있다.
#그리고 각 도시는 보내고자 하는 메시지가 있는 경우,
#다른 도시로 전보를 보내 다른 도시로 해당 메시지를 전송할 수 있다.
#하지만 X라든 도시에서 Y라는 도시로 전보를 보내고자 한다면,
#도시 X에서 Y로 향하는 통로가 설치되어 있어야 한다
#예를 들어 X에서 Y로 향하는 통로는 있지만
#Y에서 X로 향하는 통로가 없다면 Y는 x로 메세지 보낼 수 없다.
#또한 통로를 거쳐 메시지를 보낼때는 일정 시간이 소요된다.
#어느날 c라는 도시에서 위급 상황이 발생해 최대한 많은 도시로 메시지 보내려한다.
#메시지는 도시 c에서 출발하여 각 도시사이에 설치된 통로를 거쳐, 최대한 많이 퍼저나갈 것이다.
#각 도시의 번호와 통로가 설치되어 있는 정보가 주어질 때,
#도시 c에서 보낸 메세지를 받게 되는 도시의 개수는 총 몇개이며
#도시들이 모두 메시지를 받는 데까지 걸리는 시간은 얼마인가???

import heapq
import sys
input=sys.stdin.readline
INF=int(1e9)

def dijkstra(start):
    q=[]
    #시작노드로 가기 위한 최단 거리는 0으로 설정하여, 큐에 삽입
    heapq.heappush(q,(0,start))
    distance[start]=0
    while q: #큐가 비어있지 않다면
        #가장 최단 거리가 짧은 노드에 대한 정보를 꺼내기
        dist,now=heapq.heappop(q)
        if distance[now]<dist:
            continue
        #현재 노드와 연결된 다른 인접한 노드들을 확인
        for i in graph[now]:
            cost=dist+i[1]
            #현재 노드를 거쳐서, 다른 노드로 이동하는 거리가 더 짧을 때
            if cost<distance[i[0]]:
                distance[i[0]]=cost
                heapq.heappush(q,(cost,i[0]))

n,m,start=map(int,input().split())
#각 노드에 연결되어 있는 노드에 대한 정보를 담는 리스트를 만들기
graph=[[] for i in range(n+1)]
#최단 거리 테이블을 모두 무한으로 초기화
distance=[INF]*(n+1)

#모든 간선 정보를 입력받기
for _ in range(m):
    x,y,z=map(int,input().split())
    #X번 노드에서 Y번 노드로 가는 비용이 Z라는 의미
    graph[x].append((y,z))

#다익스트라 알고리즘을 수행
dijkstra(start)
#도달할 수 있는 노드의 개수
count=0
#도달할 수 있는 노드중에서, 가장 멀리 있는 노드와의 최단거리
max_distance=0
for d in distance:
    #도달할 수 있는 노드인 경우
    if d!=1e9:
        count+=1
        max_distance=max(max_distance,d)

#시작노드는 제외해야 하므로 count-1을 출력
print(count-1,max_distance)

#8,9,10,11,12,13,14,15
