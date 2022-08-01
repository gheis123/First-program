#사탕 먹기 게임
#첫째줄에 보드의 크기N (3~50)이 주어진다.
#다음 N개 줄에는 보드에 채워져 있는 사탕의 색상이 주어진다.
#빨:C, 파:P, 초:Z, 노:Y
#사탕의 위치를 서로 교환가능=>먹을 수 있는 행 또는 열에서의 색깔 같은 사탕 maximum?????

#ex)
#3[3]          4[4]        5[4 by c] 
#CCP           PPPP         YCPZY
#CCP           CYZY         CYZZP 
#PPC           CCPY         CCPPP
#              PPCC         YCYZC
#                           CPPZZ

N=int(input("Input your N:"))
board=[]
for _ in range(N):
    split_list=list(input())
    board.append(split_list)

print(board) #PPC PCP CPC 입력하면 
#board=[['P','P','C'],['P','C','P'],['C','P','C']] =>이런 형식으로 저장된다.
ans=1 #minimum은 1일 수 밖에 없다.

def search():
    global ans
    for i in range(N): #first case
        cnt=1 #다시 행이 다른 숫자로 변하기 시작하면 1로 초기화시키기.
        for j in range(1,N): #i=0이라면, b[0][0]~b[0][N-1]까지 탐구함.
            #이것을 열의 변화라고 해석할 수 있다.
            if board[i][j-1]==board[i][j]:
                cnt+=1 #하나가 같을 때마다 counting++
                ans=max(ans,cnt) #주기적 update
            else:
                cnt=1 #아무것도 같은 것이 없다면 1!

    for j in range(N): #second case
        cnt=1
        for i in range(1,N): #j=0이라면 b[0][0]~b[N-1][0] =>행의 변화.
            if board[i-1][j]==board[i][j]:
                cnt+=1
                ans=max(ans,cnt)
            else:
                cnt=1


for i in range(N):
    for j in range(N):
        if j+1<N: #i=0이라면 b[0][j],b[0][j+1]=b[0][j+1],b[0][j]
            #즉, 열과의 관계가 변하는 것 
            #but, j+1이 N이 되어서는 안된다. 따라서 j+1<N[열의 변화이다]
            board[i][j],board[i][j+1]=board[i][j+1],board[i][j]
            search()
            board[i][j],board[i][j+1]=board[i][j+1],board[i][j] #원상복구

        if i+1<N: #이것은 행의 변화를 뜻하는 것이다.
            board[i][j],board[i+1][j]=board[i+1][j],board[i][j]
            search()
            board[i][j],board[i+1][j]=board[i+1][j],board[i][j] #원상복구

print(ans)



#Greedy Algorithm : 모든 경우를 보지않기 떄문에 완전탐색보다 빠르다.
#문제에서 규칙성만 찾으면 된다. 
#coin problem
#(10,50,100,500) 800=500+100*3 =>총4개가 필요하다
#(100,400,500) 800=500+100*3 (4개) but 반례, 400*2(2개)
#반례의 이유: 작은단위가 그것보다 큰 단위에 대해 항상 배수 관계가 아니므로


#회의실 배정
#한개의 회의실이 있는데 이를 사용하고자 하는 N개의 회의에 대해
#시간표를 만드는데, 시작시간과 종료시간이 주어진다. 
#겹치지 않게 하면서 회의실 사용할 수 있는 회의 최대의 개수를 return?
#회의가 시작되면 종료될 수는 없다. 
#회의의 시작시간과 종료시간이 같을 수도 있음.
#이 경우는 시작하자마자 종료된다고 볼 수 있음.

#회의의수는 1<=N<=100,000 given
#공백을 사이에 두고 회의의 시작시간과 종료시간이 주어진다. 

#example input
#11입력했다고 가정했을때
#list=[[1,4],[3,5],[0,6],[5,7],[3,8],[5,9],[6,10],[8,11],[8,12],[2,13],[12,14]]
#(1,4),(5,7),(8,11),(12,14)를 이용하면 총 4개이다.

#탐욕법으로 이 문제를 풀어나가 보자.
#모든 경우를 살피는 완전탐색 가능? 회의실 사용가능과 불가능 2^K 불가능....
#1: 회의시간이 짧은 순서? 
#가장 짧은 회의를 고르면 1개이지만, 긴것 2개 고르면 2개이다.

#2: 회의시간을 정렬해볼까?
#시작시간이 0인 회의가 유일하게 하나 존재한다고 하면 only 1이지만
#0초과 부터 시작하는 작은 덩어리가 여러가지 있으면 모순이다.

#핵심: 그리디를 사용할때는 반례를 꼼꼼하게 따져야 한다.

#3: 이번에는 종료시간을 기준으로 정렬해볼까?
#종료시간이 같은 경우로 sorting=>일단 시작하자마자 끝나는 것 무조건 선택
#종료시간이 가급적 더 빠른 회의를 고르는게 좋은 전략이다.

#만일, 종료시간이 가장 빠른 회의를 A라고 하자
#A회의를 선택하지 않았을때 최적해가 존재한다고 가정하자.
#최적해중 가장 첫번째로 종료되는 회의를 B라고 하자
#반드시, A회의 종료시간이 B종료시간보다 빠르다.
#B회의를 제외한 나머지 회의는 A회의와 겹칠 수 없다.
#A회의를 골라서 최적해가 아닌 것은 없다.

import sys

input=sys.stdin.readline
meetings=[]
for _ in range(int(input())):
    start,end=map(int,input().split())
    meetings.append((end,start))

meetings.sort() #ending 시점을 기준으로 sortings... 
t=0
ans=0
for end,start in meetings:
    if t<=start:
        ans+=1
        t=end
print(ans)

#수리공항승
#N,L==>N은 물이 새는 곳이 몇개인지, L은 테이프의 길이이다.
#수리공은 생각한다 0.5간격에는 적어도 간격을 줘야 물이 새지 않는다!
#첫줄은 N,L을 입력받고
#둘째줄은 물이 새는 위치를 split()으로 입력받는다.

coord=[False]*1001
N,L=map(int,input().split())
for i in map(int,input().split()):
    coord[i]=True

ans=0
x=0
while x<=1000:
    if coord[x]:
        ans+=1
        x+=L #왼쪽부터 x좌표를 추가해나감. 그 이후로 탐색.
    else:
        x+=1
print(ans)



#___________________________________________________________________________________#
#완전탐색에서의 DFS[stack],BFS[Queue],백트래킹.
#graph,tree: 필수적인 개념
#역(Node,Vertex), 역을 잇는 노선은 (Edge)
#그래프의 방향성: 방향있음, 무방향그래프(방향이 없거나 양방향 그래프이다)
#방향성이 없다는건, 어느 쪽이든 오갈 수 있다는 뜻이므로, 양방향이나 일반 직선과는 같은 개념
#(수학적인 지식으로 보자면 직선과 선분의 개념으로 이해하는 것이 좋다)

#graph: 순환 그래프와 비순환 그래프로 나뉜다.
#Cyclic Graph, ACyclic Graph
#단 하나라도, 순환하는 부분이 있다면 Cyclic Graph이다.

#git: 방향성 비순환 그래프->Version Control System: 
#시간이 지나며 기록이 쌓이며, 시간은 한 방향으로만 흐른다.
#사이클이 절대 발생하지 않는 구조

#만일 노드(역)이 5개인데, 간선이 0개인 그래프가 있다면 연결요소가 5개인 그래프이다.
#기억하자! 완전히 분리된 요소들이 여럿 주어지는 그래프도 있다(사실은 하나의 그래프이다)

#Tree: 순환성이 없는 무방향 그래프
#트리의 가장 바깥쪽 노드를 leaf node라고 한다(간선이 하나만 연결된 노드)
#서로 다른 노드의 경로는 무조건 하나로 유일해야 한다.
#num[node]=num[Edge]+1
#트리는 루트노드가 하나이며, 부모-자식 관계가 존재한다
#상위 노드가 부모, 하위노드가 자식에 해당한다.

#2차원 Matrix
#Adjacency Matrix (방향 그래프의 인접행렬)
#특징은, 방향성이 있다는 것 Aij!=Aji일수 있다는 것

#무방향 그래프는 Aii==0
#Aij=Aji 관계가 항상 성립한다. 마치 직선처럼 고려할 수 있으므로

#인접 리스트를 구하는 방식도 꼭 알아두자.

#노드가 N개 일 때, 인접 행렬은 N^2의 공간을 할당한다.
#인접 리스트는 인접행렬보다 메모리를 덜 차지하는 장점이 있다.
#보통 인접 행렬로 구현할때가 많은 편에 속한다.

#DFS[Depth First Search] ->깊이 우선 탐색
#정답이 되는 노드를 찾을때까지 탐색을 지속적으로 진행한다.

#BFS[Breadth First Search]->너비 우선 탐색->큐로 구현한다.
#시작노드에서 목표 노드까지 최단거리를 구할 때 주로 사용하게 된다.
#제일 짧은 거리부터 하나씩 늘려가며 도달 가능 모든 노드를 탐색
#최초로 목표 도달했을 때 최단거리임이 보장된다.
#일반적으로 DFS보다는 BFS를 많이 사용한다.


#정점의 개수 V, 간선의 개수 E
#시간복잡도 
#1.인접행렬:   O(V^2)
#2.인접리스트: O(V+E)
#정점의 개수가 많을수록 O(V) 
#간선의 개수가 많을수록 O(V+E)

#from collections import deque

#dy=(0,1,0,-1)
#dx=(-1,0,1,0)
#N=int(input())
#chk=[[False]*N for _ in range(N)]

#def is_valid_coord(y,x): #내가 입력한 숫자만큼의 범위안에 속하는지?????
#    return 0<=y<N and 0<=x<N

#def dfs(y,x):
#    if adj[y][x]==ans:
#        return 

#    for k in range(4):
#        #(y,x)=(1,1) 가정하자.
#        #k==0 (ny,nx)=(1,0)
#        #k==1 (ny,nx)=(2,1)
#        #k==2 (ny,nx)=(1,2)
#        #k==3 (ny,nx)=(0,1)
#        #AH~ 한 point를 기점으로 상하좌우 1만큼을 더 탐구하는 것임! 
#        ny=y+dy[k] #dy=(0,1,0,-1) 
#        nx=x+dx[k] #dx=(-1,0,1,0)
#        if is_valid_coord(ny,nx) and not chk[ny][nx]:
#            chk[ny][nx]=True
#            dfs(ny,nx) #재귀함수의 호출.

#def bfs(sy,sx): #start y,x
#    q=deque() #queue created!
#    chk[sy][sx]=True
#    q.append((sy,sx))
#    while len(q):
#        y,x=q.popleft()
#        if adj[y][x]==ans:
#            return
#        for k in range(4):
#            ny=y+dy[k]
#            nx=x+dx[k]
#            if is_valid_coord(ny,nx) and not chk[ny][nx]:
#                chk[ny][nx]=True
#                q.append(ny,nx)


#example (연결요소의 개수)
#방향 없는 그래프가 주어졌을때, 연결요소의 개수 구하는 program
#첫줄에 정점개수와 간선개수가 주어진다(N,M)
#1<=N<=1000, 0<=M<=N*(N-1)/2
#둘쨰줄에 간선의 양끝점 u와 v가 주어진다.

import sys

sys.setrecursionlimit(10**6) #1000번의 재귀 제한을 푸는 code
input=sys.stdin.readline
print("N값과 M값을 한번에 space 고려하여 숙자로 입력하세요:")
N,M=map(int,input().split())

adj=[[False]*(N+1) for _ in range(N+1)] #N+1 * N+1 Matrix 생성!
print("초기의 상태 Matrix는 아래와 같다.")
print(adj)

for _ in range(M):
    print("간선의 연결상태를 알려주세요.{}번!".format(M))
    a,b=map(int,input().split())
    adj[a][b]=True
    adj[b][a]=True
print("간선을 연결한 후 Matrix의 상태는 아래와 같다.")
print(adj)

ans=0
chk=[False]*(N+1) #노드 번호가 1부터 시작하기 때문에 모든 공간을 N+1로 저장한다.

def dfs(i):
    for j in range(1,N+1):
        if adj[i][j] and not chk[j]:
            chk[j]=True
            dfs(j) #재귀함수 호출.

for i in range(1,N+1): #N=6이라면 총 6번 repeat!
    if not chk[i]:
        ans+=1
        chk[i]=True
        dfs(i)
#chk[1]가 False라면, ans를 1 증가시키고 dfs(1)함수 호출
#dfs실행이되면, 총 6번 반복한다 
#adj[1][1]이 True이고 chk[1]이 False라면 chk[1]을 True로 고친다.
#......#
#adj[1][6]이 True이고 chk[6]이 False라면 chk[6]을 True로 고친다.

#chk[2]이 False라면, ans를 1 증가시키고 dfs(2)함수 호출.
print(ans)

