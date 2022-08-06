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

N=int(input())
b=[list(input()) for _ in range(N)]
ans=1

def search(): #user 입력에 따라서 N*N 정사각행렬이 만들어진다.
    global ans 
    #[0][0]~[0][N-1] ->열의 변화(행->열)
    #[0][0]~[N-1][0] ->행의 변화(열->행)
    for i in range(N):
        cnt=1 #한번씩 변경될 때마다 cnt의 값을 바꿔준다.
        for j in range(N-1):
            if b[i][j]==b[i][j+1]:
                cnt+=1
                ans=max(cnt,ans)
    for j in range(N):
        cnt=1 #한번씩 변경될 때마다 cnt의 값을 바꿔준다.
        for i in range(N-1):
            if b[i][j]==b[i+1][j]:
                cnt+=1
                ans=max(cnt,ans)

#행렬을 바꿕면서 탐색한다.
for i in range(N):
    for j in range(N):
        if j+1<N:
            b[i][j],b[i][j+1]=b[i][j+1],b[i][j]
            search()
            b[i][j],b[i][j+1]=b[i][j+1],b[i][j] #다시 원상복귀
        if i+1<N:
            b[i][j],b[i+1][j]=b[i+1][j],b[i][j]
            search()
            b[i][j],b[i+1][j]=b[i+1][j],b[i][j]
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


#__________________________________________#

time_list=[[1,4],[3,5],[0,6],[5,7],[3,8],[5,9],[6,10],[8,11],[8,12],[2,13],[12,14]]
for i in range(len(time_list)):
    time_list[i][0],time_list[i][1]=time_list[i][1],time_list[i][0] #start지점과 ending지점을 바꾼다.

time_list.sort() #끝나는 순서로 sorting.

#순서는 [end,start]
cnt=1
start,end=time_list[0][1],time_list[0][0]
print(start,end)
for idx in range(len(time_list)-1):
    if end<=time_list[idx+1][1]: #starting point에 대한 정보를 update
       cnt+=1
       end=time_list[idx+1][0] #ending에 대한 정보를 update
print(cnt)


#수리공항승
#N,L==>N은 물이 새는 곳이 몇개인지, L은 테이프의 길이이다.
#수리공은 생각한다 0.5간격에는 적어도 간격을 줘야 물이 새지 않는다!
#첫줄은 N,L을 입력받고
#둘째줄은 물이 새는 위치를 split()으로 입력받는다.

coord=[0]*1001
print("물 새는 곳의 개수와 테이프의 길이를 순서대로 입력하시오")

N,L=map(int,input().split()) #N:물 새는 곳의 개수, L:tape length
print("정확히 어디에 물이 새는지 알려줘! 개수는 {}개".format(N))

for i in map(int,input().split()): #정확하게 물이 새는 곳만 1로 변경시켜줌.
    coord[i]=1

cnt=0
x=0
while x<=1000:
    if coord[x]==1:
        cnt+=1
        x+=L #왼쪽부터 x좌표를 추가해나감. 그 이후로 탐색.
    else:
        x+=1
print(cnt)



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



#__________________________________________________________________________________#
#미로탐색 예제
#N*M 크기로 표현되는 미로가 있다.
#미로에서 1은 이동할 수 있는 칸이고, 0은 이동 불가능한 칸이다.
#시작지점과 도착지점 역시 counting 해준다. 

#첫째줄에 두정수 N,M (2<=N, M<=100)이 주어진다.
#(1,1)->(N,M)까지 지나야 하는 최소의 칸수를 return 하도록 programming하라.

#최단거리를 구하는 문제이므로 BFS를 사용하는 것이 좋다.
from collections import deque

#tuple 형식으로 상하좌우 생성하기.
dx=(1,0,-1,0)
dy=(0,1,0,-1) 

N,M=map(int,input("N,M의 값을 차례로 space 포함하여 숫자로 입력하시오:").split())
board=[input() for _ in range(N)] #list안의 list로 구성되어져 있는 이중 list구조
chk=[[False]*M for _ in range(N)] #list안의 list 
#만일 4,6으로 입력했으면 [F,F,F,F,F,F]가 총 4개 있다는 뜻.
dq=deque()
dq.append((0,0,1))
chk[0][0]=True
print("board matrix is that {}".format(board))
print("chk matrix is that {}".format(chk))
#내가 설정한 범위 내에 들어있는가????
def is_valid_coord(y,x): 
    return 0<=y<N and 0<=x<M

while len(dq)>0:
    print(dq)
    y,x,d=dq.popleft()
    print(dq)
    if y==N-1 and x==M-1: #last case.
        print(d)
        break

    for k in range(4):
        ny=y+dy[k]
        nx=x+dx[k]
        nd=d+1
        if is_valid_coord(ny,nx) and board[ny][nx]=='1' and not chk[ny][nx]:
            chk[ny][nx]=True
            dq.append((ny,nx,nd))

print("last chk matrix is that {}".format(chk))



#_____________________________________________#
#back-Tracking
#진행과정에서 답이 아닌 분기를 만나면 탐색을 진행하지 않고, 다른 분기로 가서 가지치기함.

#Alphabet example
#세로 R칸, 가로 C칸으로 된 표 모양 보드가 있다.
#보드의 각 칸에는 대문자 알파벳이 하나씩 적혀있다.
#(1,1)에 말이 놓여있는데, 이미 같은 알파벳을 지나왔다면 그 칸은 지나갈 수 없다.
#말이 최대한 몇칸을 갈 수 있을지 탐구해보라.

#1<=R, C<=20
#말이 지나갈 수 있는 최대의 칸을 출력한다면???

from collections import deque

dy=(0,1,0,-1)
dx=(1,0,-1,0)

R,C=map(int,input("Input your R,C:").split()) 
#Goal: R*C matrix create!
board=[input() for _ in range(R)]
chk=[[set() for _ in range(C)] for _ in range(R)]
ans=0

def is_vaild_coord(y,x):
    return 0<=y<R and 0<=x<C

dq=deque()
chk[0][0].add(board[0][0])
dq.append((0,0,board[0][0]))
while dq:
    print("dq: {}".format(dq))
    print("chk: {}".format(chk))
    print("board: {}".format(board))
    print("\n\n\n")
    y,x,s=dq.popleft()
    ans=max(ans,len(s))

    for k in range(4):
        ny=y+dy[k]
        nx=x+dx[k]
        if is_vaild_coord(ny,nx) and board[ny][nx] not in s:
            ns=s+board[ny][nx]
            if ns not in chk[ny][nx]:
                chk[ny][nx].add(ns)
                dq.append((ny,nx,ns))

print(ans)

#____________________________________________________________________#
##선형탐색(Sequential search): 반복문을 돌려 하나하나 비교하면서 찾아나감
##example
#from random import *
#num_list=[]
#for _ in range(5):
#    num_list.append(randint(1,50))

#stored_num=num_list[0]
#shuffle(num_list)

#for index in range(len(num_list)):
#    if num_list[index]==stored_num:
#        print("현재의 list 상태:{}".format(num_list))
#        print("{}은(는) {}번에 존재하고 있습니다".format(stored_num,index))
#        break


##이분탐색: 구현이 복잡하지만 더 빠른 알고리즘. (탐색할 부분이 하나 남을 때까지)
##Complexity: O(logN) 밑은 2만큼의 fast Algorithm

#a=[2,1,3,6,6,8,12]
#a.sort() #반드시 순서대로 정렬해줘야 한다.
#left=0
#right=len(a)-1 #총 7개의 데이터->right=6으로 setting.
#mid=(left+right)//2
#while left<=right:
#    if a[mid]==3:
#        print("{}번째 index에 위치하고 있습니다.".format(mid))
#        break
#    elif a[mid]>3:
#        right=mid-1 #왼쪽에서 탐구하자!
#    else:
#        left=mid+1 #오른쪽에서 탐구하자!
#    mid=(left+right)//2

##훨씬 쉽게 이를 구현하는 방법을 살펴보자.
#from bisect import bisect_left,bisect_right
#a=[2,3,6,6,6,10,12,15] 
#l=bisect_left(a,6)
#r=bisect_right(a,6)
#print(r-l) #6이 몇번 counting되는지 살펴보는 코드.

##Parametric Search(매개변수탐색)
##1.최적화문제: 문제상황 만족하는 변수 최소,최대를 구하는 문제
##2.결정문제: YES,NO로 답할 수 있는 문제

##나무자르기 문제
##상근이는 나무 M미터가 필요하다.
##높이 setting(양의 정수 or 0) 톱날이 땅으로부터 H만큼 위로 올라간다.
##(20,15,10,17)->15미터로 지정했다면->(15,15,10,15)
##나무를 필요한 만큼만 집으로 가져가려한다.
##적어도 M미터의 나무를 집에 가져가기 위해 절단기에 설정할 수 있는
##높이의 최대값을 return 하는 프로그램을 작성하시오.

##나무의 수: N (1<=N<=1,000,000)
##집으로 가져갈 나무의 길이: M(1<=M<=2,000,000,000)
##나무의 길이가 주어짐.

##example1
##나무의수 4, 집으로 가져갈 나무의 길이 7
##20 15 10 17=>15

##example2
##나무의수 5, 집으로 가져갈 나무의 길이:20
##4 42 40 26 46=>36

#print("나무의수N과 집으로가져갈 나무의 길이M을 입력하시오:")
#N,M=map(int,input().split())
#print("나무의 상황입력하시오:{}개 parameter needed!".format(N))
#tree=list(map(int,input().split()))
#lo=0
#hi=max(tree) #list내의 모든 원소의 sum.
#mid=(lo+hi)//2 #몫의 개념을 사용해줘야 한다.

#def get_total_tree(h): #모든 나무에 대해 자른 후의 값들의 sum.
#    ret=0
#    for t in tree:
#        if t>h:
#            ret+=t-h
#    return ret

#ans=0
#while lo<=hi:
#    if get_total_tree(mid)>=M:
#        ans=mid
#        lo=mid+1
#    else:
#        hi=mid-1
#    mid=(lo+hi)//2

#print(ans)


#숫자 카드게임
#숫자카드는 정수 하나가 적혀있다.
#상근이는 N개의 카드를 가지고 있다.
#10
#6 3 2 10 10 10 -10 -10 7 3
#8
#10 9 -5 2 3 4 5 -10
num=int(input("Input your number:"))
num_list=list(map(int,input().split()))
second_num=int(input("Input your second number:"))
result=list(map(int,input().split()))
print(num_list,result)
cnt=0

for index in result:
    if index not in num_list:        
        result[cnt]=0
        cnt+=1
    else:
        result[cnt]=num_list.count(index)
        cnt+=1
print(result)


#다른 방식으로 구현하기
from bisect import bisect_left,bisect_right

N=int(input())
cards=sorted(map(int,input().split()))
M=int(input())
ans=[]
for i in map(int,input().split()):
    ans.append(bisect_right(cards,i)-bisect_left(cards,i))
print(' '.join(map(str,ans)))
