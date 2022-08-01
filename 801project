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
